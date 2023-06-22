"""
Inference model of SuperPoint, a feature detector and descriptor.

Described in:
    SuperPoint: Self-Supervised Interest Point Detection and Description,
    Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich, CVPRW 2018.

Original code: github.com/MagicLeapResearch/SuperPointPretrainedNetwork
"""

import torch
from torch import nn

from .. import GLUESTICK_ROOT
from ..models.base_model import BaseModel


def simple_nms(scores, radius):
    """Perform non maximum suppression on the heatmap using max-pooling.
    This method does not suppress contiguous points that have the same score.
    Args:
        scores: the score heatmap of size `(B, H, W)`.
        size: an interger scalar, the radius of the NMS window.
    """

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=radius * 2 + 1, stride=1, padding=radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, b, h, w):
    mask_h = (keypoints[:, 0] >= b) & (keypoints[:, 0] < (h - b))
    mask_w = (keypoints[:, 1] >= b) & (keypoints[:, 1] < (w - b))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s):
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(BaseModel):
    default_conf = {
        'has_detector': True,
        'has_descriptor': True,
        'descriptor_dim': 256,

        # Inference
        'return_all': False,
        'sparse_outputs': True,
        'nms_radius': 4,
        'detection_threshold': 0.005,
        'max_num_keypoints': -1,
        'force_num_keypoints': False,
        'remove_borders': 4,
    }
    required_data_keys = ['image']

    def _init(self, conf):
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        if conf.has_detector:
            self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
            self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        if conf.has_descriptor:
            self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
            self.convDb = nn.Conv2d(
                c5, conf.descriptor_dim, kernel_size=1, stride=1, padding=0)

        path = GLUESTICK_ROOT / 'resources' / 'weights' / 'superpoint_v1.pth'
        self.load_state_dict(torch.load(str(path)), strict=False)

    def _forward(self, data):
        image = data['image']
        if image.shape[1] == 3:  # RGB
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)

        # Shared Encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        pred = {}
        if self.conf.has_detector and self.conf.max_num_keypoints != 0:
            # Compute the dense keypoint scores
            cPa = self.relu(self.convPa(x))
            scores = self.convPb(cPa)
            scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
            b, c, h, w = scores.shape
            scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
            scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
            pred['keypoint_scores'] = dense_scores = scores
        if self.conf.has_descriptor:
            # Compute the dense descriptors
            cDa = self.relu(self.convDa(x))
            all_desc = self.convDb(cDa)
            all_desc = torch.nn.functional.normalize(all_desc, p=2, dim=1)
            pred['descriptors'] = all_desc

            if self.conf.max_num_keypoints == 0:  # Predict dense descriptors only
                b_size = len(image)
                device = image.device
                return {
                    'keypoints': torch.empty(b_size, 0, 2, device=device),
                    'keypoint_scores': torch.empty(b_size, 0, device=device),
                    'descriptors': torch.empty(b_size, self.conf.descriptor_dim, 0, device=device),
                    'all_descriptors': all_desc
                }

        if self.conf.sparse_outputs:
            assert self.conf.has_detector and self.conf.has_descriptor

            scores = simple_nms(scores, self.conf.nms_radius)

            # Extract keypoints
            keypoints = [
                torch.nonzero(s > self.conf.detection_threshold)
                for s in scores]
            scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

            # Discard keypoints near the image borders
            keypoints, scores = list(zip(*[
                remove_borders(k, s, self.conf.remove_borders, h * 8, w * 8)
                for k, s in zip(keypoints, scores)]))

            # Keep the k keypoints with highest score
            if self.conf.max_num_keypoints > 0:
                keypoints, scores = list(zip(*[
                    top_k_keypoints(k, s, self.conf.max_num_keypoints)
                    for k, s in zip(keypoints, scores)]))

            # Convert (h, w) to (x, y)
            keypoints = [torch.flip(k, [1]).float() for k in keypoints]

            if self.conf.force_num_keypoints:
                _, _, h, w = data['image'].shape
                assert self.conf.max_num_keypoints > 0
                scores = list(scores)
                for i in range(len(keypoints)):
                    k, s = keypoints[i], scores[i]
                    missing = self.conf.max_num_keypoints - len(k)
                    if missing > 0:
                        new_k = torch.rand(missing, 2).to(k)
                        new_k = new_k * k.new_tensor([[w - 1, h - 1]])
                        new_s = torch.zeros(missing).to(s)
                        keypoints[i] = torch.cat([k, new_k], 0)
                        scores[i] = torch.cat([s, new_s], 0)

            # Extract descriptors
            desc = [sample_descriptors(k[None], d[None], 8)[0]
                    for k, d in zip(keypoints, all_desc)]

            if (len(keypoints) == 1) or self.conf.force_num_keypoints:
                keypoints = torch.stack(keypoints, 0)
                scores = torch.stack(scores, 0)
                desc = torch.stack(desc, 0)

            pred = {
                'keypoints': keypoints,
                'keypoint_scores': scores,
                'descriptors': desc,
            }

            if self.conf.return_all:
                pred['all_descriptors'] = all_desc
                pred['dense_score'] = dense_scores
            else:
                del all_desc
                torch.cuda.empty_cache()

        return pred

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError
