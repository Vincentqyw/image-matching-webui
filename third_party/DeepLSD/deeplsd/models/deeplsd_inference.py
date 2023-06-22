"""
Regress the distance function map to all the line segments of an image.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .base_model import BaseModel
from .backbones.vgg_unet import VGGUNet
from ..geometry.line_utils import merge_lines, filter_outlier_lines
from ..utils.tensor import preprocess_angle
from pytlsd import lsd


class DeepLSD(BaseModel):
    default_conf = {
        'line_neighborhood': 5,
        'multiscale': False,
        'scale_factors': [1., 1.5],
        'detect_lines': True,
        'line_detection_params': {
            'merge': False,
            'grad_nfa': True,
            'filtering': 'normal',
            'grad_thresh': 3,
        },
    }
    required_data_keys = ['image']

    def _init(self, conf):
        # Base network
        self.backbone = VGGUNet(tiny=False)
        dim = 64

        # Predict the distance field and angle to the nearest line
        # DF head
        self.df_head = nn.Sequential(
            nn.Conv2d(dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(),
        )

        # Closest line direction head
        self.angle_head = nn.Sequential(
            nn.Conv2d(dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Loss
        self.l1_loss_fn = nn.L1Loss(reduction='none')
        self.l2_loss_fn = nn.MSELoss(reduction='none')

    def normalize_df(self, df):
        return -torch.log(df / self.conf.line_neighborhood + 1e-6)

    def denormalize_df(self, df_norm):
        return torch.exp(-df_norm) * self.conf.line_neighborhood

    def _forward(self, data):
        outputs = {}

        if self.conf.multiscale:
            outputs = self.ms_forward(data)
        else:
            base = self.backbone(data['image'])

            # DF prediction
            outputs['df_norm'] = self.df_head(base).squeeze(1)
            outputs['df'] = self.denormalize_df(outputs['df_norm'])

            # Closest line direction prediction
            outputs['line_level'] = self.angle_head(base).squeeze(1) * np.pi

        # Detect line segments
        if self.conf.detect_lines:
            lines = []
            np_img = (data['image'].cpu().numpy()[:, 0] * 255).astype(np.uint8)
            np_df = outputs['df'].cpu().numpy()
            np_ll = outputs['line_level'].cpu().numpy()
            for img, df, ll in zip(np_img, np_df, np_ll):
                line = self.detect_afm_lines(
                    img, df, ll, **self.conf.line_detection_params)
                lines.append(line)
            outputs['lines'] = lines

        return outputs

    def ms_forward(self, data):
        """ Do several forward passes at multiple image resolutions
            and aggregate the results before extracting the lines. """
        img_size = data['image'].shape[2:]

        # Forward pass for each scale
        pred_df, pred_angle = [], []
        for s in self.conf.scale_factors:
            img = F.interpolate(data['image'], scale_factor=s, mode='bilinear')
            with torch.no_grad():
                base = self.backbone(img)
                pred_df.append(self.denormalize_df(self.df_head(base)))
                pred_angle.append(self.angle_head(base) * np.pi)

        # Fuse the outputs together
        for i in range(len(self.conf.scale_factors)):
            pred_df[i] = F.interpolate(pred_df[i], img_size,
                                       mode='bilinear').squeeze(1)
            pred_angle[i] = F.interpolate(pred_angle[i], img_size,
                                          mode='nearest').squeeze(1)
        fused_df = torch.stack(pred_df, dim=0).mean(dim=0)
        fused_angle = torch.median(torch.stack(pred_angle, dim=0), dim=0)[0]

        out = {'df': fused_df, 'line_level': fused_angle}
        return out

    def detect_afm_lines(
        self, img, df, line_level, filtering='normal',
        merge=False, grad_thresh=3, grad_nfa=True):
        """ Detect lines from the line distance and angle field.
            Offer the possibility to ignore line in high DF values,
            and to merge close-by lines. """
        gradnorm = np.maximum(5 - df, 0).astype(np.float64)
        angle = line_level.astype(np.float64) - np.pi / 2
        angle = preprocess_angle(angle, img, mask=True)[0]
        angle[gradnorm < grad_thresh] = -1024
        lines = lsd(
            img.astype(np.float64), scale=1., gradnorm=gradnorm,
            gradangle=angle, grad_nfa=grad_nfa)[:, :4].reshape(-1, 2, 2)

        # Optionally filter out lines based on the DF and line_level
        if filtering:
            if filtering == 'strict':
                df_thresh, ang_thresh = 1., np.pi / 12
            else:
                df_thresh, ang_thresh = 1.5, np.pi / 9
            angle = line_level - np.pi / 2
            lines = filter_outlier_lines(
                img, lines[:, :, [1, 0]], df, angle, mode='inlier_thresh',
                use_grad=False, inlier_thresh=0.5, df_thresh=df_thresh,
                ang_thresh=ang_thresh)[0][:, :, [1, 0]]

        # Merge close-by lines together
        if merge:
            lines = merge_lines(lines, thresh=4,
                                overlap_thresh=0).astype(np.float32)

        return lines

    def loss(self, pred, data):
        raise NotImplementedError()

    def metrics(self, pred, data):
        raise NotImplementedError()
