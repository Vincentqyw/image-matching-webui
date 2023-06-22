import numpy as np
import torch
from pytlsd import lsd
from sklearn.cluster import DBSCAN

from .base_model import BaseModel
from .superpoint import SuperPoint, sample_descriptors
from ..geometry import warp_lines_torch


def lines_to_wireframe(lines, line_scores, all_descs, conf):
    """ Given a set of lines, their score and dense descriptors,
        merge close-by endpoints and compute a wireframe defined by
        its junctions and connectivity.
    Returns:
        junctions: list of [num_junc, 2] tensors listing all wireframe junctions
        junc_scores: list of [num_junc] tensors with the junction score
        junc_descs: list of [dim, num_junc] tensors with the junction descriptors
        connectivity: list of [num_junc, num_junc] bool arrays with True when 2 junctions are connected
        new_lines: the new set of [b_size, num_lines, 2, 2] lines
        lines_junc_idx: a [b_size, num_lines, 2] tensor with the indices of the junctions of each endpoint
        num_true_junctions: a list of the number of valid junctions for each image in the batch,
                            i.e. before filling with random ones
    """
    b_size, _, _, _ = all_descs.shape
    device = lines.device
    endpoints = lines.reshape(b_size, -1, 2)

    (junctions, junc_scores, junc_descs, connectivity, new_lines,
     lines_junc_idx, num_true_junctions) = [], [], [], [], [], [], []
    for bs in range(b_size):
        # Cluster the junctions that are close-by
        db = DBSCAN(eps=conf.nms_radius, min_samples=1).fit(
            endpoints[bs].cpu().numpy())
        clusters = db.labels_
        n_clusters = len(set(clusters))
        num_true_junctions.append(n_clusters)

        # Compute the average junction and score for each cluster
        clusters = torch.tensor(clusters, dtype=torch.long,
                                device=device)
        new_junc = torch.zeros(n_clusters, 2, dtype=torch.float,
                               device=device)
        new_junc.scatter_reduce_(0, clusters[:, None].repeat(1, 2),
                                 endpoints[bs], reduce='mean',
                                 include_self=False)
        junctions.append(new_junc)
        new_scores = torch.zeros(n_clusters, dtype=torch.float, device=device)
        new_scores.scatter_reduce_(
            0, clusters, torch.repeat_interleave(line_scores[bs], 2),
            reduce='mean', include_self=False)
        junc_scores.append(new_scores)

        # Compute the new lines
        new_lines.append(junctions[-1][clusters].reshape(-1, 2, 2))
        lines_junc_idx.append(clusters.reshape(-1, 2))

        # Compute the junction connectivity
        junc_connect = torch.eye(n_clusters, dtype=torch.bool,
                                 device=device)
        pairs = clusters.reshape(-1, 2)  # these pairs are connected by a line
        junc_connect[pairs[:, 0], pairs[:, 1]] = True
        junc_connect[pairs[:, 1], pairs[:, 0]] = True
        connectivity.append(junc_connect)

        # Interpolate the new junction descriptors
        junc_descs.append(sample_descriptors(
            junctions[-1][None], all_descs[bs:(bs + 1)], 8)[0])

    new_lines = torch.stack(new_lines, dim=0)
    lines_junc_idx = torch.stack(lines_junc_idx, dim=0)
    return (junctions, junc_scores, junc_descs, connectivity,
            new_lines, lines_junc_idx, num_true_junctions)


class SPWireframeDescriptor(BaseModel):
    default_conf = {
        'sp_params': {
            'has_detector': True,
            'has_descriptor': True,
            'descriptor_dim': 256,
            'trainable': False,

            # Inference
            'return_all': True,
            'sparse_outputs': True,
            'nms_radius': 4,
            'detection_threshold': 0.005,
            'max_num_keypoints': 1000,
            'force_num_keypoints': True,
            'remove_borders': 4,
        },
        'wireframe_params': {
            'merge_points': True,
            'merge_line_endpoints': True,
            'nms_radius': 3,
            'max_n_junctions': 500,
        },
        'max_n_lines': 250,
        'min_length': 15,
    }
    required_data_keys = ['image']

    def _init(self, conf):
        self.conf = conf
        self.sp = SuperPoint(conf.sp_params)

    def detect_lsd_lines(self, x, max_n_lines=None):
        if max_n_lines is None:
            max_n_lines = self.conf.max_n_lines
        lines, scores, valid_lines = [], [], []
        for b in range(len(x)):
            # For each image on batch
            img = (x[b].squeeze().cpu().numpy() * 255).astype(np.uint8)
            if max_n_lines is None:
                b_segs = lsd(img)
            else:
                for s in [0.3, 0.4, 0.5, 0.7, 0.8, 1.0]:
                    b_segs = lsd(img, scale=s)
                    if len(b_segs) >= max_n_lines:
                        break

            segs_length = np.linalg.norm(b_segs[:, 2:4] - b_segs[:, 0:2], axis=1)
            # Remove short lines
            b_segs = b_segs[segs_length >= self.conf.min_length]
            segs_length = segs_length[segs_length >= self.conf.min_length]
            b_scores = b_segs[:, -1] * np.sqrt(segs_length)
            # Take the most relevant segments with
            indices = np.argsort(-b_scores)
            if max_n_lines is not None:
                indices = indices[:max_n_lines]
            lines.append(torch.from_numpy(b_segs[indices, :4].reshape(-1, 2, 2)))
            scores.append(torch.from_numpy(b_scores[indices]))
            valid_lines.append(torch.ones_like(scores[-1], dtype=torch.bool))

        lines = torch.stack(lines).to(x)
        scores = torch.stack(scores).to(x)
        valid_lines = torch.stack(valid_lines).to(x.device)
        return lines, scores, valid_lines

    def _forward(self, data):
        b_size, _, h, w = data['image'].shape
        device = data['image'].device

        if not self.conf.sp_params.force_num_keypoints:
            assert b_size == 1, "Only batch size of 1 accepted for non padded inputs"

        # Line detection
        if 'lines' not in data or 'line_scores' not in data:
            if 'original_img' in data:
                # Detect more lines, because when projecting them to the image most of them will be discarded
                lines, line_scores, valid_lines = self.detect_lsd_lines(
                    data['original_img'], self.conf.max_n_lines * 3)
                # Apply the same transformation that is applied in homography_adaptation
                lines, valid_lines2 = warp_lines_torch(lines, data['H'], False, data['image'].shape[-2:])
                valid_lines = valid_lines & valid_lines2
                lines[~valid_lines] = -1
                line_scores[~valid_lines] = 0
                # Re-sort the line segments to pick the ones that are inside the image and have bigger score
                sorted_scores, sorting_indices = torch.sort(line_scores, dim=-1, descending=True)
                line_scores = sorted_scores[:, :self.conf.max_n_lines]
                sorting_indices = sorting_indices[:, :self.conf.max_n_lines]
                lines = torch.take_along_dim(lines, sorting_indices[..., None, None], 1)
                valid_lines = torch.take_along_dim(valid_lines, sorting_indices, 1)
            else:
                lines, line_scores, valid_lines = self.detect_lsd_lines(data['image'])

        else:
            lines, line_scores, valid_lines = data['lines'], data['line_scores'], data['valid_lines']
        if line_scores.shape[-1] != 0:
            line_scores /= (line_scores.new_tensor(1e-8) + line_scores.max(dim=1).values[:, None])

        # SuperPoint prediction
        pred = self.sp(data)

        # Remove keypoints that are too close to line endpoints
        if self.conf.wireframe_params.merge_points:
            kp = pred['keypoints']
            line_endpts = lines.reshape(b_size, -1, 2)
            dist_pt_lines = torch.norm(
                kp[:, :, None] - line_endpts[:, None], dim=-1)
            # For each keypoint, mark it as valid or to remove
            pts_to_remove = torch.any(
                dist_pt_lines < self.conf.sp_params.nms_radius, dim=2)
            # Simply remove them (we assume batch_size = 1 here)
            assert len(kp) == 1
            pred['keypoints'] = pred['keypoints'][0][~pts_to_remove[0]][None]
            pred['keypoint_scores'] = pred['keypoint_scores'][0][~pts_to_remove[0]][None]
            pred['descriptors'] = pred['descriptors'][0].T[~pts_to_remove[0]].T[None]

        # Connect the lines together to form a wireframe
        orig_lines = lines.clone()
        if self.conf.wireframe_params.merge_line_endpoints and len(lines[0]) > 0:
            # Merge first close-by endpoints to connect lines
            (line_points, line_pts_scores, line_descs, line_association,
             lines, lines_junc_idx, num_true_junctions) = lines_to_wireframe(
                lines, line_scores, pred['all_descriptors'],
                conf=self.conf.wireframe_params)

            # Add the keypoints to the junctions and fill the rest with random keypoints
            (all_points, all_scores, all_descs,
             pl_associativity) = [], [], [], []
            for bs in range(b_size):
                all_points.append(torch.cat(
                    [line_points[bs], pred['keypoints'][bs]], dim=0))
                all_scores.append(torch.cat(
                    [line_pts_scores[bs], pred['keypoint_scores'][bs]], dim=0))
                all_descs.append(torch.cat(
                    [line_descs[bs], pred['descriptors'][bs]], dim=1))

                associativity = torch.eye(len(all_points[-1]), dtype=torch.bool, device=device)
                associativity[:num_true_junctions[bs], :num_true_junctions[bs]] = \
                    line_association[bs][:num_true_junctions[bs], :num_true_junctions[bs]]
                pl_associativity.append(associativity)

            all_points = torch.stack(all_points, dim=0)
            all_scores = torch.stack(all_scores, dim=0)
            all_descs = torch.stack(all_descs, dim=0)
            pl_associativity = torch.stack(pl_associativity, dim=0)
        else:
            # Lines are independent
            all_points = torch.cat([lines.reshape(b_size, -1, 2),
                                    pred['keypoints']], dim=1)
            n_pts = all_points.shape[1]
            num_lines = lines.shape[1]
            num_true_junctions = [num_lines * 2] * b_size
            all_scores = torch.cat([
                torch.repeat_interleave(line_scores, 2, dim=1),
                pred['keypoint_scores']], dim=1)
            pred['line_descriptors'] = self.endpoints_pooling(
                lines, pred['all_descriptors'], (h, w))
            all_descs = torch.cat([
                pred['line_descriptors'].reshape(b_size, self.conf.sp_params.descriptor_dim, -1),
                pred['descriptors']], dim=2)
            pl_associativity = torch.eye(
                n_pts, dtype=torch.bool,
                device=device)[None].repeat(b_size, 1, 1)
            lines_junc_idx = torch.arange(
                num_lines * 2, device=device).reshape(1, -1, 2).repeat(b_size, 1, 1)

        del pred['all_descriptors']  # Remove dense descriptors to save memory
        torch.cuda.empty_cache()

        return {'keypoints': all_points,
                'keypoint_scores': all_scores,
                'descriptors': all_descs,
                'pl_associativity': pl_associativity,
                'num_junctions': torch.tensor(num_true_junctions),
                'lines': lines,
                'orig_lines': orig_lines,
                'lines_junc_idx': lines_junc_idx,
                'line_scores': line_scores,
                'valid_lines': valid_lines}

    @staticmethod
    def endpoints_pooling(segs, all_descriptors, img_shape):
        assert segs.ndim == 4 and segs.shape[-2:] == (2, 2)
        filter_shape = all_descriptors.shape[-2:]
        scale_x = filter_shape[1] / img_shape[1]
        scale_y = filter_shape[0] / img_shape[0]

        scaled_segs = torch.round(segs * torch.tensor([scale_x, scale_y]).to(segs)).long()
        scaled_segs[..., 0] = torch.clip(scaled_segs[..., 0], 0, filter_shape[1] - 1)
        scaled_segs[..., 1] = torch.clip(scaled_segs[..., 1], 0, filter_shape[0] - 1)
        line_descriptors = [all_descriptors[None, b, ..., torch.squeeze(b_segs[..., 1]), torch.squeeze(b_segs[..., 0])]
                            for b, b_segs in enumerate(scaled_segs)]
        line_descriptors = torch.cat(line_descriptors)
        return line_descriptors  # Shape (1, 256, 308, 2)

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        return {}
