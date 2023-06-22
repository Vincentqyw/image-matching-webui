"""
A two-view sparse feature matching pipeline.

This model contains sub-models for each step:
    feature extraction, feature matching, outlier filtering, pose estimation.
Each step is optional, and the features or matches can be provided as input.
Default: SuperPoint with nearest neighbor matching.

Convention for the matches: m0[i] is the index of the keypoint in image 1
that corresponds to the keypoint i in image 0. m0[i] = -1 if i is unmatched.
"""

import numpy as np
import torch

from .. import get_model
from .base_model import BaseModel


def keep_quadrant_kp_subset(keypoints, scores, descs, h, w):
    """Keep only keypoints in one of the four quadrant of the image."""
    h2, w2 = h // 2, w // 2
    w_x = np.random.choice([0, w2])
    w_y = np.random.choice([0, h2])
    valid_mask = ((keypoints[..., 0] >= w_x)
                  & (keypoints[..., 0] < w_x + w2)
                  & (keypoints[..., 1] >= w_y)
                  & (keypoints[..., 1] < w_y + h2))
    keypoints = keypoints[valid_mask][None]
    scores = scores[valid_mask][None]
    descs = descs.permute(0, 2, 1)[valid_mask].t()[None]
    return keypoints, scores, descs


def keep_random_kp_subset(keypoints, scores, descs, num_selected):
    """Keep a random subset of keypoints."""
    num_kp = keypoints.shape[1]
    selected_kp = torch.randperm(num_kp)[:num_selected]
    keypoints = keypoints[:, selected_kp]
    scores = scores[:, selected_kp]
    descs = descs[:, :, selected_kp]
    return keypoints, scores, descs


def keep_best_kp_subset(keypoints, scores, descs, num_selected):
    """Keep the top num_selected best keypoints."""
    sorted_indices = torch.sort(scores, dim=1)[1]
    selected_kp = sorted_indices[:, -num_selected:]
    keypoints = torch.gather(keypoints, 1,
                             selected_kp[:, :, None].repeat(1, 1, 2))
    scores = torch.gather(scores, 1, selected_kp)
    descs = torch.gather(descs, 2,
                         selected_kp[:, None].repeat(1, descs.shape[1], 1))
    return keypoints, scores, descs


class TwoViewPipeline(BaseModel):
    default_conf = {
        'extractor': {
            'name': 'superpoint',
            'trainable': False,
        },
        'use_lines': False,
        'use_points': True,
        'randomize_num_kp': False,
        'detector': {'name': None},
        'descriptor': {'name': None},
        'matcher': {'name': 'nearest_neighbor_matcher'},
        'filter': {'name': None},
        'solver': {'name': None},
        'ground_truth': {
            'from_pose_depth': False,
            'from_homography': False,
            'th_positive': 3,
            'th_negative': 5,
            'reward_positive': 1,
            'reward_negative': -0.25,
            'is_likelihood_soft': True,
            'p_random_occluders': 0,
            'n_line_sampled_pts': 50,
            'line_perp_dist_th': 5,
            'overlap_th': 0.2,
            'min_visibility_th': 0.5
        },
    }
    required_data_keys = ['image0', 'image1']
    strict_conf = False  # need to pass new confs to children models
    components = [
        'extractor', 'detector', 'descriptor', 'matcher', 'filter', 'solver']

    def _init(self, conf):
        if conf.extractor.name:
            self.extractor = get_model(conf.extractor.name)(conf.extractor)
        else:
            if self.conf.detector.name:
                self.detector = get_model(conf.detector.name)(conf.detector)
            else:
                self.required_data_keys += ['keypoints0', 'keypoints1']
            if self.conf.descriptor.name:
                self.descriptor = get_model(conf.descriptor.name)(
                    conf.descriptor)
            else:
                self.required_data_keys += ['descriptors0', 'descriptors1']

        if conf.matcher.name:
            self.matcher = get_model(conf.matcher.name)(conf.matcher)
        else:
            self.required_data_keys += ['matches0']

        if conf.filter.name:
            self.filter = get_model(conf.filter.name)(conf.filter)

        if conf.solver.name:
            self.solver = get_model(conf.solver.name)(conf.solver)

    def _forward(self, data):

        def process_siamese(data, i):
            data_i = {k[:-1]: v for k, v in data.items() if k[-1] == i}
            if self.conf.extractor.name:
                pred_i = self.extractor(data_i)
            else:
                pred_i = {}
                if self.conf.detector.name:
                    pred_i = self.detector(data_i)
                else:
                    for k in ['keypoints', 'keypoint_scores', 'descriptors',
                              'lines', 'line_scores', 'line_descriptors',
                              'valid_lines']:
                        if k in data_i:
                            pred_i[k] = data_i[k]
                if self.conf.descriptor.name:
                    pred_i = {
                        **pred_i, **self.descriptor({**data_i, **pred_i})}
            return pred_i

        pred0 = process_siamese(data, '0')
        pred1 = process_siamese(data, '1')

        pred = {**{k + '0': v for k, v in pred0.items()},
                **{k + '1': v for k, v in pred1.items()}}

        if self.conf.matcher.name:
            pred = {**pred, **self.matcher({**data, **pred})}

        if self.conf.filter.name:
            pred = {**pred, **self.filter({**data, **pred})}

        if self.conf.solver.name:
            pred = {**pred, **self.solver({**data, **pred})}

        return pred

    def loss(self, pred, data):
        losses = {}
        total = 0
        for k in self.components:
            if self.conf[k].name:
                try:
                    losses_ = getattr(self, k).loss(pred, {**pred, **data})
                except NotImplementedError:
                    continue
                losses = {**losses, **losses_}
                total = losses_['total'] + total
        return {**losses, 'total': total}

    def metrics(self, pred, data):
        metrics = {}
        for k in self.components:
            if self.conf[k].name:
                try:
                    metrics_ = getattr(self, k).metrics(pred, {**pred, **data})
                except NotImplementedError:
                    continue
                metrics = {**metrics, **metrics_}
        return metrics
