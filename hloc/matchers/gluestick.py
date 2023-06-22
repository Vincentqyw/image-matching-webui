import sys
from pathlib import Path
from os.path import join

import cv2
import torch
from matplotlib import pyplot as plt
from ..utils.base_model import BaseModel

gluestick_path = Path(__file__).parent / '../../third_party/GlueStick'
sys.path.append(str(gluestick_path))

from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from gluestick.drawing import plot_images, plot_lines, plot_color_line_matches, plot_keypoints, plot_matches
from gluestick.models.two_view_pipeline import TwoViewPipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_N_POINTS, MAX_N_LINES = 1000, 300

class GlueStick(BaseModel):
    default_conf = {
        'name': 'two_view_pipeline',
        'use_lines': True,
        'extractor': {
            'name': 'wireframe',
            'sp_params': {
                'force_num_keypoints': False,
                'max_num_keypoints': MAX_N_POINTS,
            },
            'wireframe_params': {
                'merge_points': True,
                'merge_line_endpoints': True,
            },
            'max_n_lines': MAX_N_LINES,
        },
        'matcher': {
            'name': 'gluestick',
            'weights': str(gluestick_path / 'resources' / 'weights' / 'checkpoint_GlueStick_MD.tar'),
            'trainable': False,
        },
        'ground_truth': {
            'from_pose_depth': False,
        }
    }
    required_inputs = [
        'image0',
        'image1',
    ]
    # Initialize the line matcher
    def _init(self, conf):
        self.net = TwoViewPipeline(conf)

    def _forward(self, data):
        pred = self.net(data)

        pred = batch_to_np(pred)
        kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
        m0 = pred["matches0"]

        line_seg0, line_seg1 = pred["lines0"], pred["lines1"]
        line_matches = pred["line_matches0"]

        valid_matches = m0 != -1
        match_indices = m0[valid_matches]
        matched_kps0 = kp0[valid_matches]
        matched_kps1 = kp1[match_indices]

        valid_matches = line_matches != -1
        match_indices = line_matches[valid_matches]
        matched_lines0 = line_seg0[valid_matches]
        matched_lines1 = line_seg1[match_indices]

        pred['raw_lines0'], pred['raw_lines1'] = line_seg0, line_seg1
        pred['lines0'], pred['lines1'] = matched_lines0, matched_lines1
        pred["keypoints0"], pred["keypoints1"] = torch.from_numpy(matched_kps0), torch.from_numpy(matched_kps1)
        pred = {**pred, **data}
        return pred
