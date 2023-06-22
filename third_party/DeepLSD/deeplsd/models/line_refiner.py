"""
    Regress the distance function map of an image
    and use it to refine pre-computed line segments..
"""

import numpy as np
import torch
from torch import nn
from copy import deepcopy
from omegaconf import OmegaConf

from .base_model import BaseModel
from .backbones.vgg_unet import VGGUNet
from ..geometry.line_utils import get_line_orientation, filter_outlier_lines
from ..utils.tensor import preprocess_angle
from line_refinement import line_optim


class LineRefiner(BaseModel):
    default_conf = {
        'tiny': False,
        'sharpen': True,
        'line_neighborhood': 5,
        'line_detection_params': {
            'use_vps': True,
            'optimize_vps': True,
            'filtering': False,
            'lambda_df': 1.,
            'lambda_grad': 1.,
            'lambda_vp': 0.5,
            'threshold': 1.,
            'max_iters': 100000,
            'minimum_point_number': 2,
            'maximum_model_number': -1,
            'scoring_exponent': 1,
        },
    }
    required_data_keys = ['image', 'lines']

    def _init(self, conf):
        # Base network
        self.backbone = VGGUNet(tiny=self.conf.tiny)
        dim = 32 if self.conf.tiny else 64

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
    
    def normalize_df(self, df):
        return -torch.log(df / self.conf.line_neighborhood + 1e-6)
    
    def denormalize_df(self, df_norm):
        return torch.exp(-df_norm) * self.conf.line_neighborhood

    def _forward(self, data):
        outputs = {}

        base = self.backbone(data['image'])

        # DF prediction
        if self.conf.sharpen:
            outputs['df_norm'] = self.df_head(base).squeeze(1)
            outputs['df'] = self.denormalize_df(outputs['df_norm'])
        else:
            outputs['df'] = self.df_head(base).squeeze(1)

        # Closest line direction prediction
        outputs['line_level'] = self.angle_head(base).squeeze(1) * np.pi
        
        # Refine the line segments
        np_img = (data['image'].cpu().numpy()[:, 0] * 255).astype(np.uint8)
        np_df = outputs['df'].cpu().numpy()
        np_ll = outputs['line_level'].cpu().numpy()
        outputs['refined_lines'] = []
        outputs['vp_labels'] = []
        outputs['vps'] = []
        line_detection_params = deepcopy(self.conf.line_detection_params)
        OmegaConf.set_readonly(line_detection_params, False)
        if 'line_detection_params' in data:
            line_detection_params.update(data['line_detection_params'])
        for img, df, ll, lines in zip(np_img, np_df, np_ll, data['lines']):
            out = self.refine_lines(
                img, df, ll, lines, **line_detection_params)
            for k, v in out.items():
                outputs[k].append(v)

        return outputs
    
    def refine_lines(self, img, df, line_level, lines, use_vps=False,
                     optimize_vps=False, filtering='normal', lambda_df=1.,
                     lambda_grad=1., lambda_vp=0.5, threshold=1.,
                     max_iters=10000, minimum_point_number=2,
                     maximum_model_number=-1, scoring_exponent=1):
        """ Refine the given lines using a DF+angle field.
            Lines are expected in xy convention. """
        rows, cols = df.shape
        angle, img_grad_angle = preprocess_angle(line_level - np.pi / 2, img)
        orientations = get_line_orientation(lines[:, :, [1, 0]],
                                            angle)[:, None]
        oriented_lines = np.concatenate([lines.reshape(-1, 4),
                                        orientations], axis=1)
        out = {}
        refined_lines, out['vp_labels'], out['vps'] = line_optim(
            oriented_lines, df.flatten(),
            angle.flatten(), rows, cols, use_vps, optimize_vps,
            lambda_df, lambda_grad, lambda_vp, threshold, max_iters,
            minimum_point_number, maximum_model_number,
            scoring_exponent)
        refined_lines = np.array(refined_lines).reshape(-1, 2, 2)
        out['refined_lines'] = refined_lines.astype(np.float32)
        
        # Optionally filter out lines based on the DF and line_level
        if filtering:
            if filtering == 'strict':
                df_thresh, ang_thresh = 1., np.pi / 12
            else:
                df_thresh, ang_thresh = 1.5, np.pi / 9
            angle = line_level - np.pi / 2
            lines, valid = filter_outlier_lines(
                img, out['refined_lines'][:, :, [1, 0]], df, angle,
                mode='inlier_thresh', use_grad=False, inlier_thresh=0.5,
                df_thresh=df_thresh, ang_thresh=ang_thresh)
            out['refined_lines'] = lines[:, :, [1, 0]]
            out['vp_labels'] = np.array(out['vp_labels'])[valid]
        
        return out

    def loss(self, pred, data):
        return {}

    def metrics(self, pred, data):
        return {}
