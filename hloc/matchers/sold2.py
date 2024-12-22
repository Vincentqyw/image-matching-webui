import sys
from pathlib import Path

import torch

from .. import MODEL_REPO_ID, logger
from ..utils.base_model import BaseModel

sold2_path = Path(__file__).parent / "../../third_party/SOLD2"
sys.path.append(str(sold2_path))

from sold2.model.line_matcher import LineMatcher

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SOLD2(BaseModel):
    default_conf = {
        "model_name": "sold2_wireframe.tar",
        "match_threshold": 0.2,
        "checkpoint_dir": sold2_path / "pretrained",
        "detect_thresh": 0.25,
        "multiscale": False,
        "valid_thresh": 1e-3,
        "num_blocks": 20,
        "overlap_ratio": 0.5,
    }
    required_inputs = [
        "image0",
        "image1",
    ]

    # Initialize the line matcher
    def _init(self, conf):
        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, self.conf["model_name"]),
        )
        logger.info("Loading SOLD2 model: {}".format(model_path))

        mode = "dynamic"  # 'dynamic' or 'static'
        match_config = {
            "model_cfg": {
                "model_name": "lcnn_simple",
                "model_architecture": "simple",
                # Backbone related config
                "backbone": "lcnn",
                "backbone_cfg": {
                    "input_channel": 1,  # Use RGB images or grayscale images.
                    "depth": 4,
                    "num_stacks": 2,
                    "num_blocks": 1,
                    "num_classes": 5,
                },
                # Junction decoder related config
                "junction_decoder": "superpoint_decoder",
                "junc_decoder_cfg": {},
                # Heatmap decoder related config
                "heatmap_decoder": "pixel_shuffle",
                "heatmap_decoder_cfg": {},
                # Descriptor decoder related config
                "descriptor_decoder": "superpoint_descriptor",
                "descriptor_decoder_cfg": {},
                # Shared configurations
                "grid_size": 8,
                "keep_border_valid": True,
                # Threshold of junction detection
                "detection_thresh": 0.0153846,  # 1/65
                "max_num_junctions": 300,
                # Threshold of heatmap detection
                "prob_thresh": 0.5,
                # Weighting related parameters
                "weighting_policy": mode,
                # [Heatmap loss]
                "w_heatmap": 0.0,
                "w_heatmap_class": 1,
                "heatmap_loss_func": "cross_entropy",
                "heatmap_loss_cfg": {"policy": mode},
                # [Heatmap consistency loss]
                # [Junction loss]
                "w_junc": 0.0,
                "junction_loss_func": "superpoint",
                "junction_loss_cfg": {"policy": mode},
                # [Descriptor loss]
                "w_desc": 0.0,
                "descriptor_loss_func": "regular_sampling",
                "descriptor_loss_cfg": {
                    "dist_threshold": 8,
                    "grid_size": 4,
                    "margin": 1,
                    "policy": mode,
                },
            },
            "line_detector_cfg": {
                "detect_thresh": 0.25,  # depending on your images, you might need to tune this parameter
                "num_samples": 64,
                "sampling_method": "local_max",
                "inlier_thresh": 0.9,
                "use_candidate_suppression": True,
                "nms_dist_tolerance": 3.0,
                "use_heatmap_refinement": True,
                "heatmap_refine_cfg": {
                    "mode": "local",
                    "ratio": 0.2,
                    "valid_thresh": 1e-3,
                    "num_blocks": 20,
                    "overlap_ratio": 0.5,
                },
            },
            "multiscale": False,
            "line_matcher_cfg": {
                "cross_check": True,
                "num_samples": 5,
                "min_dist_pts": 8,
                "top_k_candidates": 10,
                "grid_size": 4,
            },
        }
        self.net = LineMatcher(
            match_config["model_cfg"],
            model_path,
            device,
            match_config["line_detector_cfg"],
            match_config["line_matcher_cfg"],
            match_config["multiscale"],
        )

    def _forward(self, data):
        img0 = data["image0"]
        img1 = data["image1"]
        pred = self.net([img0, img1])
        line_seg1 = pred["line_segments"][0]
        line_seg2 = pred["line_segments"][1]
        matches = pred["matches"]

        valid_matches = matches != -1
        match_indices = matches[valid_matches]
        matched_lines1 = line_seg1[valid_matches][:, :, ::-1]
        matched_lines2 = line_seg2[match_indices][:, :, ::-1]

        pred["raw_lines0"], pred["raw_lines1"] = line_seg1, line_seg2
        pred["lines0"], pred["lines1"] = matched_lines1, matched_lines2
        pred = {**pred, **data}
        return pred
