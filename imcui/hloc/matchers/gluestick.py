import sys
from pathlib import Path

import torch

from .. import MODEL_REPO_ID, logger
from ..utils.base_model import BaseModel

gluestick_path = Path(__file__).parent / "../../third_party/GlueStick"
sys.path.append(str(gluestick_path))

from gluestick import batch_to_np
from gluestick.models.two_view_pipeline import TwoViewPipeline


class GlueStick(BaseModel):
    default_conf = {
        "name": "two_view_pipeline",
        "model_name": "checkpoint_GlueStick_MD.tar",
        "use_lines": True,
        "max_keypoints": 1000,
        "max_lines": 300,
        "force_num_keypoints": False,
    }
    required_inputs = [
        "image0",
        "image1",
    ]

    # Initialize the line matcher
    def _init(self, conf):
        # Download the model.
        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, self.conf["model_name"]),
        )
        logger.info("Loading GlueStick model...")

        gluestick_conf = {
            "name": "two_view_pipeline",
            "use_lines": True,
            "extractor": {
                "name": "wireframe",
                "sp_params": {
                    "force_num_keypoints": False,
                    "max_num_keypoints": 1000,
                },
                "wireframe_params": {
                    "merge_points": True,
                    "merge_line_endpoints": True,
                },
                "max_n_lines": 300,
            },
            "matcher": {
                "name": "gluestick",
                "weights": str(model_path),
                "trainable": False,
            },
            "ground_truth": {
                "from_pose_depth": False,
            },
        }
        gluestick_conf["extractor"]["sp_params"]["max_num_keypoints"] = conf[
            "max_keypoints"
        ]
        gluestick_conf["extractor"]["sp_params"]["force_num_keypoints"] = conf[
            "force_num_keypoints"
        ]
        gluestick_conf["extractor"]["max_n_lines"] = conf["max_lines"]
        self.net = TwoViewPipeline(gluestick_conf)

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

        pred["raw_lines0"], pred["raw_lines1"] = line_seg0, line_seg1
        pred["lines0"], pred["lines1"] = matched_lines0, matched_lines1
        pred["keypoints0"], pred["keypoints1"] = (
            torch.from_numpy(matched_kps0),
            torch.from_numpy(matched_kps1),
        )
        pred = {**pred, **data}
        return pred
