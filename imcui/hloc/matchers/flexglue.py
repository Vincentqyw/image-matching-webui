import sys
from pathlib import Path

import torch

from .. import MODEL_REPO_ID, logger
from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / "../../third_party/FlexGlue"))

from flexglue import FlexGlue as FlexGlueModel


class FlexGlue(BaseModel):
    """Sparse matcher: keypoints from any detector, matched by FlexGlue."""

    default_conf = {
        "match_threshold": 0.1,
        "max_distance_px": 3.0,
        "min_covisibility": 0.0,
    }
    required_inputs = [
        "image0",
        "image1",
        "keypoints0",
        "keypoints1",
    ]

    def _init(self, _conf):
        logger.info("Loading FlexGlue model (PyTorch)")
        weights = self._download_model(MODEL_REPO_ID, "flexglue/flexglue.pth")
        self.net = FlexGlueModel(weights=str(weights))
        logger.info("Load FlexGlue model done.")

    def _forward(self, data):
        # FlexGlue's mutual nearest neighbours need keypoints on both sides.
        if data["keypoints0"].shape[1] == 0 or data["keypoints1"].shape[1] == 0:
            num = data["keypoints0"].shape[1]
            return {
                "matches0": torch.full((1, num), -1, dtype=torch.long),
                "matching_scores0": torch.zeros((1, num)),
            }
        device = next(self.net.parameters()).device
        # The extractor's own (normalized) preprocessed images: the keypoints
        # are in that frame, and FlexGlue resizes to 512x512 regardless.
        inputs = [
            {
                "image": data[f"image{i}"].to(device),
                "keypoints": data[f"keypoints{i}"].to(device),
            }
            for i in (0, 1)
        ]
        pred = self.net.match(
            *inputs,
            sparse=True,
            filter_threshold=self.conf["match_threshold"],
            max_distance_px=self.conf["max_distance_px"],
            min_covisibility=self.conf["min_covisibility"],
            mp="float16" if device.type == "cuda" else None,
        )
        return {
            "matches0": pred["matches0"],
            "matching_scores0": pred["matching_scores0"],
        }
