import os
from pathlib import Path

import torch

from .. import logger

from ..utils.base_model import BaseModel


def _ensure_hub_trusted(repo: str) -> None:
    """Pre-trust a torch hub repo so CI/non-interactive sessions don't hang."""
    hub_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub")
    trusted_file = os.path.join(hub_dir, "trusted_list")
    os.makedirs(hub_dir, exist_ok=True)
    Path(trusted_file).touch(exist_ok=True)
    owner_name = repo.replace("/", "_")
    with open(trusted_file) as f:
        lines = f.read()
    if owner_name not in lines:
        with open(trusted_file, "a") as f:
            f.write(f"{owner_name}\n")


class XFeat(BaseModel):
    default_conf = {
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        _ensure_hub_trusted("verlab/accelerated_features")
        self.net = torch.hub.load(
            "verlab/accelerated_features",
            "XFeat",
            pretrained=True,
            top_k=self.conf["max_keypoints"],
        )
        logger.info("Load XFeat(sparse) model done.")

    def _forward(self, data):
        pred = self.net.detectAndCompute(
            data["image"], top_k=self.conf["max_keypoints"]
        )[0]
        pred = {
            "keypoints": pred["keypoints"][None],
            "scores": pred["scores"][None],
            "descriptors": pred["descriptors"].T[None],
        }
        return pred
