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


class XFeatDense(BaseModel):
    default_conf = {
        "keypoint_threshold": 0.005,
        "max_keypoints": 8000,
    }
    required_inputs = [
        "image0",
        "image1",
    ]

    def _init(self, conf):
        _ensure_hub_trusted("verlab/accelerated_features")
        self.net = torch.hub.load(
            "verlab/accelerated_features",
            "XFeat",
            pretrained=True,
            top_k=self.conf["max_keypoints"],
        )
        logger.info("Load XFeat(dense) model done.")

    def _forward(self, data):
        # Compute coarse feats
        out0 = self.net.detectAndComputeDense(
            data["image0"], top_k=self.conf["max_keypoints"]
        )
        out1 = self.net.detectAndComputeDense(
            data["image1"], top_k=self.conf["max_keypoints"]
        )

        # Match batches of pairs
        idxs_list = self.net.batch_match(out0["descriptors"], out1["descriptors"])
        B = len(data["image0"])

        # Refine coarse matches
        # this part is harder to batch, currently iterate
        matches = []
        for b in range(B):
            matches.append(
                self.net.refine_matches(out0, out1, matches=idxs_list, batch_idx=b)
            )
        # we use results from one batch
        matches = matches[0]
        pred = {
            "keypoints0": out0["keypoints"].squeeze(),
            "keypoints1": out1["keypoints"].squeeze(),
            "mkeypoints0": matches[:, :2],
            "mkeypoints1": matches[:, 2:],
            "mconf": torch.ones_like(matches[:, 0]),
        }
        return pred
