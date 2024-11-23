import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms

from hloc import MODEL_REPO_ID, logger

from ..utils.base_model import BaseModel
from kornia import feature as KF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeDoDe(BaseModel):
    default_conf = {
        "name": "dedode",
        "detector_model": "L-upright",
        "descriptor_model": "G-upright",
        "max_keypoints": 2000,
        "match_threshold": 0.2,
        "dense": False,  # Now fixed to be false
    }
    required_inputs = [
        "image",
    ]

    # Initialize the line matcher
    def _init(self, conf):
        self.model = KF.DeDoDe.from_pretrained(detector_weights=conf["detector_model"], descriptor_weights=conf["descriptor_model"])
        logger.info("Load DeDoDe model done.")

    def _forward(self, data):
        """
        data: dict, keys: {'image0','image1'}
        image shape: N x C x H x W
        color mode: RGB
        """
        img0 = data["image"].float()

        coords, scores, descriptions = self.model(img0, n=self.conf["max_keypoints"])

        return {
            "keypoints": coords,  # 1 x N x 2
            "descriptors": descriptions.permute(0, 2, 1),  # 1 x 512 x N
            "scores": scores,  # 1 x N
        }
