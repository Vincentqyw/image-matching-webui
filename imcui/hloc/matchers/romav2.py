import sys
from pathlib import Path

import torch

from .. import logger
from ..utils.base_model import BaseModel

romav2_path = Path(__file__).parent / "../../third_party/RoMaV2/src"
sys.path.append(str(romav2_path))

import romav2.device as romav2_device
from romav2 import RoMaV2


class RoMaV2Matcher(BaseModel):
    default_conf = {
        "max_keypoints": 2048,
        "match_threshold": 0.2,
        "weights_url": "https://github.com/Parskatt/RoMaV2/releases/download/weights/romav2.pt",
    }
    required_inputs = ["image0", "image1"]

    def _init(self, conf):
        # Temporarily override global device for proper initialization
        original_device = romav2_device.device
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        romav2_device.device = target_device
        try:
            cfg = RoMaV2.Cfg(compile=False)
            self.net = RoMaV2(cfg=cfg)
            weights = torch.hub.load_state_dict_from_url(
                conf["weights_url"], map_location="cpu"
            )
            self.net.load_state_dict(weights)
        finally:
            romav2_device.device = original_device

        self.net = self.net.float().eval().to(target_device)
        logger.info("Loaded RoMaV2 model.")

    def _forward(self, data):
        img0 = data["image0"]
        img1 = data["image1"]
        h0, w0 = img0.shape[-2:]
        h1, w1 = img1.shape[-2:]

        preds = self.net.match(img0, img1)
        matches, confidence, _, _ = self.net.sample(preds, self.conf["max_keypoints"])
        mkpts0, mkpts1 = self.net.to_pixel_coordinates(matches, h0, w0, h1, w1)

        return {
            "keypoints0": mkpts0,
            "keypoints1": mkpts1,
            "mconf": confidence,
        }
