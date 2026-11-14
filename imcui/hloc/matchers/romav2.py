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
    }
    required_inputs = ["image0", "image1"]

    def _init(self, conf):
        # RoMaV2's constructor unconditionally requests this upstream file.
        self._download_torch_checkpoint("romav2/romav2.0.1.pt", "romav2.0.1.pt")

        # Temporarily override global device for proper initialization
        original_device = romav2_device.device
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        romav2_device.device = target_device
        try:
            cfg = RoMaV2.Cfg(compile=False)
            self.net = RoMaV2(cfg=cfg)
            model_path = self._download_model(filename="romav2/romav2.pt")
            self.net.load_state_dict(torch.load(model_path, map_location="cpu"))
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
