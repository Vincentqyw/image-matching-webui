import sys
from pathlib import Path

import torch

from .. import DEVICE, MODEL_REPO_ID, logger
from ..utils.base_model import BaseModel

tp_path = Path(__file__).parent / "../../third_party"
sys.path.append(str(tp_path))
from pram.nets.gml import GML


class IMP(BaseModel):
    default_conf = {
        "match_threshold": 0.2,
        "features": "sfd2",
        "model_name": "imp_gml.920.pth",
        "sinkhorn_iterations": 20,
    }
    required_inputs = [
        "image0",
        "keypoints0",
        "scores0",
        "descriptors0",
        "image1",
        "keypoints1",
        "scores1",
        "descriptors1",
    ]

    def _init(self, conf):
        self.conf = {**self.default_conf, **conf}
        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format("pram", self.conf["model_name"]),
        )

        # self.net = nets.gml(self.conf).eval().to(DEVICE)
        self.net = GML(self.conf).eval().to(DEVICE)
        self.net.load_state_dict(
            torch.load(model_path, map_location="cpu")["model"], strict=True
        )
        logger.info("Load IMP model done.")

    def _forward(self, data):
        data["descriptors0"] = data["descriptors0"].transpose(2, 1).float()
        data["descriptors1"] = data["descriptors1"].transpose(2, 1).float()

        return self.net.produce_matches(data, p=0.2)
