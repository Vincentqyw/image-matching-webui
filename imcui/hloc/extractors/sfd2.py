import sys
from pathlib import Path

import torchvision.transforms as tvf

from .. import MODEL_REPO_ID, logger
from ..utils.base_model import BaseModel

tp_path = Path(__file__).parent / "../../third_party"
sys.path.append(str(tp_path))
from pram.nets.sfd2 import load_sfd2


class SFD2(BaseModel):
    default_conf = {
        "max_keypoints": 4096,
        "model_name": "sfd2_20230511_210205_resnet4x.79.pth",
        "conf_th": 0.001,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        self.conf = {**self.default_conf, **conf}
        self.norm_rgb = tvf.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format("pram", self.conf["model_name"]),
        )
        self.net = load_sfd2(weight_path=model_path).eval()

        logger.info("Load SFD2 model done.")

    def _forward(self, data):
        pred = self.net.extract_local_global(
            data={"image": self.norm_rgb(data["image"])}, config=self.conf
        )
        out = {
            "keypoints": pred["keypoints"][0][None],
            "scores": pred["scores"][0][None],
            "descriptors": pred["descriptors"][0][None],
        }
        return out
