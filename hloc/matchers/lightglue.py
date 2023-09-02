import sys
from pathlib import Path
from ..utils.base_model import BaseModel
from .. import logger

lightglue_path = Path(__file__).parent / "../../third_party/LightGlue"
sys.path.append(str(lightglue_path))
from lightglue import LightGlue as LG


class LightGlue(BaseModel):
    default_conf = {
        "match_threshold": 0.2,
        "filter_threshold": 0.2,
        "width_confidence": 0.99,  # for point pruning
        "depth_confidence": 0.95,  # for early stopping,
        "features": "superpoint",
        "model_name": "superpoint_lightglue.pth",
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
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
        weight_path = lightglue_path / "weights" / conf["model_name"]
        conf["weights"] = str(weight_path)
        conf["filter_threshold"] = conf["match_threshold"]
        self.net = LG(**conf)
        logger.info(f"Load lightglue model done.")

    def _forward(self, data):
        input = {}
        input["image0"] = {
            "image": data["image0"],
            "keypoints": data["keypoints0"],
            "descriptors": data["descriptors0"].permute(0, 2, 1),
        }
        input["image1"] = {
            "image": data["image1"],
            "keypoints": data["keypoints1"],
            "descriptors": data["descriptors1"].permute(0, 2, 1),
        }
        return self.net(input)
