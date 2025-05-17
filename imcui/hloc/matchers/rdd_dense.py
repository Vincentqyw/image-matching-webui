import sys
import yaml
import torch
from pathlib import Path
from ..utils.base_model import BaseModel
from .. import logger, MODEL_REPO_ID, DEVICE

rdd_path = Path(__file__).parent / "../../third_party/rdd"
sys.path.append(str(rdd_path))

from RDD.RDD import build as build_rdd
from RDD.RDD_helper import RDD_helper


class RddDense(BaseModel):
    default_conf = {
        "keypoint_threshold": 0.1,
        "max_keypoints": 4096,
        "model_name": "RDD-v2.pth",
        "match_threshold": 0.1,
    }

    required_inputs = ["image0", "image1"]

    def _init(self, conf):
        logger.info("Loading RDD model...")
        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format("rdd", self.conf["model_name"]),
        )
        config_path = rdd_path / "configs/default.yaml"
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        config["top_k"] = conf["max_keypoints"]
        config["detection_threshold"] = conf["keypoint_threshold"]
        config["device"] = DEVICE
        rdd_net = build_rdd(config=config, weights=model_path)
        rdd_net.eval()
        self.net = RDD_helper(rdd_net)
        logger.info("Loading RDD model done!")

    def _forward(self, data):
        img0 = data["image0"]
        img1 = data["image1"]
        mkpts_0, mkpts_1, conf = self.net.match_dense(
            img0, img1, thr=self.conf["match_threshold"]
        )
        pred = {
            "keypoints0": torch.from_numpy(mkpts_0),
            "keypoints1": torch.from_numpy(mkpts_1),
            "mconf": torch.from_numpy(conf),
        }
        return pred
