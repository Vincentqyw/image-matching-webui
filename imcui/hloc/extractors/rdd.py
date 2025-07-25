import sys
import yaml
from pathlib import Path
from ..utils.base_model import BaseModel
from .. import logger, MODEL_REPO_ID, DEVICE

rdd_path = Path(__file__).parent / "../../third_party/rdd"
sys.path.append(str(rdd_path))

from RDD.RDD import build as build_rdd


class Rdd(BaseModel):
    default_conf = {
        "keypoint_threshold": 0.1,
        "max_keypoints": 4096,
        "model_name": "RDD-v2.pth",
    }

    required_inputs = ["image"]

    def _init(self, conf):
        logger.info("Loading RDD model...")
        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, self.conf["model_name"]),
        )
        config_path = rdd_path / "configs/default.yaml"
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        config["top_k"] = conf["max_keypoints"]
        config["detection_threshold"] = conf["keypoint_threshold"]
        config["device"] = DEVICE
        self.net = build_rdd(config=config, weights=model_path)
        self.net.eval()
        logger.info("Loading RDD model done!")

    def _forward(self, data):
        image = data["image"]
        self.net.set_softdetect(
            top_k=self.conf["max_keypoints"], scores_th=self.conf["keypoint_threshold"]
        )
        pred = self.net.extract(image)[0]
        keypoints = pred["keypoints"]
        descriptors = pred["descriptors"]
        scores = pred["scores"]
        if self.conf["max_keypoints"] < len(keypoints):
            idxs = scores.argsort()[-self.conf["max_keypoints"] or None :]
            keypoints = keypoints[idxs, :2]
            descriptors = descriptors[idxs]
            scores = scores[idxs]

        pred = {
            "keypoints": keypoints[None],
            "descriptors": descriptors[None].permute(0, 2, 1),
            "scores": scores[None],
        }
        return pred
