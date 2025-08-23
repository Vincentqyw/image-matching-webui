import sys
from pathlib import Path
from ..utils.base_model import BaseModel
from .. import logger, MODEL_REPO_ID

liftfeat_path = Path(__file__).parent / "../../third_party/LiftFeat"
sys.path.insert(0, str(liftfeat_path))

from models.liftfeat_wrapper import LiftFeat


class Liftfeat(BaseModel):
    default_conf = {
        "keypoint_threshold": 0.05,
        "max_keypoints": 5000,
        "model_name": "LiftFeat.pth",
    }

    required_inputs = ["image"]

    def _init(self, conf):
        logger.info("Loading LiftFeat model...")
        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, self.conf["model_name"]),
        )
        self.net = LiftFeat(
            weight=model_path,
            detect_threshold=self.conf["keypoint_threshold"],
            top_k=self.conf["max_keypoints"],
        )
        logger.info("Loading LiftFeat model done!")

    def _forward(self, data):
        image = data["image"].cpu().numpy().squeeze() * 255
        image = image.transpose(1, 2, 0)
        pred = self.net.extract(image)

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
