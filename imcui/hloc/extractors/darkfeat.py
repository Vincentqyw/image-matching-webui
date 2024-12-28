import sys
from pathlib import Path

from hloc import MODEL_REPO_ID, logger

from ..utils.base_model import BaseModel

darkfeat_path = Path(__file__).parent / "../../third_party/DarkFeat"
sys.path.append(str(darkfeat_path))
from darkfeat import DarkFeat as DarkFeat_


class DarkFeat(BaseModel):
    default_conf = {
        "model_name": "DarkFeat.pth",
        "max_keypoints": 1000,
        "detection_threshold": 0.5,
        "sub_pixel": False,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, self.conf["model_name"]),
        )
        logger.info("Loaded DarkFeat model: {}".format(model_path))
        self.net = DarkFeat_(model_path)
        logger.info("Load DarkFeat model done.")

    def _forward(self, data):
        pred = self.net({"image": data["image"]})
        keypoints = pred["keypoints"]
        descriptors = pred["descriptors"]
        scores = pred["scores"]
        idxs = scores.argsort()[-self.conf["max_keypoints"] or None :]
        keypoints = keypoints[idxs, :2]
        descriptors = descriptors[:, idxs]
        scores = scores[idxs]
        return {
            "keypoints": keypoints[None],  # 1 x N x 2
            "scores": scores[None],  # 1 x N
            "descriptors": descriptors[None],  # 1 x 128 x N
        }
