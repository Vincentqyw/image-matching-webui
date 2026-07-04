import sys
from pathlib import Path

from ..utils.base_model import BaseModel

raco_path = Path(__file__).parent / "../../third_party/RaCo"
sys.path.append(str(raco_path))

lightglue_path = Path(__file__).parent / "../../third_party/LightGlue"
sys.path.append(str(lightglue_path))

from lightglue import ALIKED as ALIKED_
from raco import RaCo as RaCo_


class RaCo(BaseModel):
    default_conf = {
        "model_name": "raco",
        "max_num_keypoints": 1024,
        "nms_radius": 3,
        "subpixel_sampling": True,
        "subpixel_temp": 0.5,
        "ranker": True,
        "covariance_estimator": True,
        "sort_by_ranker": False,
        "aliked_model_name": "aliked-n16",
        "aliked_detection_threshold": 0.2,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        raco_conf = {
            "max_num_keypoints": conf["max_num_keypoints"],
            "nms_radius": conf["nms_radius"],
            "subpixel_sampling": conf["subpixel_sampling"],
            "subpixel_temp": conf["subpixel_temp"],
            "ranker": conf["ranker"],
            "covariance_estimator": conf["covariance_estimator"],
            "sort_by_ranker": conf["sort_by_ranker"],
        }
        self.raco = RaCo_(**raco_conf)

        aliked_conf = {
            "model_name": conf["aliked_model_name"],
            "max_num_keypoints": -1,
            "detection_threshold": conf["aliked_detection_threshold"],
            "nms_radius": 2,
        }
        self.aliked = ALIKED_(**aliked_conf)

    def _forward(self, data):
        image = data["image"]
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        raco_out = self.raco.extract(image)
        keypoints = raco_out["keypoints"]
        scores = raco_out["keypoint_scores"]
        descriptors = self.aliked.describe(keypoints, image)

        return {
            "keypoints": [f for f in keypoints],
            "scores": [f for f in scores],
            "descriptors": [f.t() for f in descriptors],
        }
