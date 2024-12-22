import sys
from pathlib import Path

import torch

from hloc import MODEL_REPO_ID, logger

from ..utils.base_model import BaseModel

lib_path = Path(__file__).parent / "../../third_party"
sys.path.append(str(lib_path))
from lanet.network_v0.model import PointModel

lanet_path = Path(__file__).parent / "../../third_party/lanet"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LANet(BaseModel):
    default_conf = {
        "model_name": "PointModel_v0.pth",
        "keypoint_threshold": 0.1,
        "max_keypoints": 1024,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        logger.info("Loading LANet model...")

        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, self.conf["model_name"]),
        )
        self.net = PointModel(is_test=True)
        state_dict = torch.load(model_path, map_location="cpu")
        self.net.load_state_dict(state_dict["model_state"])
        logger.info("Load LANet model done.")

    def _forward(self, data):
        image = data["image"]
        keypoints, scores, descriptors = self.net(image)
        _, _, Hc, Wc = descriptors.shape

        # Scores & Descriptors
        kpts_score = torch.cat([keypoints, scores], dim=1).view(3, -1).t()
        descriptors = descriptors.view(256, Hc, Wc).view(256, -1).t()

        # Filter based on confidence threshold
        descriptors = descriptors[kpts_score[:, 0] > self.conf["keypoint_threshold"], :]
        kpts_score = kpts_score[kpts_score[:, 0] > self.conf["keypoint_threshold"], :]
        keypoints = kpts_score[:, 1:]
        scores = kpts_score[:, 0]

        idxs = scores.argsort()[-self.conf["max_keypoints"] or None :]
        keypoints = keypoints[idxs, :2]
        descriptors = descriptors[idxs]
        scores = scores[idxs]

        return {
            "keypoints": keypoints[None],
            "scores": scores[None],
            "descriptors": descriptors.T[None],
        }
