import sys
from pathlib import Path
from ..utils.base_model import BaseModel
from .. import logger, MODEL_REPO_ID

ripe_path = Path(__file__).parent / "../../third_party/RIPE"
sys.path.append(str(ripe_path))

from ripe import vgg_hyper


class RIPE(BaseModel):
    default_conf = {
        "keypoint_threshold": 0.05,
        "max_keypoints": 5000,
        "model_name": "weights_ripe.pth",
    }

    required_inputs = ["image"]

    def _init(self, conf):
        logger.info("Loading RIPE model...")
        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, self.conf["model_name"]),
        )
        self.net = vgg_hyper(Path(model_path))
        logger.info("Loading RIPE model done!")

    def _forward(self, data):
        keypoints, descriptors, scores = self.net.detectAndCompute(
            data["image"], threshold=0.5, top_k=2048
        )

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
