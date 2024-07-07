import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from hloc import logger

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
    weight_urls = {
        "DarkFeat.pth": "https://drive.google.com/uc?id=1Thl6m8NcmQ7zSAF-1_xaFs3F4H8UU6HX&confirm=t",
    }
    proxy = "http://localhost:1080"
    required_inputs = ["image"]

    def _init(self, conf):
        cached_file = hf_hub_download(
            repo_type="space",
            repo_id="Realcat/image-matching-webui",
            filename="third_party/DarkFeat/checkpoints/DarkFeat.pth",
        )

        self.net = DarkFeat_(cached_file)
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
