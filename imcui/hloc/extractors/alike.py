import sys
from pathlib import Path

import torch

from hloc import MODEL_REPO_ID, logger

from ..utils.base_model import BaseModel

alike_path = Path(__file__).parent / "../../third_party/ALIKE"
sys.path.append(str(alike_path))
from alike import ALike as Alike_
from alike import configs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Alike(BaseModel):
    default_conf = {
        "model_name": "alike-t",  # 'alike-t', 'alike-s', 'alike-n', 'alike-l'
        "use_relu": True,
        "multiscale": False,
        "max_keypoints": 1000,
        "detection_threshold": 0.5,
        "top_k": -1,
        "sub_pixel": False,
    }

    required_inputs = ["image"]

    def _init(self, conf):
        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}.pth".format(Path(__file__).stem, self.conf["model_name"]),
        )
        logger.info("Loaded Alike model from {}".format(model_path))
        configs[conf["model_name"]]["model_path"] = model_path
        self.net = Alike_(
            **configs[conf["model_name"]],
            device=device,
            top_k=conf["top_k"],
            scores_th=conf["detection_threshold"],
            n_limit=conf["max_keypoints"],
        )
        logger.info("Load Alike model done.")

    def _forward(self, data):
        image = data["image"]
        image = image.permute(0, 2, 3, 1).squeeze()
        image = image.cpu().numpy() * 255.0
        pred = self.net(image, sub_pixel=self.conf["sub_pixel"])

        keypoints = pred["keypoints"]
        descriptors = pred["descriptors"]
        scores = pred["scores"]

        return {
            "keypoints": torch.from_numpy(keypoints)[None],
            "scores": torch.from_numpy(scores)[None],
            "descriptors": torch.from_numpy(descriptors.T)[None],
        }
