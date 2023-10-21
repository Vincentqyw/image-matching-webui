import sys
from pathlib import Path
import subprocess
import torch

from ..utils.base_model import BaseModel
from .. import logger

rord_path = Path(__file__).parent / "../../third_party/RoRD"
sys.path.append(str(rord_path))
from lib.model_test import D2Net as _RoRD
from lib.pyramid import process_multiscale

class RoRD(BaseModel):
    default_conf = {
        "model_name": "rord.pth",
        "checkpoint_dir": rord_path / "models",
        "use_relu": True,
        "multiscale": False,
        "max_keypoints": 1024,
    }
    required_inputs = ["image"]
    weight_urls = {
        "rord.pth": "https://drive.google.com/uc?id=12414ZGKwgPAjNTGtNrlB4VV9l7W76B2o&confirm=t",
    }
    proxy = "http://localhost:1080"

    def _init(self, conf):
        model_path = conf["checkpoint_dir"] / conf["model_name"]
        link = self.weight_urls[conf["model_name"]]
        if not model_path.exists():
            model_path.parent.mkdir(exist_ok=True)
            cmd_wo_proxy = ["gdown", link, "-O", str(model_path)]
            cmd = ["gdown", link, "-O", str(model_path), "--proxy", self.proxy]
            logger.info(
                f"Downloading the RoRD model with `{cmd_wo_proxy}`."
            )
            try:
                subprocess.run(cmd_wo_proxy, check=True)
            except subprocess.CalledProcessError as e:
                logger.info(f"Downloading the RoRD model with `{cmd}`.")
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to download the RoRD model.")
                    raise e
        logger.info("RoRD model loaded.")
        self.net = _RoRD(
            model_file=model_path, use_relu=conf["use_relu"], use_cuda=False
        )

    def _forward(self, data):
        image = data["image"]
        image = image.flip(1)  # RGB -> BGR
        norm = image.new_tensor([103.939, 116.779, 123.68])
        image = image * 255 - norm.view(1, 3, 1, 1)  # caffe normalization

        if self.conf["multiscale"]:
            keypoints, scores, descriptors = process_multiscale(image, self.net)
        else:
            keypoints, scores, descriptors = process_multiscale(
                image, self.net, scales=[1]
            )
        keypoints = keypoints[:, [1, 0]]  # (x, y) and remove the scale

        idxs = scores.argsort()[-self.conf["max_keypoints"] or None :]
        keypoints = keypoints[idxs, :2]
        descriptors = descriptors[idxs]
        scores = scores[idxs]

        return {
            "keypoints": torch.from_numpy(keypoints)[None],
            "scores": torch.from_numpy(scores)[None],
            "descriptors": torch.from_numpy(descriptors.T)[None],
        }
