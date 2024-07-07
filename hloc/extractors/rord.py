import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from hloc import logger

from ..utils.base_model import BaseModel

rord_path = Path(__file__).parent / "../../third_party"
sys.path.append(str(rord_path))
from RoRD.lib.model_test import D2Net as _RoRD
from RoRD.lib.pyramid import process_multiscale


class RoRD(BaseModel):
    default_conf = {
        "model_name": "rord.pth",
        "checkpoint_dir": rord_path / "RoRD" / "models",
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
        link = self.weight_urls[conf["model_name"]]  # noqa: F841

        if not model_path.exists():
            model_path.parent.mkdir(exist_ok=True)
            cached_file_0 = hf_hub_download(
                repo_type="space",
                repo_id="Realcat/image-matching-webui",
                filename="third_party/RoRD/models/d2net.pth",
            )
            cached_file_1 = hf_hub_download(
                repo_type="space",
                repo_id="Realcat/image-matching-webui",
                filename="third_party/RoRD/models/rord.pth",
            )

            subprocess.run(["cp", cached_file_0, model_path], check=True)
            subprocess.run(["cp", cached_file_1, model_path], check=True)

        self.net = _RoRD(
            model_file=model_path, use_relu=conf["use_relu"], use_cuda=False
        )
        logger.info("Load RoRD model done.")

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
