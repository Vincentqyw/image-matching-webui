import sys
from pathlib import Path

import torch

from hloc import MODEL_REPO_ID, logger

from ..utils.base_model import BaseModel

d2net_path = Path(__file__).parent / "../../third_party/d2net"
sys.path.append(str(d2net_path))
from lib.model_test import D2Net as _D2Net
from lib.pyramid import process_multiscale


class D2Net(BaseModel):
    default_conf = {
        "model_name": "d2_tf.pth",
        "checkpoint_dir": d2net_path / "models",
        "use_relu": True,
        "multiscale": False,
        "max_keypoints": 1024,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        logger.info("Loading D2Net model...")
        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, self.conf["model_name"]),
        )
        logger.info(f"Loading model from {model_path}...")
        self.net = _D2Net(
            model_file=model_path, use_relu=conf["use_relu"], use_cuda=False
        )
        logger.info("Load D2Net model done.")

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
