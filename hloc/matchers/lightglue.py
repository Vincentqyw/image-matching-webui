import kornia.feature as KF
import torch

from .. import MODEL_REPO_ID, logger
from ..utils.base_model import BaseModel


class LightGlue(BaseModel):
    default_conf = {
        "match_threshold": 0.2,
        "filter_threshold": 0.2,
        "width_confidence": 0.99,  # for point pruning
        "depth_confidence": 0.95,  # for early stopping,
        "features": "superpoint",
        "model_name": "superpoint_lightglue.pth",
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "add_scale_ori": False,
    }
    required_inputs = [
        "image0",
        "keypoints0",
        "scores0",
        "descriptors0",
        "image1",
        "keypoints1",
        "scores1",
        "descriptors1",
    ]

    def _init(self, conf):
        features = conf.pop("features")
        logger.info("Loading lightglue model, {}".format(features))

        self.net = KF.LightGlue(features, **conf)
        logger.info("Load lightglue model done.")

    def _forward(self, data):
        input = {}
        input["image0"] = {
            # "image": data["image0"],
            "image_size": torch.tensor(data["image0"].shape[-2:][::-1])[None],
            "keypoints": data["keypoints0"],
            "descriptors": data["descriptors0"].permute(0, 2, 1),
        }
        if "scales0" in data:
            input["image0"] = {**input["image0"], "scales": data["scales0"]}
        if "oris0" in data:
            input["image0"] = {**input["image0"], "oris": data["oris0"]}

        input["image1"] = {
            # "image": data["image1"],
            "image_size": torch.tensor(data["image1"].shape[-2:][::-1])[None],
            "keypoints": data["keypoints1"],
            "descriptors": data["descriptors1"].permute(0, 2, 1),
        }
        if "scales1" in data:
            input["image1"] = {**input["image1"], "scales": data["scales1"]}
        if "oris1" in data:
            input["image1"] = {**input["image1"], "oris": data["oris1"]}
        return self.net(input)
