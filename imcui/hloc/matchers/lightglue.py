import sys
from pathlib import Path

from .. import logger
from ..utils.base_model import BaseModel

lightglue_path = Path(__file__).parent / "../../third_party/LightGlue"
sys.path.append(str(lightglue_path))
from lightglue import LightGlue as LG


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
        logger.info("Loading lightglue model, {}".format(conf["model_name"]))
        feature = conf["features"]
        assert feature in LG.features, feature
        weights_name = conf["model_name"].removesuffix(".pth")
        if weights_name != LG.features[feature]["weights"]:
            # Upstream derives the cache filename from the features entry.
            # Register an alias for checkpoints that are not the default for
            # this feature (e.g. minima_lightglue).
            LG.features[weights_name] = {
                **LG.features[feature],
                "weights": weights_name,
            }
            feature = weights_name
        cache_name = f"{weights_name}_{LG.version.replace('.', '-')}.pth"
        self._download_torch_checkpoint(
            "{}/{}".format(Path(__file__).stem, conf["model_name"]), cache_name
        )
        net_conf = {k: v for k, v in conf.items() if k != "features"}
        net_conf["filter_threshold"] = net_conf["match_threshold"]
        self.net = LG(features=feature, **net_conf)
        logger.info("Load lightglue model done.")

    def _forward(self, data):
        input = {}
        input["image0"] = {
            "image": data["image0"],
            "keypoints": data["keypoints0"],
            "descriptors": data["descriptors0"].permute(0, 2, 1),
        }
        if "scales0" in data:
            input["image0"] = {**input["image0"], "scales": data["scales0"]}
        if "oris0" in data:
            input["image0"] = {**input["image0"], "oris": data["oris0"]}

        input["image1"] = {
            "image": data["image1"],
            "keypoints": data["keypoints1"],
            "descriptors": data["descriptors1"].permute(0, 2, 1),
        }
        if "scales1" in data:
            input["image1"] = {**input["image1"], "scales": data["scales1"]}
        if "oris1" in data:
            input["image1"] = {**input["image1"], "oris": data["oris1"]}
        return self.net(input)
