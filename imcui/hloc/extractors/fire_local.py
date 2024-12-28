import subprocess
import sys
from pathlib import Path

import torch
import torchvision.transforms as tvf

from .. import logger
from ..utils.base_model import BaseModel

fire_path = Path(__file__).parent / "../../third_party/fire"

sys.path.append(str(fire_path))


import fire_network

EPS = 1e-6


class FIRe(BaseModel):
    default_conf = {
        "global": True,
        "asmk": False,
        "model_name": "fire_SfM_120k.pth",
        "scales": [2.0, 1.414, 1.0, 0.707, 0.5, 0.353, 0.25],  # default params
        "features_num": 1000,
        "asmk_name": "asmk_codebook.bin",
        "config_name": "eval_fire.yml",
    }
    required_inputs = ["image"]

    # Models exported using
    fire_models = {
        "fire_SfM_120k.pth": "http://download.europe.naverlabs.com/ComputerVision/FIRe/official/fire.pth",
        "fire_imagenet.pth": "http://download.europe.naverlabs.com/ComputerVision/FIRe/pretraining/fire_imagenet.pth",
    }

    def _init(self, conf):
        assert conf["model_name"] in self.fire_models.keys()

        # Config paths
        model_path = fire_path / "model" / conf["model_name"]
        config_path = fire_path / conf["config_name"]  # noqa: F841
        asmk_bin_path = fire_path / "model" / conf["asmk_name"]  # noqa: F841

        # Download the model.
        if not model_path.exists():
            model_path.parent.mkdir(exist_ok=True)
            link = self.fire_models[conf["model_name"]]
            cmd = ["wget", "--quiet", link, "-O", str(model_path)]
            logger.info(f"Downloading the FIRe model with `{cmd}`.")
            subprocess.run(cmd, check=True)

        logger.info("Loading fire model...")

        # Load net
        state = torch.load(model_path)
        state["net_params"]["pretrained"] = None
        net = fire_network.init_network(**state["net_params"])
        net.load_state_dict(state["state_dict"])
        self.net = net

        self.norm_rgb = tvf.Normalize(
            **dict(zip(["mean", "std"], net.runtime["mean_std"]))
        )

        # params
        self.scales = conf["scales"]
        self.features_num = conf["features_num"]

    def _forward(self, data):
        image = self.norm_rgb(data["image"])

        local_desc = self.net.forward_local(
            image, features_num=self.features_num, scales=self.scales
        )

        logger.info(f"output[0].shape = {local_desc[0].shape}\n")

        return {
            # 'global_descriptor': desc
            "local_descriptor": local_desc
        }
