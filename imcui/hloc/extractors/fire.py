import logging
import subprocess
import sys
from pathlib import Path

import torch
import torchvision.transforms as tvf

from ..utils.base_model import BaseModel

logger = logging.getLogger(__name__)
fire_path = Path(__file__).parent / "../../third_party/fire"
sys.path.append(str(fire_path))


import fire_network


class FIRe(BaseModel):
    default_conf = {
        "global": True,
        "asmk": False,
        "model_name": "fire_SfM_120k.pth",
        "scales": [2.0, 1.414, 1.0, 0.707, 0.5, 0.353, 0.25],  # default params
        "features_num": 1000,  # TODO:not supported now
        "asmk_name": "asmk_codebook.bin",  # TODO:not supported now
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

    def _forward(self, data):
        image = self.norm_rgb(data["image"])

        # Feature extraction.
        desc = self.net.forward_global(image, scales=self.scales)

        return {"global_descriptor": desc}
