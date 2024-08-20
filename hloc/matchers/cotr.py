import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torchvision.transforms import ToPILImage

from .. import logger
from ..utils.base_model import BaseModel

cotr_path = Path(__file__).parent / "../../third_party/COTR"

sys.path.append(str(cotr_path))
from COTR.inference.sparse_engine import SparseEngine
from COTR.models import build_model
from COTR.options.options import *  # noqa: F403
from COTR.options.options_utils import *  # noqa: F403
from COTR.utils import utils as utils_cotr

utils_cotr.fix_randomness(0)
torch.set_grad_enabled(False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class COTR(BaseModel):
    default_conf = {
        "weights": "out/default",
        "match_threshold": 0.2,
        "max_keypoints": -1,
    }
    required_inputs = ["image0", "image1"]

    def _init(self, conf):
        parser = argparse.ArgumentParser()
        set_COTR_arguments(parser)  # noqa: F405
        opt = parser.parse_args()
        opt.command = " ".join(sys.argv)
        model_path = cotr_path / conf["weights"] / "checkpoint.pth.tar"

        if not model_path.exists():
            model_path.parent.mkdir(exist_ok=True, parents=True)
            cached_file = hf_hub_download(
                repo_type="space",
                repo_id="Realcat/image-matching-webui",
                filename="third_party/COTR/{}/{}".format(
                    conf["weights"], "checkpoint.pth.tar"
                ),
            )
            logger.info("Downloaded COTR model succeeed!")
            cmd = [
                "cp",
                str(cached_file),
                str(model_path.parent),
            ]
            subprocess.run(cmd, check=True)
            logger.info(f"Copy model file `{cmd}`.")

        opt.load_weights_path = str(model_path)
        layer_2_channels = {
            "layer1": 256,
            "layer2": 512,
            "layer3": 1024,
            "layer4": 2048,
        }
        opt.dim_feedforward = layer_2_channels[opt.layer]

        model = build_model(opt)
        model = model.to(device)
        weights = torch.load(opt.load_weights_path, map_location="cpu")[
            "model_state_dict"
        ]
        utils_cotr.safe_load_weights(model, weights)
        self.net = model.eval()
        self.to_pil_func = ToPILImage(mode="RGB")

    def _forward(self, data):
        img_a = np.array(self.to_pil_func(data["image0"][0].cpu()))
        img_b = np.array(self.to_pil_func(data["image1"][0].cpu()))
        corrs = SparseEngine(
            self.net, 32, mode="tile"
        ).cotr_corr_multiscale_with_cycle_consistency(
            img_a,
            img_b,
            np.linspace(0.5, 0.0625, 4),
            1,
            max_corrs=self.conf["max_keypoints"],
            queries_a=None,
        )
        pred = {
            "keypoints0": torch.from_numpy(corrs[:, :2]),
            "keypoints1": torch.from_numpy(corrs[:, 2:]),
        }
        return pred
