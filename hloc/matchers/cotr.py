import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import ToPILImage

from hloc import DEVICE, MODEL_REPO_ID

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / "../../third_party/COTR"))
from COTR.inference.sparse_engine import SparseEngine
from COTR.models import build_model
from COTR.options.options import *  # noqa: F403
from COTR.options.options_utils import *  # noqa: F403
from COTR.utils import utils as utils_cotr

utils_cotr.fix_randomness(0)
torch.set_grad_enabled(False)


class COTR(BaseModel):
    default_conf = {
        "weights": "out/default",
        "match_threshold": 0.2,
        "max_keypoints": -1,
        "model_name": "checkpoint.pth.tar",
    }
    required_inputs = ["image0", "image1"]

    def _init(self, conf):
        parser = argparse.ArgumentParser()
        set_COTR_arguments(parser)  # noqa: F405
        opt = parser.parse_args()
        opt.command = " ".join(sys.argv)
        opt.load_weights_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, self.conf["model_name"]),
        )

        layer_2_channels = {
            "layer1": 256,
            "layer2": 512,
            "layer3": 1024,
            "layer4": 2048,
        }
        opt.dim_feedforward = layer_2_channels[opt.layer]

        model = build_model(opt)
        model = model.to(DEVICE)
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
