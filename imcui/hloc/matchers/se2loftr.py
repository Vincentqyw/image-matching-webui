import sys
from pathlib import Path

import torch
import torchvision.transforms as tfm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .. import logger
from ..utils.base_model import BaseModel

se2loftr_path = Path(__file__).parent / "../../third_party/SE2LoFTR"
sys.path.append(str(se2loftr_path))

from src.utils.misc import lower_config
from src.loftr import LoFTR
from configs.loftr.outdoor.loftr_ds_e2_dense_8rot import cfg as _rot8_cfg


def resize_to_divisible(img, divisible_size):
    _, h, w = img.shape
    new_h = ((h + divisible_size - 1) // divisible_size) * divisible_size
    new_w = ((w + divisible_size - 1) // divisible_size) * divisible_size
    if new_h != h or new_w != w:
        img = tfm.Resize((new_h, new_w), antialias=True)(img)
    return img


class SE2LoFTR(BaseModel):
    default_conf = {
        "variant": "rot8",
        "max_keypoints": 2048,
        "match_threshold": 0.2,
    }
    required_inputs = ["image0", "image1"]
    divisible_size = 32

    def _init(self, conf):
        cfg = lower_config(_rot8_cfg)["loftr"]

        self.net = LoFTR(cfg)
        weights_path = hf_hub_download(
            repo_id="vismatch/se2loftr",
            filename="se2loftr_rot8.safetensors",
        )
        state_dict = load_file(weights_path)
        self.net.load_state_dict(state_dict)
        self.net = self.net.float().eval()
        logger.info("Loaded SE2-LoFTR model.")

    def _forward(self, data):
        img0 = data["image0"]
        img1 = data["image1"]

        _, h0, w0 = img0.shape
        _, h1, w1 = img1.shape

        img0 = resize_to_divisible(img0, self.divisible_size)
        img1 = resize_to_divisible(img1, self.divisible_size)
        img0 = tfm.Grayscale()(img0)
        img1 = tfm.Grayscale()(img1)

        H0, W0 = img0.shape[-2:]
        H1, W1 = img1.shape[-2:]

        batch = {"image0": img0.unsqueeze(0), "image1": img1.unsqueeze(0)}
        self.net(batch)

        mkpts0 = batch["mkpts0_f"]
        mkpts1 = batch["mkpts1_f"]
        mconf = batch.get("mconf", torch.ones(len(mkpts0), device=mkpts0.device))

        # Rescale back to original image coordinates
        mkpts0[:, 0] *= w0 / W0
        mkpts0[:, 1] *= h0 / H0
        mkpts1[:, 0] *= w1 / W1
        mkpts1[:, 1] *= h1 / H1

        return {
            "keypoints0": mkpts0,
            "keypoints1": mkpts1,
            "mconf": mconf,
        }
