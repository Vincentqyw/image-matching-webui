import sys
from pathlib import Path

import torch
import torchvision.transforms as tfm
from safetensors.torch import load_file

from .. import logger
from ..utils.base_model import BaseModel
from ..utils.geometry import scale_keypoints

matchformer_path = Path(__file__).parent / "../../third_party/MatchFormer"
sys.path.append(str(matchformer_path))

from model.matchformer import Matchformer as MF
from config.defaultmf import get_cfg_defaults as mf_cfg_defaults


def resize_to_divisible(img, divisible_size):
    _, h, w = img.shape
    new_h = ((h + divisible_size - 1) // divisible_size) * divisible_size
    new_w = ((w + divisible_size - 1) // divisible_size) * divisible_size
    if new_h != h or new_w != w:
        img = tfm.Resize((new_h, new_w), antialias=True)(img)
    return img


def pad_images_to_same_shape(img0, img1):
    _, h0, w0 = img0.shape
    _, h1, w1 = img1.shape
    max_h, max_w = max(h0, h1), max(w0, w1)
    if h0 < max_h or w0 < max_w:
        pad_h0 = max_h - h0
        pad_w0 = max_w - w0
        img0 = torch.nn.functional.pad(img0, (0, pad_w0, 0, pad_h0))
    if h1 < max_h or w1 < max_w:
        pad_h1 = max_h - h1
        pad_w1 = max_w - w1
        img1 = torch.nn.functional.pad(img1, (0, pad_w1, 0, pad_h1))
    return img0, img1


class MatchFormer(BaseModel):
    default_conf = {
        "max_keypoints": 2048,
        "match_threshold": 0.2,
    }
    required_inputs = ["image0", "image1"]
    divisible_size = 32

    def _init(self, conf):
        config = mf_cfg_defaults()
        config.MATCHFORMER.BACKBONE_TYPE = "largela"
        config.MATCHFORMER.SCENS = "outdoor"
        config.MATCHFORMER.RESOLUTION = (8, 2)
        config.MATCHFORMER.COARSE.D_MODEL = 256
        config.MATCHFORMER.COARSE.D_FFN = 256
        config.MATCHFORMER.MATCH_COARSE.THR = conf["match_threshold"]

        # Flatten yacs config keys to lowercase for the model
        def lower_config(cfg):
            if isinstance(cfg, dict):
                return {k.lower(): lower_config(v) for k, v in cfg.items()}
            return cfg

        self.net = MF(config=lower_config(config)["matchformer"])
        weights_path = self._download_model(
            repo_id="vismatch/matchformer",
            filename="matchformer_outdoor-large-LA.safetensors",
        )
        state_dict = load_file(weights_path)
        self.net.load_state_dict(
            {k.replace("matcher.", ""): v for k, v in state_dict.items()}
        )
        self.net = self.net.float().eval()
        logger.info("Loaded MatchFormer model.")

    def _forward(self, data):
        img0 = data["image0"]
        img1 = data["image1"]
        assert img0.shape[0] == 1 and img1.shape[0] == 1
        img0, img1 = img0[0], img1[0]

        _, h0, w0 = img0.shape
        _, h1, w1 = img1.shape

        img0 = resize_to_divisible(img0, self.divisible_size)
        img1 = resize_to_divisible(img1, self.divisible_size)
        H0, W0 = img0.shape[-2:]
        H1, W1 = img1.shape[-2:]

        img0 = tfm.Grayscale()(img0)
        img1 = tfm.Grayscale()(img1)
        img0, img1 = pad_images_to_same_shape(img0, img1)

        batch = {"image0": img0.unsqueeze(0), "image1": img1.unsqueeze(0)}
        self.net(batch)

        mkpts0 = batch["mkpts0_f"]
        mkpts1 = batch["mkpts1_f"]
        mconf = batch.get("mconf", torch.ones(len(mkpts0), device=mkpts0.device))

        mkpts0 = scale_keypoints(mkpts0 + 0.5, [w0 / W0, h0 / H0]) - 0.5
        mkpts1 = scale_keypoints(mkpts1 + 0.5, [w1 / W1, h1 / H1]) - 0.5
        if self.conf["max_keypoints"] > 0 and len(mkpts0) > self.conf["max_keypoints"]:
            idxs = mconf.topk(self.conf["max_keypoints"]).indices
            mkpts0, mkpts1, mconf = mkpts0[idxs], mkpts1[idxs], mconf[idxs]

        return {
            "keypoints0": mkpts0,
            "keypoints1": mkpts1,
            "mconf": mconf,
        }
