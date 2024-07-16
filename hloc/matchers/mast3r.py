import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as tfm

from .. import logger

mast3r_path = Path(__file__).parent / "../../third_party/mast3r"
sys.path.append(str(mast3r_path))

dust3r_path = Path(__file__).parent / "../../third_party/dust3r"
sys.path.append(str(dust3r_path))

from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.model import AsymmetricMASt3R

from hloc.matchers.duster import Duster

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Mast3r(Duster):
    default_conf = {
        "name": "Mast3r",
        "model_path": mast3r_path
        / "model_weights/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        "max_keypoints": 2000,
        "vit_patch_size": 16,
    }

    def _init(self, conf):
        self.normalize = tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.model_path = self.conf["model_path"]
        self.download_weights()
        self.net = AsymmetricMASt3R.from_pretrained(self.model_path).to(device)
        logger.info("Loaded Mast3r model")

    def download_weights(self):
        url = "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        if not os.path.isfile(self.model_path):
            logger.info("Downloading Mast3r(ViT large)... (takes a while)")
            urllib.request.urlretrieve(url, self.model_path)
            logger.info("Downloading Mast3r(ViT large)... done!")

    def _forward(self, data):
        img0, img1 = data["image0"], data["image1"]
        mean = torch.tensor([0.5, 0.5, 0.5]).to(device)
        std = torch.tensor([0.5, 0.5, 0.5]).to(device)

        img0 = (img0 - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        img1 = (img1 - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

        images = [
            {"img": img0, "idx": 0, "instance": 0},
            {"img": img1, "idx": 1, "instance": 1},
        ]
        pairs = make_pairs(
            images, scene_graph="complete", prefilter=None, symmetrize=True
        )
        output = inference(pairs, self.net, device, batch_size=1)

        # at this stage, you have the raw dust3r predictions
        _, pred1 = output["view1"], output["pred1"]
        _, pred2 = output["view2"], output["pred2"]

        desc1, desc2 = (
            pred1["desc"][1].squeeze(0).detach(),
            pred2["desc"][1].squeeze(0).detach(),
        )

        # find 2D-2D matches between the two images
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1,
            desc2,
            subsample_or_initxy1=2,
            device=device,
            dist="dot",
            block_size=2**13,
        )

        mkpts0 = matches_im0.copy()
        mkpts1 = matches_im1.copy()

        if len(mkpts0) == 0:
            pred = {
                "keypoints0": torch.zeros([0, 2]),
                "keypoints1": torch.zeros([0, 2]),
            }
            logger.warning(f"Matched {0} points")
        else:

            top_k = self.conf["max_keypoints"]
            if top_k is not None and len(mkpts0) > top_k:
                keep = np.round(np.linspace(0, len(mkpts0) - 1, top_k)).astype(
                    int
                )
                mkpts0 = mkpts0[keep]
                mkpts1 = mkpts1[keep]
            pred = {
                "keypoints0": torch.from_numpy(mkpts0),
                "keypoints1": torch.from_numpy(mkpts1),
            }
        return pred
