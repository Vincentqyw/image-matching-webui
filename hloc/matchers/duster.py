import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as tfm

from .. import MODEL_REPO_ID, logger
from ..utils.base_model import BaseModel

duster_path = Path(__file__).parent / "../../third_party/dust3r"
sys.path.append(str(duster_path))

from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Duster(BaseModel):
    default_conf = {
        "name": "Duster3r",
        "model_name": "duster_vit_large.pth",
        "max_keypoints": 3000,
        "vit_patch_size": 16,
    }

    def _init(self, conf):
        self.normalize = tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, self.conf["model_name"]),
        )
        self.net = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
        logger.info("Loaded Dust3r model")

    def preprocess(self, img):
        # the super-class already makes sure that img0,img1 have
        # same resolution and that h == w
        _, h, _ = img.shape
        imsize = h
        if not ((h % self.vit_patch_size) == 0):
            imsize = int(self.vit_patch_size * round(h / self.vit_patch_size, 0))
            img = tfm.functional.resize(img, imsize, antialias=True)

        _, new_h, new_w = img.shape
        if not ((new_w % self.vit_patch_size) == 0):
            safe_w = int(self.vit_patch_size * round(new_w / self.vit_patch_size, 0))
            img = tfm.functional.resize(img, (new_h, safe_w), antialias=True)

        img = self.normalize(img).unsqueeze(0)

        return img

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
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)
        # retrieve useful values from scene:
        imgs = scene.imgs
        confidence_masks = scene.get_masks()
        pts3d = scene.get_pts3d()
        pts2d_list, pts3d_list = [], []
        for i in range(2):
            conf_i = confidence_masks[i].cpu().numpy()
            pts2d_list.append(
                xy_grid(*imgs[i].shape[:2][::-1])[conf_i]
            )  # imgs[i].shape[:2] = (H, W)
            pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])

        if len(pts3d_list[1]) == 0:
            pred = {
                "keypoints0": torch.zeros([0, 2]),
                "keypoints1": torch.zeros([0, 2]),
            }
            logger.warning(f"Matched {0} points")
        else:
            reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(
                *pts3d_list
            )
            logger.info(f"Found {num_matches} matches")
            mkpts1 = pts2d_list[1][reciprocal_in_P2]
            mkpts0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]
            top_k = self.conf["max_keypoints"]
            if top_k is not None and len(mkpts0) > top_k:
                keep = np.round(np.linspace(0, len(mkpts0) - 1, top_k)).astype(int)
                mkpts0 = mkpts0[keep]
                mkpts1 = mkpts1[keep]
            pred = {
                "keypoints0": torch.from_numpy(mkpts0),
                "keypoints1": torch.from_numpy(mkpts1),
            }
        return pred
