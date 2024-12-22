import sys
from pathlib import Path

import numpy as np
import torch

from .. import MODEL_REPO_ID, logger
from ..utils.base_model import BaseModel

thirdparty_path = Path(__file__).parent / "../../third_party"
sys.path.append(str(thirdparty_path))
from omniglue.src import omniglue

omniglue_path = thirdparty_path / "omniglue"


class OmniGlue(BaseModel):
    default_conf = {
        "match_threshold": 0.02,
        "max_keypoints": 2048,
    }
    required_inputs = ["image0", "image1"]
    dino_v2_link_dict = {
        "dinov2_vitb14_pretrain.pth": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
    }

    def _init(self, conf):
        logger.info("Loading OmniGlue model")
        og_model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, "omniglue.onnx"),
        )
        sp_model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, "sp_v6.onnx"),
        )
        dino_model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, "dinov2_vitb14_pretrain.pth"),
        )

        self.net = omniglue.OmniGlue(
            og_export=str(og_model_path),
            sp_export=str(sp_model_path),
            dino_export=str(dino_model_path),
            max_keypoints=self.conf["max_keypoints"],
        )
        logger.info("Loaded OmniGlue model done!")

    def _forward(self, data):
        image0_rgb_np = data["image0"][0].permute(1, 2, 0).cpu().numpy() * 255
        image1_rgb_np = data["image1"][0].permute(1, 2, 0).cpu().numpy() * 255
        image0_rgb_np = image0_rgb_np.astype(np.uint8)  # RGB, 0-255
        image1_rgb_np = image1_rgb_np.astype(np.uint8)  # RGB, 0-255
        match_kp0, match_kp1, match_confidences = self.net.FindMatches(
            image0_rgb_np, image1_rgb_np, self.conf["max_keypoints"]
        )
        # filter matches
        match_threshold = self.conf["match_threshold"]
        keep_idx = []
        for i in range(match_kp0.shape[0]):
            if match_confidences[i] > match_threshold:
                keep_idx.append(i)
        scores = torch.from_numpy(match_confidences[keep_idx]).reshape(-1, 1)
        pred = {
            "keypoints0": torch.from_numpy(match_kp0[keep_idx]),
            "keypoints1": torch.from_numpy(match_kp1[keep_idx]),
            "mconf": scores,
        }

        top_k = self.conf["max_keypoints"]
        if top_k is not None and len(scores) > top_k:
            keep = torch.argsort(scores, descending=True)[:top_k]
            scores = scores[keep]
            pred["keypoints0"], pred["keypoints1"], pred["mconf"] = (
                pred["keypoints0"][keep],
                pred["keypoints1"][keep],
                scores,
            )
        return pred
