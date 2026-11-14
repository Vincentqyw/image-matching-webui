import sys
from pathlib import Path

import torch
from PIL import Image

from .. import MODEL_REPO_ID, logger
from ..utils.base_model import BaseModel

redi_match_path = Path(__file__).parent / "../../third_party/REDI-Match"
sys.path.append(str(redi_match_path))

from redimatch.models.model_builder import get_model, install_gp16_cholesky_solver


class RediMatch(BaseModel):
    default_conf = {
        "name": "two_view_pipeline",
        "model_name": "outdoor.pth",
        "resolution": "medium",  # low: 448, medium: 560, high: 672
        "max_keypoints": 2000,
        "symmetric": True,
        "c4": True,
    }
    required_inputs = [
        "image0",
        "image1",
    ]

    def _init(self, conf):
        weights_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename=f"redi_match/{self.conf['model_name']}",
        )
        logger.info(f"Loading REDI-Match model ({self.conf['model_name']})")
        self.net = get_model(
            resolution=self.conf["resolution"],
            symmetric=self.conf["symmetric"],
            upsample_preds=False,
            attenuate_cert=False,
        )
        install_gp16_cholesky_solver(self.net)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)["model"]
        self.net.load_state_dict(state, strict=True)

        if self.conf["c4"]:
            # C4 rotation-invariant matching, as enabled by the upstream demo.
            decoder = self.net.decoder
            decoder.c4_rotation_matching = True
            decoder.c4_perm_reverse = True
            decoder.c4_symmetric_reverse = "align_to_query"
            decoder.c4_hybrid = True
            decoder.c4_hybrid_alpha = 0.35
            decoder.c4_hybrid_temp = 0.05
            decoder.c4_hybrid_margin_keep = 0.03
            decoder.c4_hybrid_disagree_margin = 0.03
            decoder.c4_hybrid_ot_pool_size = 12
            decoder.c4_hybrid_ot_epsilon = 0.07
            decoder.c4_hybrid_ot_iters = 10
        logger.info("Load REDI-Match model done.")

    def _forward(self, data):
        img0 = Image.fromarray(
            (data["image0"][0].cpu().numpy() * 255).transpose(1, 2, 0).astype("uint8")
        )
        img1 = Image.fromarray(
            (data["image1"][0].cpu().numpy() * 255).transpose(1, 2, 0).astype("uint8")
        )
        H0, W0 = img0.size[::-1]
        H1, W1 = img1.size[::-1]

        warp, certainty = self.net.match(img0, img1, device=data["image0"].device)
        if self.conf["symmetric"]:
            # The warp is [A->B | B->A]; sample the A->B half only.
            mid = warp.shape[2] // 2
            warp, certainty = warp[0, :, :mid, :], certainty[0, :, :mid]
        else:
            warp, certainty = warp[0], certainty[0]

        matches, mconf = self.net.sample(
            warp, certainty, num=self.conf["max_keypoints"]
        )
        kpts0, kpts1 = self.net.to_pixel_coordinates(matches, H0, W0, H1, W1)
        return {
            "keypoints0": kpts0,
            "keypoints1": kpts1,
            "mconf": mconf,
        }
