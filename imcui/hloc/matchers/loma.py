import dataclasses
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

from .. import logger
from ..utils.base_model import BaseModel

loma_path = Path(__file__).parent / "../../third_party/LoMa/src"
sys.path.append(str(loma_path))

# LoMa's submodules hardcode amp=True and use amp_dtype from loma.device.
# On MPS/CPU, float16/bfloat16 autocast causes dtype mismatches in Conv2d
# and weight dtype mismatches when models cast themselves to amp_dtype.
# We patch loma.device before importing any LoMa submodules so that all
# modules pick up float32 on non-CUDA devices, effectively disabling
# mixed precision without modifying any third_party code.
import loma.device as _loma_device

if not torch.cuda.is_available():
    _loma_device.amp_dtype = torch.float32

from loma import LoMa as LoMaModel, LoMaB, LoMaB128, LoMaL, LoMaG, LoMaR
from loma.device import device as loma_device
from loma.loma import filter_matches, to_pixel_coords

# Also patch the amp_dtype in submodules that imported it via
# "from loma.device import amp_dtype" — these are local bindings
# that don't update when we change _loma_device.amp_dtype.
import loma.detector.dad as _dad
import loma.descriptor.dedode as _dedode

if not torch.cuda.is_available():
    _dad.amp_dtype = torch.float32
    _dedode.amp_dtype = torch.float32


LOMA_CONFIGS = {
    "loma_b": LoMaB,
    "loma_b128": LoMaB128,
    "loma_l": LoMaL,
    "loma_g": LoMaG,
    "loma_r": LoMaR,
}


class LoMa(BaseModel):
    default_conf = {
        "name": "two_view_pipeline",
        "model_name": "loma_b",
        "max_keypoints": 2048,
        "filter_threshold": 0.1,
    }
    required_inputs = [
        "image0",
        "image1",
    ]

    def _init(self, conf):
        model_name = self.conf["model_name"]
        assert model_name in LOMA_CONFIGS, (
            f"Unknown LoMa model: {model_name}, available: {list(LOMA_CONFIGS.keys())}"
        )
        loma_cfg = LOMA_CONFIGS[model_name]
        cfg = loma_cfg()
        if not torch.cuda.is_available():
            cfg = dataclasses.replace(cfg, mp=False)
        logger.info(f"Loading LoMa model ({model_name})")
        self.net = LoMaModel(cfg)
        # Disable amp on submodules to avoid MPS autocast warnings
        # (MPS does not support float32 autocast dtype)
        if not torch.cuda.is_available():
            for module in self.net.modules():
                if hasattr(module, "amp"):
                    module.amp = False
        logger.info("Load LoMa model done.")

    # LoMa internally manages its own device placement via loma.device,
    # which may differ from the global DEVICE (e.g., loma uses MPS while
    # the project uses CPU). Override .to() to keep the model on LoMa's
    # expected device, preventing weight/data device mismatches.
    def to(self, device=None, **kwargs):
        return super().to(loma_device, **kwargs)

    def _forward(self, data):
        # Ensure model is on LoMa's expected device before inference
        self.net.to(loma_device)

        img0 = data["image0"].cpu().numpy().squeeze() * 255
        img1 = data["image1"].cpu().numpy().squeeze() * 255
        img0 = img0.transpose(1, 2, 0)
        img1 = img1.transpose(1, 2, 0)
        img0 = Image.fromarray(img0.astype("uint8"))
        img1 = Image.fromarray(img1.astype("uint8"))

        # Save to temp files so LoMa's internal preprocessing is applied correctly
        with (
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp0,
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp1,
        ):
            img0.save(tmp0.name)
            img1.save(tmp1.name)
            img0_path = tmp0.name
            img1_path = tmp1.name

        num_keypoints = self.conf["max_keypoints"]
        filter_threshold = self.conf["filter_threshold"]

        # Step 1: Detect and describe keypoints separately for each image
        # Use torch.no_grad instead of LoMa's torch.inference_mode to allow
        # the matching forward pass to track gradients internally.
        with torch.no_grad():
            kpts_A, desc_A, h1, w1 = self.net.detect_and_describe(
                img0_path, num_keypoints
            )
            kpts_B, desc_B, h2, w2 = self.net.detect_and_describe(
                img1_path, num_keypoints
            )
            # Clone to detach from inference mode graph
            kpts_A = kpts_A.clone()
            kpts_B = kpts_B.clone()
            desc_A = desc_A.clone()
            desc_B = desc_B.clone()

        # All detected keypoints (for UI "Keypoints" display)
        all_kpts0 = to_pixel_coords(kpts_A[0], h1, w1)
        all_kpts1 = to_pixel_coords(kpts_B[0], h2, w2)

        # Step 2: Match keypoints
        scores = self.net(kpts_A, kpts_B, desc_A, desc_B)["scores"]
        m0, _, _, _ = filter_matches(scores, filter_threshold)

        valid = m0[0] > -1
        idx_A = torch.where(valid)[0]
        idx_B = m0[0][valid]
        matched_kpts0 = to_pixel_coords(kpts_A[0][idx_A], h1, w1)
        matched_kpts1 = to_pixel_coords(kpts_B[0][idx_B], h2, w2)
        mconf = scores[0][idx_A, idx_B]

        pred = {
            "keypoints0": all_kpts0,
            "keypoints1": all_kpts1,
            "mkeypoints0": matched_kpts0,
            "mkeypoints1": matched_kpts1,
            "mconf": mconf,
        }
        return pred
