import os
import sys
import types
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import logger
from ..utils.base_model import BaseModel

clidd_path = Path(__file__).parent / "../../third_party/CLIDD"
sys.path.insert(0, str(clidd_path))

# --------------------------------------------------------------------------
# Handle triton dependency — CLIDD uses triton for a CUDA-accelerated
# deformable sampling kernel.  On platforms without triton (CPU, MPS,
# Windows, non-Linux CUDA) we inject a pure-PyTorch fallback that is
# functionally equivalent but slower.
# --------------------------------------------------------------------------
try:
    import triton  # noqa: F401
    import triton.language  # noqa: F401
except ImportError:
    _mock_plugin = types.ModuleType("model.triton_plugin")

    def _fallback_deformable_sample_project(
        input, grid, weight, bias, *,
        is_input_nhwc=False,
        align_corners=False,
    ):
        """Pure-PyTorch equivalent of the triton deformable_sample_project.

        Parameters
        ----------
        input : (B, C_in, H, W)  or  (B, H, W, C_in) when *is_input_nhwc*
        grid  : (B, N, M, 2)          sampling locations in [-1, 1]
        weight: (C_out, C_in, 1, M)   projection weights (Conv2d kernel)
        bias  : (C_out,) or None
        """
        if is_input_nhwc:
            input_chw = input.permute(0, 3, 1, 2).contiguous()
        else:
            input_chw = input

        B, N, M_grid, _ = grid.shape
        C_out, C_in, _, W_weight = weight.shape
        assert W_weight == M_grid, (
            f"Weight last dim {W_weight} != grid M {M_grid}"
        )

        output = torch.zeros(
            B, N, C_out, device=input_chw.device, dtype=input_chw.dtype,
        )

        for m in range(M_grid):
            grid_m = grid[:, :, m, :].unsqueeze(1)  # (B, 1, N, 2)
            sampled = F.grid_sample(
                input_chw, grid_m,
                mode="bilinear", padding_mode="zeros",
                align_corners=align_corners,
            )
            sampled = sampled.squeeze(2).permute(0, 2, 1)  # (B, N, C_in)
            w_m = weight[:, :, 0, m]                        # (C_out, C_in)
            output += sampled @ w_m.T                      # (B, N, C_out)

        if bias is not None:
            output = output + bias.view(1, 1, -1)

        return output

    _mock_plugin.deformable_sample_project = _fallback_deformable_sample_project
    sys.modules["model.triton_plugin"] = _mock_plugin

from clidd import CLIDD as _CLIDD  # noqa: E402

# All CLIDD model variants defined in clidd.py
CLIDD_VARIANTS = list(_CLIDD.cfgs.keys())  # ['A48','N64','T64','S64','M64','L64','G128','E128','U128']

_WEIGHT_BASE_URL = "https://github.com/HITCSC/CLIDD/releases/download/v1.0"


def _download_weight(model_name: str) -> Path:
    """Download a CLIDD weight from GitHub releases and cache locally."""
    cache_dir = Path(torch.hub.get_dir()) / "CLIDD_weights"
    cache_dir.mkdir(parents=True, exist_ok=True)
    weight_file = cache_dir / f"{model_name}.pth"

    if not weight_file.exists():
        url = f"{_WEIGHT_BASE_URL}/{model_name}.pth"
        logger.info(f"Downloading CLIDD weights: {url}")
        urllib.request.urlretrieve(url, weight_file)

    return weight_file


class CLIDD(BaseModel):
    """CLIDD: Cross-Layer Independent Deformable Description.

    A high-performance local feature representation method that supports
    a wide range of model sizes from 0.004M to 4.4M parameters.

    Paper: https://arxiv.org/abs/2601.09230
    Repo:  https://github.com/HITCSC/CLIDD
    """

    default_conf = {
        "name": "two_view_pipeline",
        "model_name": "A48",
        "top_k": 4096,
        "radius": 2,
        "score_thresh": -5,
        "match_beta": 20.0,
    }
    required_inputs = [
        "image0",
        "image1",
    ]

    def _init(self, conf):
        model_name = self.conf["model_name"]
        if model_name not in CLIDD_VARIANTS:
            raise ValueError(
                f"Unknown CLIDD variant: '{model_name}'. "
                f"Available: {CLIDD_VARIANTS}"
            )

        # Download weights to cache
        weight_path = _download_weight(model_name)
        state_dict = torch.load(weight_path, map_location="cpu")

        top_k = self.conf["top_k"]
        radius = self.conf["radius"]
        score_thresh = self.conf["score_thresh"]

        # CLIDD's __init__ calls torch.load('./weights/{cfg}.pth', 'cpu').
        # We chdir to the clidd_path so the relative path resolves correctly,
        # then override with our cached state dict.
        _saved_cwd = os.getcwd()
        try:
            os.chdir(str(clidd_path))
            self.net = _CLIDD(
                model_name, top_k=top_k, radius=radius, score=score_thresh,
            ).eval()
        finally:
            os.chdir(_saved_cwd)

        # Override with the cached weights (in case the local file was stale)
        self.net.model.load_state_dict(state_dict)

        logger.info(f"Load CLIDD({model_name}) model done.")

    def _forward(self, data):
        match_beta = self.conf["match_beta"]

        img0 = data["image0"]
        img1 = data["image1"]
        B = img0.shape[0]

        # Run CLIDD on each image — CLIDD.forward has @torch.inference_mode()
        out0 = self.net(img0)
        out1 = self.net(img1)

        all_mkpts0 = []
        all_mkpts1 = []
        all_mconf = []
        all_kpts0 = []
        all_kpts1 = []

        for b in range(B):
            kpts0 = out0[b]["keypoints"]        # (N0, 2) in pixel coords
            kpts1 = out1[b]["keypoints"]        # (N1, 2) in pixel coords
            desc0 = out0[b]["descriptors"]       # (N0, D)
            desc1 = out1[b]["descriptors"]       # (N1, D)

            # Store all detected keypoints
            all_kpts0.append(kpts0)
            all_kpts1.append(kpts1)

            # Match descriptors using CLIDD's built-in mutual-nearest-neighbor
            idxs0, idxs1 = self.net.match(desc0, desc1, beta=match_beta)

            if len(idxs0) > 0:
                mkpts0 = kpts0[idxs0]
                mkpts1 = kpts1[idxs1]
                # Compute cosine similarity as match confidence
                cossim = (desc0[idxs0] * desc1[idxs1]).sum(dim=-1)
                mconf = (cossim + 1) / 2  # map [-1, 1] → [0, 1]
            else:
                mkpts0 = kpts0.new_zeros((0, 2))
                mkpts1 = kpts1.new_zeros((0, 2))
                mconf = kpts0.new_zeros((0,))

            all_mkpts0.append(mkpts0)
            all_mkpts1.append(mkpts1)
            all_mconf.append(mconf)

        # Return results — B is always 1 in the WebUI, so we squeeze.
        pred = {
            "keypoints0": all_kpts0[0],
            "keypoints1": all_kpts1[0],
            "mkeypoints0": all_mkpts0[0],
            "mkeypoints1": all_mkpts1[0],
            "mconf": all_mconf[0],
        }
        return pred
