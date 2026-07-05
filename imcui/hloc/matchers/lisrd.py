"""Matcher wrapper for LISRD — Online Invariance Selection for Local Descriptors.

LISRD extracts multiple descriptors with different invariances (rotation
variant/invariant × illumination variant/invariant) and uses a learned
meta-descriptor to select the best invariance at match time.

References:
    Pautrat et al., "Online Invariance Selection for Local Feature Descriptors",
    ECCV 2020.  https://arxiv.org/abs/2007.08988
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .. import logger
from ..utils.base_model import BaseModel

lisrd_path = Path(__file__).parent / "../../third_party/LISRD"
sys.path.append(str(lisrd_path))

from lisrd.models import get_model as get_lisrd_model  # noqa: E402
from lisrd.models.base_model import Mode  # noqa: E402

MODEL_REPO_ID = "Realcat/imcui_checkpoints"

LISRD_CONFIG: dict = {
    "name": "lisrd",
    "desc_size": 128,
    "tile": 3,
    "n_clusters": 8,
    "meta_desc_dim": 128,
    "learning_rate": 0.001,
    "compute_meta_desc": True,
    "freeze_local_desc": False,
}


# ---------------------------------------------------------------------------
# Inlined helpers from LISRD's geometry_utils and pytorch_utils.
# We inline these to avoid importing lisrd.utils.geometry_utils, which pulls in
# homographies → keypoint_detectors → super_point_magic_leap (unused here).
# ---------------------------------------------------------------------------

def _keypoints_to_grid(keypoints, img_size):
    """Convert (N, 2) keypoints in (row, col) to grid_sample grid [-1, 1]."""
    n_points = keypoints.size(-2)
    device = keypoints.device
    grid = keypoints.float() * 2.0 / torch.tensor(
        img_size, dtype=torch.float, device=device
    ) - 1.0
    return grid[..., [1, 0]].view(-1, n_points, 1, 2)


def _extract_descriptors(keypoints, descriptors, meta_descriptors, img_size):
    """Sample dense descriptor maps at keypoint locations.

    Adapted from ``lisrd.utils.geometry_utils.extract_descriptors``.

    Args:
        keypoints: ``(N, 2)`` tensor in (row, col) pixel coordinates.
        descriptors: dict mapping variance name → ``(1, C, H, W)`` tensor.
        meta_descriptors: dict mapping variance name → ``(1, C, H, W)`` tensor.
        img_size: ``(H, W)`` tuple.

    Returns:
        ``(descs, meta_descs)`` — both ``(N, 4, D)`` tensors.
    """
    grid = _keypoints_to_grid(keypoints, img_size)

    descs = []
    for k in descriptors:
        desc = F.normalize(
            F.grid_sample(descriptors[k], grid, align_corners=False), dim=1
        )[0, :, :, 0].t()
        descs.append(desc)

    meta_descs = []
    for k in meta_descriptors:
        meta_desc = F.normalize(
            F.grid_sample(meta_descriptors[k], grid, align_corners=False),
            dim=1,
        )[0, :, :, 0].t()
        meta_descs.append(meta_desc)

    return torch.stack(descs, dim=1), torch.stack(meta_descs, dim=1)


def _lisrd_matcher(desc1, desc2, meta_desc1, meta_desc2):
    """Mutual nearest neighbour matching via meta-weighted descriptor similarity.

    Adapted from ``lisrd.utils.geometry_utils.lisrd_matcher``.
    """
    device = desc1.device
    # (N1, N2, 4) — per-variance meta similarity
    desc_weights = torch.einsum("nid,mid->nim", meta_desc1, meta_desc2)
    desc_weights = F.softmax(desc_weights, dim=1)
    # (N1, N2, 4) — per-variance descriptor similarity
    desc_sims = torch.einsum("nid,mid->nim", desc1, desc2) * desc_weights
    desc_sims = torch.sum(desc_sims, dim=1)  # (N1, N2)

    nn12 = torch.max(desc_sims, dim=1)[1]
    nn21 = torch.max(desc_sims, dim=0)[1]
    ids1 = torch.arange(desc_sims.shape[0], dtype=torch.long, device=device)
    mask = ids1 == nn21[nn12]
    return torch.stack([ids1[mask], nn12[mask]], dim=1)


# ---------------------------------------------------------------------------
# Matcher class
# ---------------------------------------------------------------------------


class Lisrd(BaseModel):
    """LISRD standalone matcher.

    Internally detects SIFT keypoints, extracts LISRD descriptors with four
    invariance combinations, and matches them via mutual nearest neighbour
    weighted by meta-descriptor similarity.
    """

    default_conf = {
        "name": "two_view_pipeline",
        "model_name": "lisrd_vidit",
        "max_keypoints": 2048,
    }
    required_inputs = [
        "image0",
        "image1",
    ]

    def _init(self, conf):
        model_name = self.conf["model_name"]

        weight_file = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename=f"lisrd/{model_name}.pth",
        )

        logger.info(f"Loading LISRD model ({model_name})")
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.net = get_lisrd_model("lisrd")(None, LISRD_CONFIG, self.device)
        self.net.load(weight_file, Mode.EXPORT)
        self.net._net.eval()
        logger.info("Load LISRD model done.")

    def _forward(self, data):
        num_keypoints = self.conf["max_keypoints"]

        # WebUI delivers images as float32 [0, 1]; LISRD expects [0, 255].
        img0 = data["image0"] * 255.0
        img1 = data["image1"] * 255.0
        h0, w0 = img0.shape[2], img0.shape[3]
        h1, w1 = img1.shape[2], img1.shape[3]

        with torch.no_grad():
            # -- 1. Detect keypoints with OpenCV SIFT ------------------------
            kp0 = self._detect_sift_keypoints(img0, num_keypoints)
            kp1 = self._detect_sift_keypoints(img1, num_keypoints)

            # -- 2. Extract LISRD descriptors --------------------------------
            inputs0 = {"image0": img0.to(self.device)}
            outputs0 = self.net._forward(inputs0, Mode.EXPORT, LISRD_CONFIG)
            desc0 = outputs0["descriptors"]
            meta_desc0 = outputs0["meta_descriptors"]

            inputs1 = {"image0": img1.to(self.device)}
            outputs1 = self.net._forward(inputs1, Mode.EXPORT, LISRD_CONFIG)
            desc1 = outputs1["descriptors"]
            meta_desc1 = outputs1["meta_descriptors"]

            # -- 3. Sample descriptors at keypoint locations -----------------
            gpu_kp0 = torch.tensor(
                kp0[:, :2], dtype=torch.float, device=self.device
            )
            gpu_kp1 = torch.tensor(
                kp1[:, :2], dtype=torch.float, device=self.device
            )

            sampled_desc0, sampled_meta0 = _extract_descriptors(
                gpu_kp0, desc0, meta_desc0, (h0, w0)
            )
            sampled_desc1, sampled_meta1 = _extract_descriptors(
                gpu_kp1, desc1, meta_desc1, (h1, w1)
            )

            # -- 4. Mutual nearest neighbour matching ------------------------
            matches = _lisrd_matcher(
                sampled_desc0, sampled_meta0, sampled_desc1, sampled_meta1
            )

            # -- 5. Confidence from meta-weighted descriptor similarity ------
            idx0, idx1 = matches[:, 0], matches[:, 1]
            mconf = _compute_confidence(
                sampled_desc0[idx0],
                sampled_desc1[idx1],
                sampled_meta0[idx0],
                sampled_meta1[idx1],
            )

        # -- 6. Pack outputs (coordinates in pixel space, (x,y) = (col,row))
        # kp is (row, col, response); swap to (x, y).
        all_kpts0 = torch.tensor(
            kp0[:, [1, 0]], dtype=torch.float, device=self.device
        )
        all_kpts1 = torch.tensor(
            kp1[:, [1, 0]], dtype=torch.float, device=self.device
        )

        matched_kpts0 = all_kpts0[idx0]
        matched_kpts1 = all_kpts1[idx1]

        return {
            "keypoints0": all_kpts0,
            "keypoints1": all_kpts1,
            "mkeypoints0": matched_kpts0,
            "mkeypoints1": matched_kpts1,
            "mconf": mconf,
        }

    @staticmethod
    def _detect_sift_keypoints(img_tensor, num_keypoints):
        """OpenCV SIFT keypoint detection.

        Args:
            img_tensor: ``(1, 3, H, W)`` float32 tensor in [0, 255].
            num_keypoints: maximum number of keypoints.

        Returns:
            ``np.ndarray`` of shape ``(N, 3)`` with columns (row, col, response).
        """
        img_np = img_tensor[0].cpu().numpy().transpose(1, 2, 0)  # HWC
        img_np = np.uint8(img_np)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        sift = cv2.SIFT_create(
            nfeatures=num_keypoints, contrastThreshold=0.01
        )
        keypoints = sift.detect(gray, None)

        if len(keypoints) == 0:
            return np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        return np.array(
            [[k.pt[1], k.pt[0], k.response] for k in keypoints],
            dtype=np.float32,
        )


def _compute_confidence(desc0, desc1, meta0, meta1):
    """Meta-descriptor weighted cosine similarity for matched pairs.

    Args:
        desc0, desc1: ``(M, 4, D)`` — descriptors per invariance type.
        meta0, meta1: ``(M, 4, D')`` — meta descriptors per invariance type.

    Returns:
        ``(M,)`` tensor of confidence scores.
    """
    desc0 = F.normalize(desc0, dim=2)
    desc1 = F.normalize(desc1, dim=2)
    meta0 = F.normalize(meta0, dim=2)
    meta1 = F.normalize(meta1, dim=2)

    desc_sim = torch.sum(desc0 * desc1, dim=2)  # (M, 4)
    meta_sim = torch.sum(meta0 * meta1, dim=2)  # (M, 4)

    weights = F.softmax(meta_sim, dim=1)
    confidence = torch.sum(desc_sim * weights, dim=1)  # (M,)
    return confidence
