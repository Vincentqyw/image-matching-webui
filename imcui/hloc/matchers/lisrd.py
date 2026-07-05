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

import torch
import torch.nn.functional as F
from kornia.color import rgb_to_grayscale

from .. import logger
from ..extractors.aliked import ALIKED as ALIKEDExtractor
from ..extractors.sift import SIFT as SIFTExtractor
from ..extractors.superpoint import SuperPoint as SuperPointExtractor
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

# Available keypoint detectors with their default configs.
_DETECTOR_BUILDERS = {
    "superpoint": lambda max_kpts: SuperPointExtractor(
        {
            "nms_radius": 4,
            "model_name": "superpoint_v1.pth",
            "keypoint_threshold": 0.005,
            "max_keypoints": max_kpts,
            "remove_borders": 4,
        }
    ),
    "aliked": lambda max_kpts: ALIKEDExtractor(
        {
            "name": "aliked",
            "model_name": "aliked-n16",
            "max_num_keypoints": max_kpts,
            "detection_threshold": 0.2,
        }
    ),
    "sift": lambda max_kpts: SIFTExtractor(
        {
            "max_keypoints": max_kpts,
            "backend": "opencv",
        }
    ),
}


# ---------------------------------------------------------------------------
# Inlined helpers from LISRD's geometry_utils and pytorch_utils.
# ---------------------------------------------------------------------------


def _keypoints_to_grid(keypoints, img_size):
    """Convert (N, 2) keypoints in (row, col) to grid_sample grid [-1, 1]."""
    n_points = keypoints.size(-2)
    device = keypoints.device
    grid = (
        keypoints.float()
        * 2.0
        / torch.tensor(img_size, dtype=torch.float, device=device)
        - 1.0
    )
    return grid[..., [1, 0]].view(-1, n_points, 1, 2)


def _extract_descriptors(keypoints, descriptors, meta_descriptors, img_size):
    """Sample dense descriptor maps at keypoint locations.

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
    """Mutual nearest neighbour matching via meta-weighted descriptor similarity."""
    device = desc1.device
    desc_weights = torch.einsum("nid,mid->nim", meta_desc1, meta_desc2)
    desc_weights = F.softmax(desc_weights, dim=1)
    desc_sims = torch.einsum("nid,mid->nim", desc1, desc2) * desc_weights
    desc_sims = torch.sum(desc_sims, dim=1)  # (N1, N2)

    nn12 = torch.max(desc_sims, dim=1)[1]
    nn21 = torch.max(desc_sims, dim=0)[1]
    ids1 = torch.arange(desc_sims.shape[0], dtype=torch.long, device=device)
    mask = ids1 == nn21[nn12]
    return torch.stack([ids1[mask], nn12[mask]], dim=1)


def _compute_confidence(desc0, desc1, meta0, meta1):
    """Meta-descriptor weighted cosine similarity for matched pairs."""
    desc0 = F.normalize(desc0, dim=2)
    desc1 = F.normalize(desc1, dim=2)
    meta0 = F.normalize(meta0, dim=2)
    meta1 = F.normalize(meta1, dim=2)

    desc_sim = torch.sum(desc0 * desc1, dim=2)  # (M, 4)
    meta_sim = torch.sum(meta0 * meta1, dim=2)  # (M, 4)

    weights = F.softmax(meta_sim, dim=1)
    confidence = torch.sum(desc_sim * weights, dim=1)  # (M,)
    return confidence


# ---------------------------------------------------------------------------
# Matcher class
# ---------------------------------------------------------------------------


class Lisrd(BaseModel):
    """LISRD standalone matcher.

    Detects keypoints with a configurable detector (SuperPoint, ALIKED, or
    SIFT), extracts LISRD descriptors with four invariance combinations, and
    matches them via mutual nearest neighbour weighted by meta-descriptor
    similarity.
    """

    default_conf = {
        "name": "two_view_pipeline",
        "model_name": "lisrd_aachen",
        "max_keypoints": 2048,
        "detector": "superpoint",  # "superpoint", "aliked", or "sift"
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = get_lisrd_model("lisrd")(None, LISRD_CONFIG, self.device)
        self.net.load(weight_file, Mode.EXPORT)
        self.net._net.eval()
        logger.info("Load LISRD model done.")

        # ---- Keypoint detector -----------------------------------------------
        detector_name = self.conf.get("detector", "superpoint")
        if detector_name not in _DETECTOR_BUILDERS:
            raise ValueError(
                f"Unknown LISRD detector: {detector_name}. "
                f"Choose from {list(_DETECTOR_BUILDERS.keys())}."
            )
        self.detector = _DETECTOR_BUILDERS[detector_name](self.conf["max_keypoints"])
        # GPU-capable detectors run on the LISRD device; SIFT stays on CPU.
        if detector_name != "sift":
            self.detector.to(self.device)
        logger.info(f"LISRD detector: {detector_name}")

    def _forward(self, data):
        detector_name = self.conf.get("detector", "superpoint")

        # LISRD expects [0, 255]; WebUI delivers [0, 1].
        img0 = data["image0"] * 255.0
        img1 = data["image1"] * 255.0
        h0, w0 = img0.shape[2], img0.shape[3]
        h1, w1 = img1.shape[2], img1.shape[3]

        # Detector images stay in [0, 1].
        det_img0 = data["image0"]
        det_img1 = data["image1"]

        with torch.no_grad():
            # -- 1. Detect keypoints -------------------------------------------
            if detector_name in ("superpoint", "aliked"):
                # SuperPoint needs grayscale; ALIKED handles RGB natively.
                if detector_name == "superpoint":
                    if det_img0.shape[1] == 3:
                        det_img0 = rgb_to_grayscale(det_img0)
                    if det_img1.shape[1] == 3:
                        det_img1 = rgb_to_grayscale(det_img1)

                kp_out0 = self.detector({"image": det_img0.to(self.device)})
                kp_out1 = self.detector({"image": det_img1.to(self.device)})
                kps0 = kp_out0["keypoints"]
                kps1 = kp_out1["keypoints"]

                # ALIKED returns a list of per-image tensors.
                if isinstance(kps0, list):
                    kps0, kps1 = kps0[0], kps1[0]

                # Guard against zero keypoints.
                if kps0.numel() == 0:
                    kps0 = torch.zeros(0, 2, device=self.device)
                if kps1.numel() == 0:
                    kps1 = torch.zeros(0, 2, device=self.device)
                if kps0.dim() == 1:
                    kps0 = kps0.view(-1, 2)
                if kps1.dim() == 1:
                    kps1 = kps1.view(-1, 2)
            else:
                # SIFT (CPU-based OpenCV).
                kp_out0 = self.detector({"image": det_img0})
                kp_out1 = self.detector({"image": det_img1})
                # SIFT extractor returns (B, N, 2); B=1 here.
                kps0 = kp_out0["keypoints"][0].to(self.device)
                kps1 = kp_out1["keypoints"][0].to(self.device)

            # All detectors return keypoints in (x, y).  Convert to (row, col)
            # for _extract_descriptors, which matches the original LISRD API.
            gpu_kp0 = kps0[:, [1, 0]]
            gpu_kp1 = kps1[:, [1, 0]]

            # -- 2. Extract LISRD descriptors -----------------------------------
            inputs0 = {"image0": img0.to(self.device)}
            outputs0 = self.net._forward(inputs0, Mode.EXPORT, LISRD_CONFIG)
            desc0 = outputs0["descriptors"]
            meta_desc0 = outputs0["meta_descriptors"]

            inputs1 = {"image0": img1.to(self.device)}
            outputs1 = self.net._forward(inputs1, Mode.EXPORT, LISRD_CONFIG)
            desc1 = outputs1["descriptors"]
            meta_desc1 = outputs1["meta_descriptors"]

            # -- 3. Sample descriptors at keypoint locations ---------------------
            sampled_desc0, sampled_meta0 = _extract_descriptors(
                gpu_kp0, desc0, meta_desc0, (h0, w0)
            )
            sampled_desc1, sampled_meta1 = _extract_descriptors(
                gpu_kp1, desc1, meta_desc1, (h1, w1)
            )

            # -- 4. Mutual nearest neighbour matching ----------------------------
            matches = _lisrd_matcher(
                sampled_desc0, sampled_desc1, sampled_meta0, sampled_meta1
            )

            # -- 5. Confidence --------------------------------------------------
            idx0, idx1 = matches[:, 0], matches[:, 1]
            mconf = _compute_confidence(
                sampled_desc0[idx0],
                sampled_desc1[idx1],
                sampled_meta0[idx0],
                sampled_meta1[idx1],
            )

        # -- 6. Pack outputs ----------------------------------------------------
        # kps0/kps1 are already in (x, y) format.
        all_kpts0 = kps0
        all_kpts1 = kps1

        matched_kpts0 = all_kpts0[idx0]
        matched_kpts1 = all_kpts1[idx1]

        return {
            "keypoints0": all_kpts0,
            "keypoints1": all_kpts1,
            "mkeypoints0": matched_kpts0,
            "mkeypoints1": matched_kpts1,
            "mconf": mconf,
        }
