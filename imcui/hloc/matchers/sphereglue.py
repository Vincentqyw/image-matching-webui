import sys
from pathlib import Path

import numpy as np
import torch

from .. import MODEL_REPO_ID, logger
from ..utils.base_model import BaseModel

# ── third-party path ──────────────────────────────────────────────
sphereglue_path = Path(__file__).parent / "../../third_party/SphereGlue"
sys.path.append(str(sphereglue_path))

from model.sphereglue import SphereGlue as SG  # noqa: E402


class SphereGlue(BaseModel):
    """SphereGlue: GNN-based keypoint matching for spherical images (CVPRW 2023).

    This is a *sparse* matcher — it takes keypoints + descriptors + scores
    produced by a separate feature extractor (e.g. SuperPoint, DISK, ALIKED).

    Because SphereGlue was trained on spherical (equirectangular) images its
    Chebyshev graph-convolution uses 3-D unit-Cartesian coordinates.  We
    approximate this for pinhole images by mapping pixel coordinates →
    spherical (φ, θ) → unit Cartesian (x, y, z), which lets the k-NN graph
    still encode spatial proximity reasonably.
    """

    default_conf = {
        "match_threshold": 0.2,
        "sinkhorn_iterations": 20,
        "max_kpts": 20000,
        "knn": 20,
        "K": 2,  # Chebyshev filter size
        "GNN_layers": ["cross"],
        "aggr": "add",
        # descriptor_dim and output_dim are detector-specific
        "descriptor_dim": 256,  # superpoint / disk / aliked / kp2d
        "output_dim": 512,  # 2 × descriptor_dim
        "model_name": "sphereglue_superpoint.pth",
    }

    required_inputs = [
        "image0",
        "keypoints0",
        "scores0",
        "descriptors0",
        "image1",
        "keypoints1",
        "scores1",
        "descriptors1",
    ]

    def _init(self, conf):
        logger.info("Loading SphereGlue model: %s", conf["model_name"])

        sg_config = {
            "K": conf["K"],
            "GNN_layers": conf["GNN_layers"],
            "match_threshold": conf["match_threshold"],
            "sinkhorn_iterations": conf["sinkhorn_iterations"],
            "aggr": conf["aggr"],
            "knn": conf["knn"],
            "max_kpts": conf["max_kpts"],
            "descriptor_dim": conf["descriptor_dim"],
            "output_dim": conf["output_dim"],
        }

        self.net = SG(sg_config)

        # ── load pretrained weights from HuggingFace ──────────────
        try:
            model_path = self._download_model(
                repo_id=MODEL_REPO_ID,
                filename="{}/{}".format(Path(__file__).stem, conf["model_name"]),
            )
            ckpt = torch.load(model_path, map_location="cpu")
            self.net.load_state_dict(ckpt["MODEL_STATE_DICT"])
            logger.info("Loaded SphereGlue weights from HuggingFace.")
        except Exception as exc:
            logger.warning(
                "Could not load SphereGlue weights from HuggingFace: %s", exc
            )
            logger.warning("SphereGlue will use *random* weights!")

        self.net.eval()
        logger.info("Load SphereGlue model done.")

    # ── helpers ───────────────────────────────────────────────────
    @staticmethod
    def _pixel_to_unit_cartesian(keypoints, img_width, img_height):
        """Convert pixel keypoints to 3-D unit-Cartesian coordinates.

        Pixel → spherical (equirectangular model):
            θ = (1 - (x + 0.5) / W) · 2π
            φ = (y + 0.5) · π / H
        Spherical → unit Cartesian:
            x = cos θ · sin φ
            y = sin θ · sin φ
            z = cos φ
        """
        if keypoints.dim() == 3:
            kpts = keypoints[0]  # (N, 2)
        else:
            kpts = keypoints

        kpts_np = kpts.cpu().numpy()
        x_px, y_px = kpts_np[:, 0], kpts_np[:, 1]

        theta = (1.0 - (x_px + 0.5) / img_width) * (2.0 * np.pi)
        phi = (y_px + 0.5) * np.pi / img_height

        phi_t = torch.from_numpy(phi).float().to(keypoints.device)
        theta_t = torch.from_numpy(theta).float().to(keypoints.device)

        x = torch.cos(theta_t) * torch.sin(phi_t)
        y = torch.sin(theta_t) * torch.sin(phi_t)
        z = torch.cos(phi_t)

        return torch.stack((x, y, z), dim=1).unsqueeze(0)  # (1, N, 3)

    # ── forward ───────────────────────────────────────────────────
    def _forward(self, data):
        _, _, h0, w0 = data["image0"].shape
        _, _, h1, w1 = data["image1"].shape

        # Convert pixel coordinates → unit Cartesian on the sphere
        ucart0 = self._pixel_to_unit_cartesian(data["keypoints0"], w0, h0)
        ucart1 = self._pixel_to_unit_cartesian(data["keypoints1"], w1, h1)

        # Descriptors arrive as (1, D, N) — transpose to (1, N, D)
        desc0 = data["descriptors0"]
        desc1 = data["descriptors1"]
        if desc0.dim() == 3:
            desc0 = desc0.transpose(1, 2)
        if desc1.dim() == 3:
            desc1 = desc1.transpose(1, 2)

        sg_input = {
            "h1": desc0,
            "h2": desc1,
            "unitCartesian1": ucart0,
            "unitCartesian2": ucart1,
            "scores1": data["scores0"].reshape(1, -1),
            "scores2": data["scores1"].reshape(1, -1),
        }

        return self.net(sg_input)
