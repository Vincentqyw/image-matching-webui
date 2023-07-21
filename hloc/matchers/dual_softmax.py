import torch

from ..utils.base_model import BaseModel
import numpy as np


def dual_softmax_matcher(
    desc_A: tuple["B", "N", "C"],
    desc_B: tuple["B", "M", "C"],
    threshold=0.1,
    inv_temperature=1,
    normalize=True,
):
    if len(desc_A.shape) < 3:
        desc_A, desc_B = desc_A[None], desc_B[None]
    desc_A = desc_A.permute(0, 2, 1)
    desc_B = desc_B.permute(0, 2, 1)
    if normalize:
        desc_A = desc_A / desc_A.norm(dim=-1, keepdim=True)
        desc_B = desc_B / desc_B.norm(dim=-1, keepdim=True)
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    else:
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    P = corr.softmax(dim=-2) * corr.softmax(dim=-1)
    inds = torch.nonzero(
        (P == P.max(dim=-1, keepdim=True).values)
        * (P == P.max(dim=-2, keepdim=True).values)
        * (P > threshold)
    )
    inds = inds.cpu().numpy()
    matches0 = np.ones((1, P.shape[-2]), dtype=int) * (-1)
    matches0[:, inds[:, 1]] = inds[:, 2]
    matches0 = torch.from_numpy(matches0).to(P.device)
    return matches0


class DualSoftMax(BaseModel):
    default_conf = {
        "ratio_threshold": None,
        "distance_threshold": None,
        "do_mutual_check": True,
        "match_threshold": 0.2,
        "inv_temperature": 20,
    }
    # shape: B x DIM x M
    required_inputs = ["descriptors0", "descriptors1"]

    def _init(self, conf):
        pass

    def _forward(self, data):
        if data["descriptors0"].size(-1) == 0 or data["descriptors1"].size(-1) == 0:
            matches0 = torch.full(
                data["descriptors0"].shape[:2], -1, device=data["descriptors0"].device
            )
            return {
                "matches0": matches0,
                "matching_scores0": torch.zeros_like(matches0),
            }

        matches0 = dual_softmax_matcher(
            data["descriptors0"],
            data["descriptors1"],
            threshold=self.conf["match_threshold"],
            inv_temperature=self.conf["inv_temperature"],
        )
        return {
            "matches0": matches0,  # 1 x M
            "matching_scores0": torch.ones_like(matches0),
        }
