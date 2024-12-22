import numpy as np
import torch

from ..utils.base_model import BaseModel


# borrow from dedode
def dual_softmax_matcher(
    desc_A: tuple["B", "C", "N"],  # noqa: F821
    desc_B: tuple["B", "C", "M"],  # noqa: F821
    threshold=0.1,
    inv_temperature=20,
    normalize=True,
):
    B, C, N = desc_A.shape
    if len(desc_A.shape) < 3:
        desc_A, desc_B = desc_A[None], desc_B[None]
    if normalize:
        desc_A = desc_A / desc_A.norm(dim=1, keepdim=True)
        desc_B = desc_B / desc_B.norm(dim=1, keepdim=True)
    sim = torch.einsum("b c n, b c m -> b n m", desc_A, desc_B) * inv_temperature
    P = sim.softmax(dim=-2) * sim.softmax(dim=-1)
    mask = torch.nonzero(
        (P == P.max(dim=-1, keepdim=True).values)
        * (P == P.max(dim=-2, keepdim=True).values)
        * (P > threshold)
    )
    mask = mask.cpu().numpy()
    matches0 = np.ones((B, P.shape[-2]), dtype=int) * (-1)
    scores0 = np.zeros((B, P.shape[-2]), dtype=float)
    matches0[:, mask[:, 1]] = mask[:, 2]
    tmp_P = P.cpu().numpy()
    scores0[:, mask[:, 1]] = tmp_P[mask[:, 0], mask[:, 1], mask[:, 2]]
    matches0 = torch.from_numpy(matches0).to(P.device)
    scores0 = torch.from_numpy(scores0).to(P.device)
    return matches0, scores0


class DualSoftMax(BaseModel):
    default_conf = {
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
                data["descriptors0"].shape[:2],
                -1,
                device=data["descriptors0"].device,
            )
            return {
                "matches0": matches0,
                "matching_scores0": torch.zeros_like(matches0),
            }

        matches0, scores0 = dual_softmax_matcher(
            data["descriptors0"],
            data["descriptors1"],
            threshold=self.conf["match_threshold"],
            inv_temperature=self.conf["inv_temperature"],
        )
        return {
            "matches0": matches0,  # 1 x M
            "matching_scores0": scores0,
        }
