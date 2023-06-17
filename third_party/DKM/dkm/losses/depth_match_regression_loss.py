from einops.einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from dkm.utils.utils import warp_kpts


class DepthRegressionLoss(nn.Module):
    def __init__(
        self,
        robust=True,
        center_coords=False,
        scale_normalize=False,
        ce_weight=0.01,
        local_loss=True,
        local_dist=4.0,
        local_largest_scale=8,
    ):
        super().__init__()
        self.robust = robust  # measured in pixels
        self.center_coords = center_coords
        self.scale_normalize = scale_normalize
        self.ce_weight = ce_weight
        self.local_loss = local_loss
        self.local_dist = local_dist
        self.local_largest_scale = local_largest_scale

    def geometric_dist(self, depth1, depth2, T_1to2, K1, K2, dense_matches, scale):
        """[summary]

        Args:
            H ([type]): [description]
            scale ([type]): [description]

        Returns:
            [type]: [description]
        """
        b, h1, w1, d = dense_matches.shape
        with torch.no_grad():
            x1_n = torch.meshgrid(
                *[
                    torch.linspace(
                        -1 + 1 / n, 1 - 1 / n, n, device=dense_matches.device
                    )
                    for n in (b, h1, w1)
                ]
            )
            x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(b, h1 * w1, 2)
            mask, x2 = warp_kpts(
                x1_n.double(),
                depth1.double(),
                depth2.double(),
                T_1to2.double(),
                K1.double(),
                K2.double(),
            )
            prob = mask.float().reshape(b, h1, w1)
        gd = (dense_matches - x2.reshape(b, h1, w1, 2)).norm(dim=-1)  # *scale?
        return gd, prob

    def dense_depth_loss(self, dense_certainty, prob, gd, scale, eps=1e-8):
        """[summary]

        Args:
            dense_certainty ([type]): [description]
            prob ([type]): [description]
            eps ([type], optional): [description]. Defaults to 1e-8.

        Returns:
            [type]: [description]
        """
        smooth_prob = prob
        ce_loss = F.binary_cross_entropy_with_logits(dense_certainty[:, 0], smooth_prob)
        depth_loss = gd[prob > 0]
        if not torch.any(prob > 0).item():
            depth_loss = (gd * 0.0).mean()  # Prevent issues where prob is 0 everywhere
        return {
            f"ce_loss_{scale}": ce_loss.mean(),
            f"depth_loss_{scale}": depth_loss.mean(),
        }

    def forward(self, dense_corresps, batch):
        """[summary]

        Args:
            out ([type]): [description]
            batch ([type]): [description]

        Returns:
            [type]: [description]
        """
        scales = list(dense_corresps.keys())
        tot_loss = 0.0
        prev_gd = 0.0
        for scale in scales:
            dense_scale_corresps = dense_corresps[scale]
            dense_scale_certainty, dense_scale_coords = (
                dense_scale_corresps["dense_certainty"],
                dense_scale_corresps["dense_flow"],
            )
            dense_scale_coords = rearrange(dense_scale_coords, "b d h w -> b h w d")
            b, h, w, d = dense_scale_coords.shape
            gd, prob = self.geometric_dist(
                batch["query_depth"],
                batch["support_depth"],
                batch["T_1to2"],
                batch["K1"],
                batch["K2"],
                dense_scale_coords,
                scale,
            )
            if (
                scale <= self.local_largest_scale and self.local_loss
            ):  # Thought here is that fine matching loss should not be punished by coarse mistakes, but should identify wrong matching
                prob = prob * (
                    F.interpolate(prev_gd[:, None], size=(h, w), mode="nearest")[:, 0]
                    < (2 / 512) * (self.local_dist * scale)
                )
            depth_losses = self.dense_depth_loss(dense_scale_certainty, prob, gd, scale)
            scale_loss = (
                self.ce_weight * depth_losses[f"ce_loss_{scale}"]
                + depth_losses[f"depth_loss_{scale}"]
            )  # scale ce loss for coarser scales
            if self.scale_normalize:
                scale_loss = scale_loss * 1 / scale
            tot_loss = tot_loss + scale_loss
            prev_gd = gd.detach()
        return tot_loss
