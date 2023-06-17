import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class NormedCorrelationKernel(nn.Module):  # similar to softmax kernel
    def __init__(self):
        super().__init__()

    def __call__(self, x, y, eps=1e-6):
        c = torch.einsum("bnd,bmd->bnm", x, y) / (
            x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps
        )
        return c


class NormedCorr(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.corr = NormedCorrelationKernel()

    def reshape(self, x):
        return rearrange(x, "b d h w -> b (h w) d")

    def forward(self, x, y, **kwargs):
        b, c, h, w = y.shape
        assert x.shape == y.shape
        x, y = self.reshape(x), self.reshape(y)
        corr_xy = self.corr(x, y)
        corr_xy_flat = rearrange(corr_xy, "b (h w) c -> b c h w", h=h, w=w)
        return corr_xy_flat
