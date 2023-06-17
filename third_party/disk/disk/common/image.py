import torch, math, warnings, imageio
import torch.nn.functional as F
import numpy as np

from torch_dimcheck import dimchecked

@dimchecked
def _rescale(tensor: ['C', 'H', 'W'], size) -> ['C', 'h', 'w']:
    return F.interpolate(
        tensor.unsqueeze(0),
        size=size,
        mode='bilinear',
        align_corners=False,
    ).squeeze(0)

@dimchecked
def _pad(tensor: ['C', 'H', 'W'], size, value=0.):
    xpad = size[1] - tensor.shape[2]
    ypad = size[0] - tensor.shape[1]

    # not that F.pad takes sizes starting from the last dimension
    padded = F.pad(
        tensor,
        (0, xpad, 0, ypad),
        mode='constant',
        value=value
    )

    assert padded.shape[1:] == tuple(size)
    return padded

class Image:
    @dimchecked
    def __init__(
        self,
        K     : [3, 3],
        R     : [3, 3],
        T     : [3],
        bitmap: [3, 'H', 'W'],
        depth, #[1, 'H', 'W'],
        bitmap_path: str
    ):
        self.K = K
        self.R = R
        self.T = T

        self.bitmap = bitmap
        self.depth  = depth

        # save bitmap path for potential debugging purposes
        self.bitmap_path = bitmap_path

    @property
    def K_inv(self):
        return self.K.inverse()

    @property
    def hwc(self):
        return self.bitmap.permute(1, 2, 0)

    @property
    def shape(self):
        return self.bitmap.shape[1:]

    def scale(self, size):
        '''
        Rescale the image to at most size=(height, width). One dimension is
        guaranteed to be equally matched
        '''
        
        x_factor = self.shape[0] / size[0]
        y_factor = self.shape[1] / size[1]

        f = 1 / max(x_factor, y_factor)
        if x_factor > y_factor:
            new_size = (size[0], int(f * self.shape[1]))
        else:
            new_size = (int(f * self.shape[0]), size[1])

        K_scaler = torch.tensor([
            [f, 0, 0],
            [0, f, 0],
            [0, 0, 1]
        ], dtype=self.K.dtype, device=self.K.device)
        K = K_scaler @ self.K
 
        bitmap = _rescale(self.bitmap, new_size)
        if self.depth is not None:
            depth = _rescale(self.depth, new_size)
        else:
            depth = None

        return Image(K, self.R, self.T, bitmap, depth, self.bitmap_path)

    def pad(self, size):
        bitmap = _pad(self.bitmap, size, value=0)
        if self.depth is not None:
            depth  = _pad(self.depth, size, value=float('NaN'))
        else:
            depth = None

        return Image(self.K, self.R, self.T, bitmap, depth, self.bitmap_path)

    def to(self, *args, **kwargs):
        # use getattr/setattr to avoid repetitive code.
        # exclude `self.bitmap` because we don't need it on GPU (it's treated
        # separately by the dataloader)
        TRANSFERRED_ATTRS = ['K', 'R', 'T', 'depth']

        for key in TRANSFERRED_ATTRS:
            attr = getattr(self, key)
            if attr is not None:
                attr_transferred = attr.to(*args, **kwargs)
            setattr(self, key, attr_transferred)

        return self

    @dimchecked
    def unproject(self, xy: [2, 'N']) -> [3, 'N']:
        depth = self.fetch_depth(xy)

        xyw = torch.cat([
            xy.to(depth.dtype),
            torch.ones(1, xy.shape[1], dtype=depth.dtype, device=xy.device)
        ], dim=0)

        xyz = (self.K_inv @ xyw) * depth
        xyz_w = self.R.T @ (xyz - self.T[:, None])

        return xyz_w

    @dimchecked
    def project(self, xyw: [3, 'N']) -> [2, 'N']:
        extrinsic = self.R @ xyw + self.T[:, None]
        intrinsic = self.K @ extrinsic
        return intrinsic[:2] / intrinsic[2]

    @dimchecked
    def in_range_mask(self, xy: [2, 'N']) -> ['N']:
        h, w = self.shape
        x, y = xy

        return (0 <= x) & (x < w) & (0 <= y) & (y < h)

    @dimchecked
    def fetch_depth(self, xy: [2, 'N']) -> ['N']:
        if self.depth is None:
            raise ValueError(f'Depth is not loaded')

        in_range = self.in_range_mask(xy)
        finite = torch.isfinite(xy).all(dim=0)
        valid_depth = in_range & finite
        x, y = xy[:, valid_depth].to(torch.int64)
        depth = torch.full(
            (xy.shape[1], ),
            fill_value=float('NaN'),
            device=xy.device,
            dtype=self.depth.dtype
        )
        depth[valid_depth] = self.depth[0, y, x]

        return depth
