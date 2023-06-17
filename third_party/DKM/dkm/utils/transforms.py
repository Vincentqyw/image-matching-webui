from typing import Dict
import numpy as np
import torch
import kornia.augmentation as K
from kornia.geometry.transform import warp_perspective

# Adapted from Kornia
class GeometricSequential:
    def __init__(self, *transforms, align_corners=True) -> None:
        self.transforms = transforms
        self.align_corners = align_corners

    def __call__(self, x, mode="bilinear"):
        b, c, h, w = x.shape
        M = torch.eye(3, device=x.device)[None].expand(b, 3, 3)
        for t in self.transforms:
            if np.random.rand() < t.p:
                M = M.matmul(
                    t.compute_transformation(x, t.generate_parameters((b, c, h, w)))
                )
        return (
            warp_perspective(
                x, M, dsize=(h, w), mode=mode, align_corners=self.align_corners
            ),
            M,
        )

    def apply_transform(self, x, M, mode="bilinear"):
        b, c, h, w = x.shape
        return warp_perspective(
            x, M, dsize=(h, w), align_corners=self.align_corners, mode=mode
        )


class RandomPerspective(K.RandomPerspective):
    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        distortion_scale = torch.as_tensor(
            self.distortion_scale, device=self._device, dtype=self._dtype
        )
        return self.random_perspective_generator(
            batch_shape[0],
            batch_shape[-2],
            batch_shape[-1],
            distortion_scale,
            self.same_on_batch,
            self.device,
            self.dtype,
        )

    def random_perspective_generator(
        self,
        batch_size: int,
        height: int,
        width: int,
        distortion_scale: torch.Tensor,
        same_on_batch: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, torch.Tensor]:
        r"""Get parameters for ``perspective`` for a random perspective transform.

        Args:
            batch_size (int): the tensor batch size.
            height (int) : height of the image.
            width (int): width of the image.
            distortion_scale (torch.Tensor): it controls the degree of distortion and ranges from 0 to 1.
            same_on_batch (bool): apply the same transformation across the batch. Default: False.
            device (torch.device): the device on which the random numbers will be generated. Default: cpu.
            dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

        Returns:
            params Dict[str, torch.Tensor]: parameters to be passed for transformation.
                - start_points (torch.Tensor): element-wise perspective source areas with a shape of (B, 4, 2).
                - end_points (torch.Tensor): element-wise perspective target areas with a shape of (B, 4, 2).

        Note:
            The generated random numbers are not reproducible across different devices and dtypes.
        """
        if not (distortion_scale.dim() == 0 and 0 <= distortion_scale <= 1):
            raise AssertionError(
                f"'distortion_scale' must be a scalar within [0, 1]. Got {distortion_scale}."
            )
        if not (
            type(height) is int and height > 0 and type(width) is int and width > 0
        ):
            raise AssertionError(
                f"'height' and 'width' must be integers. Got {height}, {width}."
            )

        start_points: torch.Tensor = torch.tensor(
            [[[0.0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]],
            device=distortion_scale.device,
            dtype=distortion_scale.dtype,
        ).expand(batch_size, -1, -1)

        # generate random offset not larger than half of the image
        fx = distortion_scale * width / 2
        fy = distortion_scale * height / 2

        factor = torch.stack([fx, fy], dim=0).view(-1, 1, 2)
        offset = (torch.rand_like(start_points) - 0.5) * 2
        end_points = start_points + factor * offset

        return dict(start_points=start_points, end_points=end_points)
