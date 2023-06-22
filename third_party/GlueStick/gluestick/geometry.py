from typing import Tuple

import numpy as np
import torch


def to_homogeneous(points):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1] + (1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError


def from_homogeneous(points, eps=0.):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + eps)


def skew_symmetric(v):
    """Create a skew-symmetric matrix from a (batched) vector of size (..., 3).
    """
    z = torch.zeros_like(v[..., 0])
    M = torch.stack([
        z, -v[..., 2], v[..., 1],
        v[..., 2], z, -v[..., 0],
        -v[..., 1], v[..., 0], z,
    ], dim=-1).reshape(v.shape[:-1] + (3, 3))
    return M


def T_to_E(T):
    """Convert batched poses (..., 4, 4) to batched essential matrices."""
    return skew_symmetric(T[..., :3, 3]) @ T[..., :3, :3]


def warp_points_torch(points, H, inverse=True):
    """
    Warp a list of points with the INVERSE of the given homography.
    The inverse is used to be coherent with tf.contrib.image.transform
    Arguments:
        points: batched list of N points, shape (B, N, 2).
        homography: batched or not (shapes (B, 8) and (8,) respectively).
    Returns: a Tensor of shape (B, N, 2) containing the new coordinates of the warped points.
    """
    # H = np.expand_dims(homography, axis=0) if len(homography.shape) == 1 else homography

    # Get the points to the homogeneous format
    points = to_homogeneous(points)

    # Apply the homography
    out_shape = tuple(list(H.shape[:-1]) + [3, 3])
    H_mat = torch.cat([H, torch.ones_like(H[..., :1])], axis=-1).reshape(out_shape)
    if inverse:
        H_mat = torch.inverse(H_mat)
    warped_points = torch.einsum('...nj,...ji->...ni', points, H_mat.transpose(-2, -1))

    warped_points = from_homogeneous(warped_points, eps=1e-5)

    return warped_points


def seg_equation(segs):
    # calculate list of start, end and midpoints points from both lists
    start_points, end_points = to_homogeneous(segs[..., 0, :]), to_homogeneous(segs[..., 1, :])
    # Compute the line equations as ax + by + c = 0 , where x^2 + y^2 = 1
    lines = torch.cross(start_points, end_points, dim=-1)
    lines_norm = (torch.sqrt(lines[..., 0] ** 2 + lines[..., 1] ** 2)[..., None])
    assert torch.all(lines_norm > 0), 'Error: trying to compute the equation of a line with a single point'
    lines = lines / lines_norm
    return lines


def is_inside_img(pts: torch.Tensor, img_shape: Tuple[int, int]):
    h, w = img_shape
    return (pts >= 0).all(dim=-1) & (pts[..., 0] < w) & (pts[..., 1] < h) & (~torch.isinf(pts).any(dim=-1))


def shrink_segs_to_img(segs: torch.Tensor, img_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Shrink an array of segments to fit inside the image.
    :param segs: The tensor of segments with shape (N, 2, 2)
    :param img_shape: The image shape in format (H, W)
    """
    EPS = 1e-4
    device = segs.device
    w, h = img_shape[1], img_shape[0]
    # Project the segments to the reference image
    segs = segs.clone()
    eqs = seg_equation(segs)
    x0, y0 = torch.tensor([1., 0, 0.], device=device), torch.tensor([0., 1, 0], device=device)
    x0 = x0.repeat(eqs.shape[:-1] + (1,))
    y0 = y0.repeat(eqs.shape[:-1] + (1,))
    pt_x0s = torch.cross(eqs, x0, dim=-1)
    pt_x0s = pt_x0s[..., :-1] / pt_x0s[..., None, -1]
    pt_x0s_valid = is_inside_img(pt_x0s, img_shape)
    pt_y0s = torch.cross(eqs, y0, dim=-1)
    pt_y0s = pt_y0s[..., :-1] / pt_y0s[..., None, -1]
    pt_y0s_valid = is_inside_img(pt_y0s, img_shape)

    xW, yH = torch.tensor([1., 0, EPS - w], device=device), torch.tensor([0., 1, EPS - h], device=device)
    xW = xW.repeat(eqs.shape[:-1] + (1,))
    yH = yH.repeat(eqs.shape[:-1] + (1,))
    pt_xWs = torch.cross(eqs, xW, dim=-1)
    pt_xWs = pt_xWs[..., :-1] / pt_xWs[..., None, -1]
    pt_xWs_valid = is_inside_img(pt_xWs, img_shape)
    pt_yHs = torch.cross(eqs, yH, dim=-1)
    pt_yHs = pt_yHs[..., :-1] / pt_yHs[..., None, -1]
    pt_yHs_valid = is_inside_img(pt_yHs, img_shape)

    # If the X coordinate of the first endpoint is out
    mask = (segs[..., 0, 0] < 0) & pt_x0s_valid
    segs[mask, 0, :] = pt_x0s[mask]
    mask = (segs[..., 0, 0] > (w - 1)) & pt_xWs_valid
    segs[mask, 0, :] = pt_xWs[mask]
    # If the X coordinate of the second endpoint is out
    mask = (segs[..., 1, 0] < 0) & pt_x0s_valid
    segs[mask, 1, :] = pt_x0s[mask]
    mask = (segs[:, 1, 0] > (w - 1)) & pt_xWs_valid
    segs[mask, 1, :] = pt_xWs[mask]
    # If the Y coordinate of the first endpoint is out
    mask = (segs[..., 0, 1] < 0) & pt_y0s_valid
    segs[mask, 0, :] = pt_y0s[mask]
    mask = (segs[..., 0, 1] > (h - 1)) & pt_yHs_valid
    segs[mask, 0, :] = pt_yHs[mask]
    # If the Y coordinate of the second endpoint is out
    mask = (segs[..., 1, 1] < 0) & pt_y0s_valid
    segs[mask, 1, :] = pt_y0s[mask]
    mask = (segs[..., 1, 1] > (h - 1)) & pt_yHs_valid
    segs[mask, 1, :] = pt_yHs[mask]

    assert torch.all(segs >= 0) and torch.all(segs[..., 0] < w) and torch.all(segs[..., 1] < h)
    return segs


def warp_lines_torch(lines, H, inverse=True, dst_shape: Tuple[int, int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param lines: A tensor of shape (B, N, 2, 2) where B is the batch size, N the number of lines.
    :param H: The homography used to convert the lines. batched or not (shapes (B, 8) and (8,) respectively).
    :param inverse: Whether to apply H or the inverse of H
    :param dst_shape:If provided, lines are trimmed to be inside the image
    """
    device = lines.device
    batch_size, n = lines.shape[:2]
    lines = warp_points_torch(lines.reshape(batch_size, -1, 2), H, inverse).reshape(lines.shape)

    if dst_shape is None:
        return lines, torch.ones(lines.shape[:-2], dtype=torch.bool, device=device)

    out_img = torch.any((lines < 0) | (lines >= torch.tensor(dst_shape[::-1], device=device)), -1)
    valid = ~out_img.all(-1)
    any_out_of_img = out_img.any(-1)
    lines_to_trim = valid & any_out_of_img

    for b in range(batch_size):
        lines_to_trim_mask_b = lines_to_trim[b]
        lines_to_trim_b = lines[b][lines_to_trim_mask_b]
        corrected_lines = shrink_segs_to_img(lines_to_trim_b, dst_shape)
        lines[b][lines_to_trim_mask_b] = corrected_lines

    return lines, valid
