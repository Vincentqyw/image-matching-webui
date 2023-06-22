"""
A set of geometry tools for PyTorch tensors and sometimes NumPy arrays.
"""

import torch


def keypoints_to_grid(keypoints, img_size):
    """ Convert the cartesian coordinates of 2D keypoints into a grid in
        [-1, 1]² that can be used in torch.nn.functional.interpolate.
    Args:
        keypoints: a (..., N, 2) tensor of N keypoints.
        img_size: image size.
    Returns:
        A (B, N, 1, 2) tensor of normalized coordinates.
    """
    n_points = keypoints.size()[-2]
    device = keypoints.device
    grid_points = keypoints.float() * 2. / torch.tensor(
        img_size, dtype=torch.float, device=device) - 1.
    grid_points = grid_points[..., [1, 0]].view(-1, n_points, 1, 2)
    return grid_points


def get_dist_mask(kp0, kp1, valid_mask, dist_thresh):
    """ Compute a 2D matrix indicating the local neighborhood of each point
        for a given threshold and two lists of corresponding keypoints.
    Args:
        kp0, kp1: a (B, N, 2) tensor of 2D points.
        valid_mask: a (B*N) boolean mask indicating valid points.
        dist_thresh: distance in pixels defining the local neighborhood.
    Returns:
        A (B*N, B*N) bool tensor indicating points that are spatially close. 
    """
    b_size, n_points, _ = kp0.size()
    dist_mask0 = torch.norm(kp0.unsqueeze(2) - kp0.unsqueeze(1), dim=-1)
    dist_mask1 = torch.norm(kp1.unsqueeze(2) - kp1.unsqueeze(1), dim=-1)
    dist_mask = torch.min(dist_mask0, dist_mask1)
    dist_mask = dist_mask <= dist_thresh
    dist_mask = dist_mask.repeat(1, 1, b_size).reshape(b_size * n_points,
                                                       b_size * n_points)
    dist_mask = dist_mask[valid_mask, :][:, valid_mask]
    return dist_mask
