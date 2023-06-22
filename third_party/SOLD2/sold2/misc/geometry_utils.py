import numpy as np
import torch


### Point-related utils

# Warp a list of points using a homography
def warp_points(points, homography):
    # Convert to homogeneous and in xy format
    new_points = np.concatenate([points[..., [1, 0]],
                                 np.ones_like(points[..., :1])], axis=-1)
    # Warp
    new_points = (homography @ new_points.T).T
    # Convert back to inhomogeneous and hw format
    new_points = new_points[..., [1, 0]] / new_points[..., 2:]
    return new_points


# Mask out the points that are outside of img_size
def mask_points(points, img_size):
    mask = ((points[..., 0] >= 0)
            & (points[..., 0] < img_size[0])
            & (points[..., 1] >= 0)
            & (points[..., 1] < img_size[1]))
    return mask


# Convert a tensor [N, 2] or batched tensor [B, N, 2] of N keypoints into
# a grid in [-1, 1]² that can be used in torch.nn.functional.interpolate
def keypoints_to_grid(keypoints, img_size):
    n_points = keypoints.size()[-2]
    device = keypoints.device
    grid_points = keypoints.float() * 2. / torch.tensor(
        img_size, dtype=torch.float, device=device) - 1.
    grid_points = grid_points[..., [1, 0]].view(-1, n_points, 1, 2)
    return grid_points


# Return a 2D matrix indicating the local neighborhood of each point
# for a given threshold and two lists of corresponding keypoints
def get_dist_mask(kp0, kp1, valid_mask, dist_thresh):
    b_size, n_points, _ = kp0.size()
    dist_mask0 = torch.norm(kp0.unsqueeze(2) - kp0.unsqueeze(1), dim=-1)
    dist_mask1 = torch.norm(kp1.unsqueeze(2) - kp1.unsqueeze(1), dim=-1)
    dist_mask = torch.min(dist_mask0, dist_mask1)
    dist_mask = dist_mask <= dist_thresh
    dist_mask = dist_mask.repeat(1, 1, b_size).reshape(b_size * n_points,
                                                       b_size * n_points)
    dist_mask = dist_mask[valid_mask, :][:, valid_mask]
    return dist_mask


### Line-related utils

# Sample n points along lines of shape (num_lines, 2, 2)
def sample_line_points(lines, n):
    line_points_x = np.linspace(lines[:, 0, 0], lines[:, 1, 0], n, axis=-1)
    line_points_y = np.linspace(lines[:, 0, 1], lines[:, 1, 1], n, axis=-1)
    line_points = np.stack([line_points_x, line_points_y], axis=2)
    return line_points


# Return a mask of the valid lines that are within a valid mask of an image
def mask_lines(lines, valid_mask):
    h, w = valid_mask.shape
    int_lines = np.clip(np.round(lines).astype(int), 0, [h - 1, w - 1])
    h_valid = valid_mask[int_lines[:, 0, 0], int_lines[:, 0, 1]]
    w_valid = valid_mask[int_lines[:, 1, 0], int_lines[:, 1, 1]]
    valid = h_valid & w_valid
    return valid


# Return a 2D matrix indicating for each pair of points
# if they are on the same line or not
def get_common_line_mask(line_indices, valid_mask):
    b_size, n_points = line_indices.shape
    common_mask = line_indices[:, :, None] == line_indices[:, None, :]
    common_mask = common_mask.repeat(1, 1, b_size).reshape(b_size * n_points,
                                                           b_size * n_points)
    common_mask = common_mask[valid_mask, :][:, valid_mask]
    return common_mask
