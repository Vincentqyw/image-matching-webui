"""
    A set of geometry tools to handle lines in Pytorch and Numpy.
"""

import numpy as np
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from scipy.sparse.csgraph import connected_components

from ..datasets.utils.homographies import warp_lines
from ..utils.tensor import (nn_interpolate_numpy, bilinear_interpolate_numpy,
                            compute_image_grad, preprocess_angle)


UPM_EPS = 1e-8

### Gradient computation

def compute_gradient_torch(img):
    """ Compute the x and y components of the gradient of a torch image. """
    if isinstance(img, torch.Tensor):
        torch_img = img.clone()
    else:
        torch_img = torch.tensor(img, dtype=torch.float)
    img_size = torch_img.shape
    device = torch_img.device
    if len(img_size) == 2:
        torch_img = torch_img[None, None]
    if len(img_size) == 3:
        torch_img = torch_img[:, None]
    x_kernel = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float,
                            device=device)[None, None]
    y_kernel = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float,
                            device=device)[None, None]
    grad_x = F.conv2d(torch_img, x_kernel, padding=1)[:, :, 1:, 1:]
    grad_y = F.conv2d(torch_img, y_kernel, padding=1)[:, :, 1:, 1:]
    return [grad_x / .2, grad_y / .2]
    
    
def compute_line_level_torch(img):
    """ Compute the orientation and magnitude of the line level
        (orthogonal to the gradient), after smoothing the image. """
    # Smooth the image
    kernel = cv2.getGaussianKernel(7, 0.6)
    kernel = torch.tensor(np.outer(kernel, kernel), dtype=torch.float,
                          device=img.device)[None, None]
    img = F.conv2d(F.pad(img, (3, 3, 3, 3), 'reflect'), kernel)
    
    # Compute the gradient and line-level orientation
    grad_x, grad_y = compute_gradient_torch(img)
    grad_norm = torch.sqrt(grad_x * grad_x + grad_y * grad_y)
    line_level_ang = torch.atan2(grad_x, -grad_y)
    return line_level_ang, grad_norm


### Line segment distances

def absolute_angle_distance(a0, a1):
    """ Compute the absolute distance between two angles, modulo pi.
        a0 and a1 are two angles in [-pi, +pi]. """
    pi_torch = torch.tensor(np.pi, device=a0.device)
    diff = torch.fmod(torch.abs(a0 - a1), pi_torch)
    return torch.min(diff, pi_torch - diff)


def get_structural_line_dist(warped_ref_line_seg, target_line_seg):
    """ Compute the distances between two sets of lines
        using the structural distance. """
    dist = (((warped_ref_line_seg[:, None, :, None]
              - target_line_seg[:, None]) ** 2).sum(-1)) ** 0.5
    dist = np.minimum(
        dist[:, :, 0, 0] + dist[:, :, 1, 1],
        dist[:, :, 0, 1] + dist[:, :, 1, 0]
    ) / 2
    return dist


def project_point_to_line(line_segs, points):
    """ Given a list of line segments and a list of points (2D or 3D coordinates),
        compute the orthogonal projection of all points on all lines.
        This returns the 1D coordinates of the projection on the line,
        as well as the list of orthogonal distances. """
    # Compute the 1D coordinate of the points projected on the line
    dir_vec = (line_segs[:, 1] - line_segs[:, 0])[:, None]
    coords1d = (((points[None] - line_segs[:, None, 0]) * dir_vec).sum(axis=2)
                / np.linalg.norm(dir_vec, axis=2) ** 2)
    # coords1d is of shape (n_lines, n_points)
    
    # Compute the orthogonal distance of the points to each line
    projection = line_segs[:, None, 0] + coords1d[:, :, None] * dir_vec
    dist_to_line = np.linalg.norm(projection - points[None], axis=2)

    return coords1d, dist_to_line


def get_segment_overlap(seg_coord1d):
    """ Given a list of segments parameterized by the 1D coordinate
        of the endpoints, compute the overlap with the segment [0, 1]. """
    seg_coord1d = np.sort(seg_coord1d, axis=-1)
    overlap = ((seg_coord1d[..., 1] > 0) * (seg_coord1d[..., 0] < 1)
               * (np.minimum(seg_coord1d[..., 1], 1)
                  - np.maximum(seg_coord1d[..., 0], 0)))
    return overlap


def get_orth_line_dist(line_seg1, line_seg2, min_overlap=0.5,
                       return_overlap=False, mode='min'):
    """ Compute the symmetrical orthogonal line distance between two sets
        of lines and the average overlapping ratio of both lines.
        Enforce a high line distance for small overlaps.
        This is compatible for nD objects (e.g. both lines in 2D or 3D). """
    n_lines1, n_lines2 = len(line_seg1), len(line_seg2)

    # Compute the average orthogonal line distance
    coords_2_on_1, line_dists2 = project_point_to_line(
        line_seg1, line_seg2.reshape(n_lines2 * 2, -1))
    line_dists2 = line_dists2.reshape(n_lines1, n_lines2, 2).sum(axis=2)
    coords_1_on_2, line_dists1 = project_point_to_line(
        line_seg2, line_seg1.reshape(n_lines1 * 2, -1))
    line_dists1 = line_dists1.reshape(n_lines2, n_lines1, 2).sum(axis=2)
    line_dists = (line_dists2 + line_dists1.T) / 2

    # Compute the average overlapping ratio
    coords_2_on_1 = coords_2_on_1.reshape(n_lines1, n_lines2, 2)
    overlaps1 = get_segment_overlap(coords_2_on_1)
    coords_1_on_2 = coords_1_on_2.reshape(n_lines2, n_lines1, 2)
    overlaps2 = get_segment_overlap(coords_1_on_2).T
    overlaps = (overlaps1 + overlaps2) / 2
    min_overlaps = np.minimum(overlaps1, overlaps2)
    
    if return_overlap:
        return line_dists, overlaps

    # Enforce a max line distance for line segments with small overlap
    if mode == 'mean':
        low_overlaps = overlaps < min_overlap
    else:
        low_overlaps = min_overlaps < min_overlap
    line_dists[low_overlaps] = np.amax(line_dists)
    return line_dists


def angular_distance(segs1, segs2):
    """ Compute the angular distance (via the cosine similarity)
        between two sets of line segments. """
    # Compute direction vector of segs1
    dirs1 = segs1[:, 1] - segs1[:, 0]
    dirs1 /= (np.linalg.norm(dirs1, axis=1, keepdims=True) + UPM_EPS)
    # Compute direction vector of segs2
    dirs2 = segs2[:, 1] - segs2[:, 0]
    dirs2 /= (np.linalg.norm(dirs2, axis=1, keepdims=True) + UPM_EPS)
    # https://en.wikipedia.org/wiki/Cosine_similarity
    return np.arccos(np.minimum(1, np.abs(np.einsum('ij,kj->ik', dirs1, dirs2))))


def overlap_distance_asym(line_seg1, line_seg2):
    """ Compute the overlap distance of line_seg2 projected to line_seg1. """
    n_lines1, n_lines2 = len(line_seg1), len(line_seg2)

    # Project endpoints 2 onto lines 1
    coords_2_on_1, _ = project_point_to_line(
        line_seg1, line_seg2.reshape(n_lines2 * 2, 2))
    coords_2_on_1 = coords_2_on_1.reshape(n_lines1, n_lines2, 2)

    # Compute the overlap
    overlaps = get_segment_overlap(coords_2_on_1)
    return overlaps


def overlap_distance_sym(line_seg1, line_seg2):
    """ Compute the symmetric overlap distance of line_seg2 and line_seg1. """
    overlap_2_on_1 = overlap_distance_asym(line_seg1, line_seg2)
    overlap_1_on_2 = overlap_distance_asym(line_seg2, line_seg1).T
    return (overlap_2_on_1 + overlap_1_on_2) / 2


def orientation(p, q, r):
    """ Compute the orientation of a list of triplets of points. """
    return np.sign((q[:, 1] - p[:, 1]) * (r[:, 0] - q[:, 0])
                   - (q[:, 0] - p[:, 0]) * (r[:, 1] - q[:, 1]))


def is_on_segment(line_seg, p):
    """ Check whether a point p is on a line segment, assuming the point
        to be colinear with the two endpoints. """
    return ((p[:, 0] >= np.min(line_seg[:, :, 0], axis=1))
            & (p[:, 0] <= np.max(line_seg[:, :, 0], axis=1))
            & (p[:, 1] >= np.min(line_seg[:, :, 1], axis=1))
            & (p[:, 1] <= np.max(line_seg[:, :, 1], axis=1)))


def intersect(line_seg1, line_seg2):
    """ Check whether two sets of lines segments
        intersects with each other. """
    ori1 = orientation(line_seg1[:, 0], line_seg1[:, 1], line_seg2[:, 0])
    ori2 = orientation(line_seg1[:, 0], line_seg1[:, 1], line_seg2[:, 1])
    ori3 = orientation(line_seg2[:, 0], line_seg2[:, 1], line_seg1[:, 0])
    ori4 = orientation(line_seg2[:, 0], line_seg2[:, 1], line_seg1[:, 1])
    return (((ori1 != ori2) & (ori3 != ori4))
            | ((ori1 == 0) & is_on_segment(line_seg1, line_seg2[:, 0]))
            | ((ori2 == 0) & is_on_segment(line_seg1, line_seg2[:, 1]))
            | ((ori3 == 0) & is_on_segment(line_seg2, line_seg1[:, 0]))
            | ((ori4 == 0) & is_on_segment(line_seg2, line_seg1[:, 1])))


def get_area_line_dist_asym(line_seg1, line_seg2, lbd=1/24):
    """ Compute an asymmetric line distance function which is not biased by
        the line length and is based on the area between segments.
        Here, line_seg2 are projected to the infinite line of line_seg1. """
    n1, n2 = len(line_seg1), len(line_seg2)

    # Determine which segments are intersecting each other
    all_line_seg1 = line_seg1[:, None].repeat(n2, axis=1).reshape(n1 * n2,
                                                                  2, 2)
    all_line_seg2 = line_seg2[None].repeat(n1, axis=0).reshape(n1 * n2, 2, 2)
    are_crossing = intersect(all_line_seg1, all_line_seg2)  # [n1 * n2]
    are_crossing = are_crossing.reshape(n1, n2)

    # Compute the orthogonal distance of the endpoints of line_seg2
    orth_dists2 = project_point_to_line(
        line_seg1, line_seg2.reshape(n2 * 2, 2))[1].reshape(n1, n2, 2)

    # Compute the angle between the line segments
    theta = angular_distance(line_seg1, line_seg2)  # [n1, n2]
    parallel = np.abs(theta) < 1e-8

    # Compute the orthogonal distance of the closest endpoint of line_seg2
    T = orth_dists2.min(axis=2)  # [n1, n2]

    # The distance for the intersecting lines is the area of two triangles,
    # divided by the length of line_seg2 squared:
    # area_dist = (d1^2+d2^2)/(2*tan(theta)*l^2)
    tan_theta = np.tan(theta)
    tan_theta[parallel] = 1
    length2 = np.linalg.norm(all_line_seg2[:, 0] - all_line_seg2[:, 1],
                             axis=1).reshape(n1, n2)
    area_dist = ((orth_dists2 ** 2).sum(axis=2)
                 / (2 * tan_theta * length2 ** 2) * (1. - parallel))

    # The distance for the non intersecting lines is lbd*T+1/4*sin(2*theta)
    non_int_area_dist = lbd * T + 1/4 * np.sin(2 * theta)
    area_dist[~are_crossing] = non_int_area_dist[~are_crossing]

    return area_dist


def get_area_line_dist(line_seg1, line_seg2, lbd=1/24):
    """ Compute a fairer line distance function which is not biased by
        the line length and is based on the area between segments. """
    area_dist_2_on_1 = get_area_line_dist_asym(line_seg1, line_seg2, lbd)
    area_dist_1_on_2 = get_area_line_dist_asym(line_seg2, line_seg1, lbd)
    area_dist = (area_dist_2_on_1 + area_dist_1_on_2.T) / 2
    return area_dist


def get_lip_line_dist_asym(line_seg1, line_seg2, default_len=30):
    """ Compute an asymmetrical length-invariant perpendicular distance. """
    n1, n2 = len(line_seg1), len(line_seg2)

    # Determine which segments are intersecting each other
    all_line_seg1 = line_seg1[:, None].repeat(n2, axis=1).reshape(n1 * n2,
                                                                  2, 2)
    all_line_seg2 = line_seg2[None].repeat(n1, axis=0).reshape(n1 * n2, 2, 2)
    are_crossing = intersect(all_line_seg1, all_line_seg2)  # [n1 * n2]
    are_crossing = are_crossing.reshape(n1, n2)

    # Compute the angle difference
    theta = angular_distance(line_seg1, line_seg2)  # [n1, n2]

    # Compute the orthogonal distance of the closest endpoint of line_seg2
    orth_dists2 = project_point_to_line(
        line_seg1, line_seg2.reshape(n2 * 2, 2))[1].reshape(n1, n2, 2)
    T = orth_dists2.min(axis=2)  # [n1, n2]
    
    # The distance is default_len * sin(theta) / 2 for intersecting lines
    # and T + default_len * sin(theta) / 2 for non intersecting ones
    # This means that a line crossing with theta=30deg is equivalent to a
    # parallel line with an offset T = default_len / 4
    lip_dist = default_len * np.sin(theta) / 2
    lip_dist[~are_crossing] += T[~are_crossing]
    return lip_dist


def get_lip_line_dist(line_seg1, line_seg2):
    """ Compute a length-invariant perpendicular distance. """
    lip_dist_2_on_1 = get_lip_line_dist_asym(line_seg1, line_seg2)
    lip_dist_1_on_2 = get_lip_line_dist_asym(line_seg2, line_seg1)
    lip_dist = (lip_dist_2_on_1 + lip_dist_1_on_2.T) / 2
    return lip_dist


### Line pre and post-processing

def clip_line_to_boundary(lines):
    """ Clip the first coordinate of a set of lines to the lower boundary 0
        and indicate which lines are completely outside of the boundary.
    Args:
        lines: a [N, 2, 2] tensor of lines.
    Returns:
        The clipped coordinates + a mask indicating invalid lines.
    """
    updated_lines = lines.copy()
    
    # Detect invalid lines completely outside of the first boundary
    invalid = np.all(lines[:, :, 0] < 0, axis=1)
    
    # Clip the lines to the boundary and update the second coordinate
    # First endpoint
    out = lines[:, 0, 0] < 0
    denom = lines[:, 1, 0] - lines[:, 0, 0]
    denom[denom == 0] = 1e-6
    ratio = lines[:, 1, 0] / denom
    updated_y = ratio * lines[:, 0, 1] + (1 - ratio) * lines[:, 1, 1]
    updated_lines[out, 0, 1] = updated_y[out]
    updated_lines[out, 0, 0] = 0
    # Second endpoint
    out = lines[:, 1, 0] < 0
    denom = lines[:, 0, 0] - lines[:, 1, 0]
    denom[denom == 0] = 1e-6
    ratio = lines[:, 0, 0] / denom
    updated_y = ratio * lines[:, 1, 1] + (1 - ratio) * lines[:, 0, 1]
    updated_lines[out, 1, 1] = updated_y[out]
    updated_lines[out, 1, 0] = 0
    
    return updated_lines, invalid


def clip_line_to_boundaries(lines, img_size, min_len=10):
    """ Clip a set of lines to the image boundaries and indicate
        which lines are completely outside of the boundaries.
    Args:
        lines: a [N, 2, 2] tensor of lines.
        img_size: the original image size.
    Returns:
        The clipped coordinates + a mask indicating valid lines.
    """
    new_lines = lines.copy()
    
    # Clip the first coordinate to the 0 boundary of img1
    new_lines, invalid_x0 = clip_line_to_boundary(lines)
    
    # Mirror in first coordinate to clip to the H-1 boundary
    new_lines[:, :, 0] = img_size[0] - 1 - new_lines[:, :, 0]
    new_lines, invalid_xh = clip_line_to_boundary(new_lines)
    new_lines[:, :, 0] = img_size[0] - 1 - new_lines[:, :, 0]
    
    # Swap the two coordinates, perform the same for y, and swap back
    new_lines = new_lines[:, :, [1, 0]]
    new_lines, invalid_y0 = clip_line_to_boundary(new_lines)
    new_lines[:, :, 0] = img_size[1] - 1 - new_lines[:, :, 0]
    new_lines, invalid_yw = clip_line_to_boundary(new_lines)
    new_lines[:, :, 0] = img_size[1] - 1 - new_lines[:, :, 0]
    new_lines = new_lines[:, :, [1, 0]]
    
    # Merge all the invalid lines and also remove lines that became too short
    short = np.linalg.norm(new_lines[:, 1] - new_lines[:, 0],
                           axis=1) < min_len
    valid = np.logical_not(invalid_x0 | invalid_xh
                           | invalid_y0 | invalid_yw | short)
    
    return new_lines, valid


def get_common_lines(lines0, lines1, H, img_size):
    """ Extract the lines in common between two views, by warping lines1
        into lines0 frame.
    Args:
        lines0, lines1: sets of lines of size [N, 2, 2].
        H: homography relating the lines.
        img_size: size of the original images img0 and img1.
    Returns:
        Updated lines0 with a valid reprojection in img1 and warped_lines1.
    """
    # First warp lines0 to img1 to detect invalid lines
    warped_lines0 = warp_lines(lines0, H)
    
    # Clip them to the boundary
    warped_lines0, valid = clip_line_to_boundaries(warped_lines0, img_size)
    
    # Warp all the valid lines back in img0
    inv_H = np.linalg.inv(H)
    new_lines0 = warp_lines(warped_lines0[valid], inv_H)
    warped_lines1 = warp_lines(lines1, inv_H)
    warped_lines1, valid = clip_line_to_boundaries(warped_lines1, img_size)

    return new_lines0, warped_lines1[valid]


def merge_line_cluster(lines):
    """ Merge a cluster of line segments.
    First compute the principal direction of the lines, compute the
    endpoints barycenter, project the endpoints onto the middle line,
    keep the two extreme projections.
    Args:
        lines: a (n, 2, 2) np array containing n lines.
    Returns:
        The merged (2, 2) np array line segment.
    """
    # Get the principal direction of the endpoints
    points = lines.reshape(-1, 2)
    weights = np.linalg.norm(lines[:, 0] - lines[:, 1], axis=1)
    weights = np.repeat(weights, 2)[:, None]
    weights /= weights.sum()  # More weights for longer lines
    avg_points = (points * weights).sum(axis=0)
    points_bar = points - avg_points[None]
    cov = 1 / 3 * np.einsum(
        'ij,ik->ijk', points_bar, points_bar * weights).sum(axis=0)
    a, b, c = cov[0, 0], cov[0, 1], cov[1, 1]
    # Principal component of a 2x2 symmetric matrix
    if b == 0:
        u = np.array([1, 0]) if a >= c else np.array([0, 1])
    else:
        m = (c - a + np.sqrt((a - c) ** 2 + 4 * b ** 2)) / (2 * b)
        u = np.array([1, m]) / np.sqrt(1 + m ** 2)
        
    # Get the center of gravity of all endpoints
    cross = np.mean(points, axis=0)
        
    # Project the endpoints on the line defined by cross and u
    avg_line_seg = np.stack([cross, cross + u], axis=0)
    proj = project_point_to_line(avg_line_seg[None], points)[0]
    
    # Take the two extremal projected endpoints
    new_line = np.stack([cross + np.amin(proj) * u,
                         cross + np.amax(proj) * u], axis=0)
    return new_line


def merge_lines(lines, thresh=5., overlap_thresh=0.):
    """ Given a set of lines, merge close-by lines together.
    Two lines are merged when their orthogonal distance is smaller
    than a threshold and they have a positive overlap.
    Args:
        lines: a (N, 2, 2) np array.
        thresh: maximum orthogonal distance between two lines to be merged.
        overlap_thresh: maximum distance between 2 endpoints to merge
                        two aligned lines.
    Returns:
        The new lines after merging.
    """
    if len(lines) == 0:
        return lines

    # Compute the pairwise orthogonal distances and overlap
    orth_dist, overlaps = get_orth_line_dist(lines, lines, return_overlap=True)
    
    # Define clusters of close-by lines to merge
    if overlap_thresh == 0:
        adjacency_mat = (overlaps > 0) * (orth_dist < thresh)
    else:
        # Filter using the distance between the two closest endpoints
        n = len(lines)
        endpoints = lines.reshape(n * 2, 2)
        close_endpoint = np.linalg.norm(endpoints[:, None] - endpoints[None],
                                        axis=2)
        close_endpoint = close_endpoint.reshape(n, 2, n, 2).transpose(
            0, 2, 1, 3).reshape(n, n, 4)
        close_endpoint = np.amin(close_endpoint, axis=2)
        adjacency_mat = (((overlaps > 0) | (close_endpoint < overlap_thresh))
                         * (orth_dist < thresh))
    n_comp, components = connected_components(adjacency_mat, directed=False)
    
    # For each cluster, merge all lines into a single one
    new_lines = []
    for i in range(n_comp):
        cluster = lines[components == i]
        new_lines.append(merge_line_cluster(cluster))
        
    return np.stack(new_lines, axis=0)


def filter_lines(lines, df, line_level, df_thresh=0.004, ang_thresh=0.383):
    """ Filter out lines that have an average DF value too high, or
    whose direction does not agree with the line level map.
    Args:
        lines: a (N, 2, 2) torch tensor.
        df, line_level: (rows, cols) tensors with the DF and line-level values.
        df_thresh, ang_thresh: threshold for the DF and line_level.
    Returns:
        A tuple with the filtered lines and a mask of valid lines.
    """
    n_samples = 10
    rows, cols = df.shape
    
    # DF check
    alpha = torch.linspace(0, 1, n_samples, device=df.device)[None]
    x_coord = 2 * (lines[:, :1, 1] * alpha
                   + lines[:, 1:, 1] * (1 - alpha)) / cols - 1
    y_coord = 2 * (lines[:, :1, 0] * alpha
                   + lines[:, 1:, 0] * (1 - alpha)) / rows - 1
    grid = torch.stack([x_coord, y_coord], dim=-1)[None]
    # grid is of size [1, num_lines, n_samples, 2]
    df_samples = F.grid_sample(df[None, None], grid, mode='bilinear',
                               padding_mode='border')[0, 0]
    df_check = df_samples.mean(dim=1) < df_thresh

    # Line orientation check
    line_len = torch.sqrt(((lines[:, 0] - lines[:, 1]) ** 2).sum(dim=1))
    vec_x = ((lines[:, 1, 1] - lines[:, 0, 1]) / line_len).unsqueeze(-1)
    vec_y = ((lines[:, 1, 0] - lines[:, 0, 0]) / line_len).unsqueeze(-1)
    cos_line_level = torch.cos(line_level)[None, None]
    sin_line_level = torch.sin(line_level)[None, None]
    cos_samples = F.grid_sample(cos_line_level, grid, mode='bilinear',
                                padding_mode='border')[0, 0]
    sin_samples = F.grid_sample(sin_line_level, grid, mode='bilinear',
                                padding_mode='border')[0, 0]
    ang_check = torch.abs(
        cos_samples * vec_y - sin_samples * vec_x).mean(dim=1) < ang_thresh
    
    # Gather the results
    valid = df_check & ang_check
    return lines[valid], valid


def filter_outlier_lines(
    img, lines, df, angle, mode='inlier_thresh', use_grad=False,
    inlier_thresh=0.5, df_thresh=1.5, ang_thresh=np.pi / 6, n_samples=50):
    """ Filter out outlier lines either by comparing the average DF and
        line level values to a threshold or by counting the number of inliers
        across the line. It can also optionally use the image gradient.
    Args:
        img: the original image.
        lines: a (N, 2, 2) np array.
        df: np array with the distance field.
        angle: np array with the grad angle field.
        mode: 'mean' or 'inlier_thresh'.
        use_grad: True to use the image gradient instead of line_level.
        inlier_thresh: ratio of inliers to get accepted.
        df_thresh, ang_thresh: thresholds to determine a valid value.
        n_samples: number of points sampled along each line.
    Returns:
        A tuple with the filtered lines and a mask of valid lines.
    """
    # Get the right orientation of the line_level and the lines orientation
    oriented_line_level, img_grad_angle = preprocess_angle(angle, img)
    orientations = get_line_orientation(lines, oriented_line_level)

    # Get the sample positions
    t = np.linspace(0, 1, n_samples)[None, :, None]
    samples = lines[:, 0][:, None] + t * (lines[:, 1][:, None]
                                          - lines[:, 0][:, None])
    samples = samples.reshape(-1, 2)

    # Interpolate the DF and angle map
    df_samples = bilinear_interpolate_numpy(df, samples[:, 1], samples[:, 0])
    df_samples = df_samples.reshape(-1, n_samples)
    if use_grad:
        oriented_line_level = np.mod(img_grad_angle - np.pi / 2, 2 * np.pi)
    ang_samples = nn_interpolate_numpy(oriented_line_level, samples[:, 1],
                                       samples[:, 0]).reshape(-1, n_samples)

    # Check the average value or number of inliers
    if mode == 'mean':
        df_check = np.mean(df_samples, axis=1) < df_thresh
        ang_avg = np.arctan2(np.sin(ang_samples).sum(axis=1),
                             np.cos(ang_samples).sum(axis=1))
        ang_diff = np.minimum(np.abs(ang_avg - orientations),
                              2 * np.pi - np.abs(ang_avg - orientations))
        ang_check = ang_diff < ang_thresh
        valid = df_check & ang_check
    elif mode == 'inlier_thresh':
        df_check = df_samples < df_thresh
        ang_diff = np.minimum(
            np.abs(ang_samples - orientations[:, None]),
            2 * np.pi - np.abs(ang_samples - orientations[:, None]))
        ang_check = ang_diff < ang_thresh
        valid = (df_check & ang_check).mean(axis=1) > inlier_thresh
    else:
        raise ValueError("Unknown filtering mode: " + mode)

    return lines[valid], valid


### DF-related methods

def seg_to_df(lines, img_size):
    """ Convert a list of lines into a distance field map in pixels.
    Compute also the point on the closest line.
    Args:
        lines: set of lines of size [N, 2, 2].
        img_size: the original image size (H, W).
    Returns:
        A tuple containing the following elements:
        - a 2D map of size (H, W) with the distance of each pixel
        to the closest line.
        - a [H, W, 2] array containing the closest point for each pixel.
        - a bool indicating if the closest point is an endpoint or not.
    """
    h, w = img_size
    diag_len = np.sqrt(h ** 2 + w ** 2)
    
    # Stop if no lines are present
    if len(lines) == 0:
        return (np.ones(img_size), np.zeros((h, w, 2)),
                np.ones(img_size, dtype=bool))

    # Get the pixel positions
    pix_loc = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    pix_loc = np.stack(pix_loc, axis=-1).reshape(-1, 2)
    
    # Compute their distance and angle to every line segment
    # Orthogonal line distance
    proj, dist = project_point_to_line(lines, pix_loc)
    out_seg = (proj < 0) | (proj > 1)
    dist[out_seg] = diag_len
    closest_on_line = np.argmin(dist, axis=0)
    closest_on_line = (lines[closest_on_line][:, 0]
                       + proj[closest_on_line, np.arange(len(pix_loc))][:, None]
                       * (lines[closest_on_line][:, 1]
                          - lines[closest_on_line][:, 0]))
    dist = np.amin(dist, axis=0)
    # Distance to the closest endpoint
    dist_e1 = np.linalg.norm(pix_loc[:, None] - lines[None, :, 0], axis=2)
    dist_e2 = np.linalg.norm(pix_loc[:, None] - lines[None, :, 1], axis=2)
    struct_dist = np.minimum(dist_e1, dist_e2)
    closest_on_endpoint = np.argmin(struct_dist, axis=1)
    endpoint_candidates = lines[closest_on_endpoint]
    closest_endpoint = np.argmin(
        np.stack([dist_e1[np.arange(len(pix_loc)), closest_on_endpoint],
                  dist_e2[np.arange(len(pix_loc)), closest_on_endpoint]],
                 axis=-1), axis=1)
    closest_on_endpoint = endpoint_candidates[np.arange(len(pix_loc)),
                                              closest_endpoint]
    struct_dist = np.amin(struct_dist, axis=1)
    # Take the minimum of the orthogonal line distance and endpoint distance
    is_closest_endpoint = struct_dist < dist
    dist = np.minimum(dist, struct_dist)
    closest_point = np.where(is_closest_endpoint[:, None], closest_on_endpoint,
                             closest_on_line)  
    
    # Reshape
    dist = dist.reshape(h, w)
    is_closest_endpoint = is_closest_endpoint.reshape(h, w)
    closest_point = closest_point.reshape(h, w, 2)
    return dist, closest_point, is_closest_endpoint


### General util functions

def sample_along_line(lines, img, n_samples=10, mode='mean'):
    """ Sample a fixed number of points along each line and interpolate
        an img at these points, and finally aggregate the values. """
    # Get the sample positions
    t = np.linspace(0, 1, 10)[None, :, None]
    samples = lines[:, 0][:, None] + t * (lines[:, 1][:, None]
                                          - lines[:, 0][:, None])
    samples = samples.reshape(-1, 2)

    # Interpolate the img at the samples and aggregate the values
    if mode == 'mean':
        # Average
        val = bilinear_interpolate_numpy(img, samples[:, 1], samples[:, 0])
        val = np.mean(val.reshape(-1, n_samples), axis=-1)
    elif mode == 'angle':
        # Average of angles
        val = nn_interpolate_numpy(img, samples[:, 1], samples[:, 0])
        val = val.reshape(-1, n_samples)
        val = np.arctan2(np.sin(val).sum(axis=-1), np.cos(val).sum(axis=-1))
    elif mode == 'median':
        # Median
        val = nn_interpolate_numpy(img, samples[:, 1], samples[:, 0])
        val = np.median(val.reshape(-1, n_samples), axis=-1)
    else:
        # No aggregation
        val = nn_interpolate_numpy(img, samples[:, 1], samples[:, 0])
        val = val.reshape(-1, n_samples)

    return val


def get_line_orientation(lines, angle):
    """ Get the orientation in [-pi, pi] of a line, based on the gradient. """
    grad_val = sample_along_line(lines, angle, mode='angle')
    line_ori = np.mod(np.arctan2(lines[:, 1, 0] - lines[:, 0, 0],
                                 lines[:, 1, 1] - lines[:, 0, 1]), np.pi)

    pos_dist = np.minimum(np.abs(grad_val - line_ori),
                          2 * np.pi - np.abs(grad_val - line_ori))
    neg_dist = np.minimum(np.abs(grad_val - line_ori + np.pi),
                          2 * np.pi - np.abs(grad_val - line_ori + np.pi))
    line_ori = np.where(pos_dist <= neg_dist, line_ori, line_ori - np.pi)
    return line_ori
