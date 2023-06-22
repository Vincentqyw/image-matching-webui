"""
    Utility functions to project data across different views.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pycolmap import image_to_world, world_to_image


def to_homogeneous(arr):
    # Adds a new column with ones
    if isinstance(arr, torch.Tensor):
        return torch.cat([arr, torch.ones_like(arr[..., :1])], dim=-1)
    else:
        return np.concatenate([arr, np.ones_like(arr[..., :1])], axis=-1)


def to_homogeneous_t(arr):
    # Adds a new row with ones
    if isinstance(arr, torch.Tensor):
        return torch.cat([arr, torch.ones_like(arr[..., :1, :])], dim=-2)
    else:
        return np.concatenate([arr, np.ones_like(arr[..., :1, :])], axis=-2)


def to_cartesian(arr):
    return arr[..., :-1] / arr[..., -1:]


def to_cartesian_t(arr):
    return arr[..., :-1, :] / arr[..., -1:, :]


def warp_points(points, H, img_shape):
    """ Warp 2D points by an homography H.
    Args:
        points: a [b_size, N, 2] or [N, 2] torch tensor (ij coords).
        H: a [N, 3, 3] torch homography tensor.
    Returns:
        The reprojected points and a mask of valid points.
    """
    reproj_points = points.clone()[..., [1, 0]]
    reproj_points = to_homogeneous(reproj_points)
    reproj_points = (H @ reproj_points.transpose(-1, -2)).transpose(-1, -2)
    reproj_points = reproj_points[..., :2] / reproj_points[..., 2:]
    reproj_points = reproj_points[..., [1, 0]]
    
    # Compute the valid points
    h, w = img_shape
    valid = ((reproj_points[..., 0] >= 0)
             & (reproj_points[..., 0] <= h - 1)
             & (reproj_points[..., 1] >= 0)
             & (reproj_points[..., 1] <= w - 1))
    
    return reproj_points, valid


### 3D geometry utils for ETH3D

# Convert from quaternions to rotation matrix
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


# Convert a rotation matrix to quaternions
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


# Read the camera intrinsics from a file in COLMAP format
def read_cameras(camera_file, scale_factor=None):
    with open(camera_file, 'r') as f:
        raw_cameras = f.read().rstrip().split('\n')
    raw_cameras = raw_cameras[3:]
    cameras = []
    for c in raw_cameras:
        data = c.split(' ')
        cameras.append({
            "model": data[1],
            "width": int(data[2]),
            "height": int(data[3]),
            "params": np.array(list(map(float, data[4:])))})

    # Optionally scale the intrinsics if the image are resized
    if scale_factor is not None:
        cameras = [scale_intrinsics(c, scale_factor) for c in cameras]
    return cameras


# Adapt the camera intrinsics to an image resize
def scale_intrinsics(intrinsics, scale_factor):
    new_intrinsics = {"model": intrinsics["model"],
                      "width": int(intrinsics["width"] * scale_factor + 0.5),
                      "height": int(intrinsics["height"] * scale_factor + 0.5)
                      }
    params = intrinsics["params"]
    # Adapt the focal length
    params[:2] *= scale_factor
    # Adapt the principal point
    params[2:4] = (params[2:4] * scale_factor + 0.5) - 0.5
    new_intrinsics["params"] = params
    return new_intrinsics


# Project points from 2D to 3D, in (x, y, z) format
def project_2d_to_3d(points, depth, T_local_to_world, intrinsics):
    # Warp to world homogeneous coordinates
    world_points = image_to_world(points[:, [1, 0]],
                                  intrinsics)['world_points']
    world_points *= depth[:, None]
    world_points = np.concatenate([world_points, depth[:, None],
                                   np.ones((len(depth), 1))], axis=1)

    # Warp to the world coordinates
    world_points = (T_local_to_world @ world_points.T).T
    world_points = world_points[:, :3] / world_points[:, 3:]
    return world_points


# Project points from 3D in (x, y, z) format to 2D
def project_3d_to_2d(points, T_world_to_local, intrinsics):
    norm_points = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    norm_points = (T_world_to_local @ norm_points.T).T
    norm_points = norm_points[:, :3] / norm_points[:, 3:]
    norm_points = norm_points[:, :2] / norm_points[:, 2:]
    image_points = world_to_image(norm_points, intrinsics)
    image_points = np.stack(image_points['image_points'])[:, [1, 0]]
    return image_points


# Mask out the points that are outside of img_size
def mask_points(points, img_size):
    mask = ((points[..., 0] >= 0)
            & (points[..., 0] < img_size[0])
            & (points[..., 1] >= 0)
            & (points[..., 1] < img_size[1]))
    return mask


def get_depth(img_points, dist_depth, dist_camera, undist_camera):
    """
        Get the depth of a list of image points in the undistorted image,
        given the depth of the distorted image.
    """
    # Warp the points to world coordinates
    world_points = image_to_world(img_points[:, [1, 0]],
                                  undist_camera)['world_points']

    # Warp them back to the distorted coordinates
    dist_img_points = world_to_image(world_points, dist_camera)
    dist_img_points = np.stack(dist_img_points['image_points'])
    dist_img_points = np.round(dist_img_points).astype(int)

    # Get the depth of valid points (inf otherwise)
    dist_shape = (int(dist_camera['height']), int(dist_camera['width']))
    valid = ((dist_img_points[:, 0] >= 0)
             & (dist_img_points[:, 0] < dist_shape[1])
             & (dist_img_points[:, 1] >= 0)
             & (dist_img_points[:, 1] < dist_shape[0]))
    depths = np.array([np.inf] * len(dist_img_points))
    valid_dist_img_points = dist_img_points[valid]
    depths[valid] = dist_depth[valid_dist_img_points[:, 1],
                                valid_dist_img_points[:, 0]]
    return depths


def filter_and_project_lines(
    ref_line_seg, target_line_seg, ref_depth, target_depth, data):
    """ Filter out lines without depth, project them to 3D, warp them in
        the other view, and keep lines shared between both views. """
    # Get the points with valid depth
    ref_depths = get_depth(
        ref_line_seg.reshape(-1, 2), ref_depth,
        data["ref_dist_camera"], data["ref_undist_camera"]).reshape(-1, 2)
    ref_valid = ~np.any(np.isinf(ref_depths), axis=1)
    ref_valid_line_seg = ref_line_seg[ref_valid]
    target_depths = get_depth(
        target_line_seg.reshape(-1, 2), target_depth,
        data["target_dist_camera"],
        data["target_undist_camera"]).reshape(-1, 2)
    target_valid = ~np.any(np.isinf(target_depths), axis=1)
    target_valid_line_seg = target_line_seg[target_valid]

    # Useful image shapes
    ref_dist_shape = (int(data["ref_dist_camera"]['height']),
                        int(data["ref_dist_camera"]['width']))
    target_dist_shape = (int(data["target_dist_camera"]['height']),
                            int(data["target_dist_camera"]['width']))

    # Project the lines in 3D and then in the other view
    # Keep only the lines in common between the two views
    # Ref
    if len(ref_valid_line_seg) > 0:
        ref_3d_lines = project_2d_to_3d(
            ref_valid_line_seg.reshape(-1, 2),
            ref_depths[ref_valid].flatten(),
            np.linalg.inv(data["T_world_to_ref"]),
            data["ref_undist_camera"])
        warped_ref_valid_line_seg = project_3d_to_2d(
            ref_3d_lines, data["T_world_to_target"],
            data["target_undist_camera"])
    else:
        ref_3d_lines = np.empty((0, 3))
        warped_ref_valid_line_seg = np.empty((0, 2))
    valid_mask = mask_points(warped_ref_valid_line_seg, target_dist_shape)
    valid_mask = np.all(valid_mask.reshape(-1, 2), axis=1)
    ref_valid[ref_valid] = valid_mask
    ref_valid_line_seg = ref_valid_line_seg[valid_mask]
    ref_3d_lines = ref_3d_lines.reshape(-1, 2, 3)[valid_mask]
    warped_ref_valid_line_seg = warped_ref_valid_line_seg.reshape(
        -1, 2, 2)[valid_mask]
    # Target
    if len(target_valid_line_seg) > 0:
        target_3d_lines = project_2d_to_3d(
            target_valid_line_seg.reshape(-1, 2),
            target_depths[target_valid].flatten(),
            np.linalg.inv(data["T_world_to_target"]),
            data["target_undist_camera"])
        warped_target_valid_line_seg = project_3d_to_2d(
            target_3d_lines, data["T_world_to_ref"],
            data["ref_undist_camera"])
    else:
        target_3d_lines = np.empty((0, 3))
        warped_target_valid_line_seg = np.empty((0, 2))
    valid_mask = mask_points(warped_target_valid_line_seg, ref_dist_shape)
    valid_mask = np.all(valid_mask.reshape(-1, 2), axis=1)
    target_valid[target_valid] = valid_mask
    target_valid_line_seg = target_valid_line_seg[valid_mask]
    target_3d_lines = target_3d_lines.reshape(-1, 2, 3)[valid_mask]
    warped_target_valid_line_seg = warped_target_valid_line_seg.reshape(
        -1, 2, 2)[valid_mask]

    return (ref_valid_line_seg, target_valid_line_seg,
            ref_3d_lines, target_3d_lines,
            warped_ref_valid_line_seg, warped_target_valid_line_seg,
            ref_valid, target_valid)
