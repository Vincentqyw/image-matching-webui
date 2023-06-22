"""
This file implements the homographic transforms for data augmentation.
Code adapted from https://github.com/rpautrat/SuperPoint
"""
import numpy as np
from math import pi

from ..synthetic_util import get_line_map, get_line_heatmap
import cv2
import copy
import shapely.geometry


def sample_homography(
        shape, perspective=True, scaling=True, rotation=True,
        translation=True, n_scales=5, n_angles=25, scaling_amplitude=0.1,
        perspective_amplitude_x=0.1, perspective_amplitude_y=0.1,
        patch_ratio=0.5, max_angle=pi/2, allow_artifacts=False,
        translation_overflow=0.):
    """
    Computes the homography transformation between a random patch in the
    original image and a warped projection with the same image size.
    As in `tf.contrib.image.transform`, it maps the output point
    (warped patch) to a transformed input point (original patch).
    The original patch, initialized with a simple half-size centered crop,
    is iteratively projected, scaled, rotated and translated.

    Arguments:
        shape: A rank-2 `Tensor` specifying the height and width of the original image.
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography.
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.

    Returns:
        homo_mat: A numpy array of shape `[1, 3, 3]` corresponding to the
                  homography transform.
        selected_scale: The selected scaling factor.
    """
    # Convert shape to ndarry
    if not isinstance(shape, np.ndarray):
        shape = np.array(shape)

    # Corners of the output image
    pts1 = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    # Corners of the input patch
    margin = (1 - patch_ratio) / 2
    pts2 = margin + np.array([[0, 0], [0, patch_ratio],
                             [patch_ratio, patch_ratio], [patch_ratio, 0]])

    # Random perspective and affine perturbations
    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)

        # normal distribution with mean=0, std=perspective_amplitude_y/2
        perspective_displacement = np.random.normal(
            0., perspective_amplitude_y/2, [1])
        h_displacement_left = np.random.normal(
            0., perspective_amplitude_x/2, [1])
        h_displacement_right = np.random.normal(
            0., perspective_amplitude_x/2, [1])
        pts2 += np.stack([np.concatenate([h_displacement_left,
                                          perspective_displacement], 0),
                          np.concatenate([h_displacement_left,
                                          -perspective_displacement], 0),
                          np.concatenate([h_displacement_right,
                                          perspective_displacement], 0),
                          np.concatenate([h_displacement_right,
                                          -perspective_displacement], 0)])

    # Random scaling: sample several scales, check collision with borders,
    # randomly pick a valid one
    if scaling:
        scales = np.concatenate(
            [[1.], np.random.normal(1, scaling_amplitude/2, [n_scales])], 0)
        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[None, ...] * scales[..., None, None] + center
        # all scales are valid except scale=1
        if allow_artifacts:
            valid = np.array(range(n_scales))
        # Chech the valid scale
        else:
            valid = np.where(np.all((scaled >= 0.)
                             & (scaled < 1.), (1, 2)))[0]
        # No valid scale found => recursively call
        if valid.shape[0] == 0:
            return sample_homography(
                shape, perspective, scaling, rotation, translation,
                n_scales, n_angles, scaling_amplitude, 
                perspective_amplitude_x, perspective_amplitude_y,
                patch_ratio, max_angle, allow_artifacts, translation_overflow)

        idx = valid[np.random.uniform(0., valid.shape[0], ()).astype(np.int32)]
        pts2 = scaled[idx]

        # Additionally save and return the selected scale.
        selected_scale = scales[idx]

    # Random translation
    if translation:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += (np.stack([np.random.uniform(-t_min[0], t_max[0], ()),
                           np.random.uniform(-t_min[1],
                                             t_max[1], ())]))[None, ...]

    # Random rotation: sample several rotations, check collision with borders,
    # randomly pick a valid one
    if rotation:
        angles = np.linspace(-max_angle, max_angle, n_angles)
        # in case no rotation is valid
        angles = np.concatenate([[0.], angles], axis=0)
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack(
            [np.cos(angles), -np.sin(angles),
             np.sin(angles), np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul(
                np.tile((pts2 - center)[None, ...], [n_angles+1, 1, 1]),
                rot_mat) + center
        if allow_artifacts:
            # All angles are valid, except angle=0
            valid = np.array(range(n_angles))
        else:
            valid = np.where(np.all((rotated >= 0.)
                             & (rotated < 1.), axis=(1, 2)))[0]
        
        if valid.shape[0] == 0:
            return sample_homography(
                shape, perspective, scaling, rotation, translation,
                n_scales, n_angles, scaling_amplitude, 
                perspective_amplitude_x, perspective_amplitude_y,
                patch_ratio, max_angle, allow_artifacts, translation_overflow)

        idx = valid[np.random.uniform(0., valid.shape[0],
                                      ()).astype(np.int32)]
        pts2 = rotated[idx]

    # Rescale to actual size
    shape = shape[::-1].astype(np.float32)  # different convention [y, x]
    pts1 *= shape[None, ...]
    pts2 *= shape[None, ...]

    def ax(p, q): return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q): return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4)
                      for f in (ax, ay)], axis=0)
    p_mat = np.transpose(np.stack([[pts2[i][j] for i in range(4)
                                    for j in range(2)]], axis=0))
    homo_vec, _, _, _ = np.linalg.lstsq(a_mat, p_mat, rcond=None)

    # Compose the homography vector back to matrix
    homo_mat = np.concatenate([
        homo_vec[0:3, 0][None, ...], homo_vec[3:6, 0][None, ...],
        np.concatenate((homo_vec[6], homo_vec[7], [1]),
                       axis=0)[None, ...]], axis=0)

    return homo_mat, selected_scale


def convert_to_line_segments(junctions, line_map):
    """ Convert junctions and line map to line segments. """
    # Copy the line map
    line_map_tmp = copy.copy(line_map)

    line_segments = np.zeros([0, 4])
    for idx in range(junctions.shape[0]):
        # If no connectivity, just skip it
        if line_map_tmp[idx, :].sum() == 0:
            continue
        # Record the line segment
        else:
            for idx2 in np.where(line_map_tmp[idx, :] == 1)[0]:
                p1 = junctions[idx, :]
                p2 = junctions[idx2, :]
                line_segments = np.concatenate(
                    (line_segments,
                     np.array([p1[0], p1[1], p2[0], p2[1]])[None, ...]),
                    axis=0)
                # Update line_map
                line_map_tmp[idx, idx2] = 0
                line_map_tmp[idx2, idx] = 0

    return line_segments


def compute_valid_mask(image_size, homography,
                       border_margin, valid_mask=None):
    # Warp the mask
    if valid_mask is None:
        initial_mask = np.ones(image_size)
    else:
        initial_mask = valid_mask
    mask = cv2.warpPerspective(
        initial_mask, homography, (image_size[1], image_size[0]),
        flags=cv2.INTER_NEAREST)

    # Optionally perform erosion
    if border_margin > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (border_margin*2, )*2)
        mask = cv2.erode(mask, kernel)
    
    # Perform dilation if border_margin is negative
    if border_margin < 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (abs(int(border_margin))*2, )*2)
        mask = cv2.dilate(mask, kernel)

    return mask


def warp_line_segment(line_segments, homography, image_size):
    """ Warp the line segments using a homography. """
    # Separate the line segements into 2N points to apply matrix operation
    num_segments = line_segments.shape[0]

    junctions = np.concatenate(
        (line_segments[:, :2], # The first junction of each segment.
        line_segments[:, 2:]), # The second junction of each segment.
        axis=0)
    # Convert to homogeneous coordinates
    # Flip the junctions before converting to homogeneous (xy format)
    junctions = np.flip(junctions, axis=1)
    junctions = np.concatenate((junctions, np.ones([2*num_segments, 1])),
                               axis=1)
    warped_junctions = np.matmul(homography, junctions.T).T

    # Convert back to segments
    warped_junctions = warped_junctions[:, :2] / warped_junctions[:, 2:]
    # (Convert back to hw format)
    warped_junctions = np.flip(warped_junctions, axis=1)
    warped_segments = np.concatenate(
        (warped_junctions[:num_segments, :],
         warped_junctions[num_segments:, :]),
        axis=1
    )

    # Check the intersections with the boundary
    warped_segments_new = np.zeros([0, 4])
    image_poly = shapely.geometry.Polygon(
        [[0, 0], [image_size[1]-1, 0], [image_size[1]-1, image_size[0]-1],
        [0, image_size[0]-1]])
    for idx in range(warped_segments.shape[0]):
        # Get the line segment
        seg_raw = warped_segments[idx, :]   # in HW format.
        # Convert to shapely line (flip to xy format)
        seg = shapely.geometry.LineString([np.flip(seg_raw[:2]), 
                                           np.flip(seg_raw[2:])])

        # The line segment is just inside the image.
        if seg.intersection(image_poly) == seg:
            warped_segments_new = np.concatenate((warped_segments_new,
                                                  seg_raw[None, ...]), axis=0)
        
        # Intersect with the image.
        elif seg.intersects(image_poly):
            # Check intersection
            try:
                p = np.array(
                    seg.intersection(image_poly).coords).reshape([-1, 4])
            # If intersect at exact one point, just continue.
            except:
                continue
            segment = np.concatenate([np.flip(p[0, :2]), np.flip(p[0, 2:],
                                     axis=0)])[None, ...]
            warped_segments_new = np.concatenate(
                (warped_segments_new, segment), axis=0)

        else:
            continue

    warped_segments = (np.round(warped_segments_new)).astype(np.int)
    return warped_segments


class homography_transform(object):
    """ # Homography transformations. """
    def __init__(self, image_size, homograpy_config,
                 border_margin=0, min_label_len=20):
        self.homo_config = homograpy_config
        self.image_size = image_size
        self.target_size = (self.image_size[1], self.image_size[0])
        self.border_margin = border_margin
        if (min_label_len < 1) and isinstance(min_label_len, float):
            raise ValueError("[Error] min_label_len should be in pixels.")
        self.min_label_len = min_label_len

    def __call__(self, input_image, junctions, line_map,
                 valid_mask=None, homo=None, scale=None):
        # Sample one random homography or use the given one
        if homo is None or scale is None:
            homo, scale = sample_homography(self.image_size,
                                            **self.homo_config)

        # Warp the image
        warped_image = cv2.warpPerspective(
            input_image, homo, self.target_size, flags=cv2.INTER_LINEAR)
        
        valid_mask = compute_valid_mask(self.image_size, homo,
                                        self.border_margin, valid_mask)

        # Convert junctions and line_map back to line segments
        line_segments = convert_to_line_segments(junctions, line_map)

        # Warp the segments and check the length.
        # Adjust the min_label_length
        warped_segments = warp_line_segment(line_segments, homo,
                                             self.image_size)

        # Convert back to junctions and line_map
        junctions_new = np.concatenate((warped_segments[:, :2],
                                        warped_segments[:, 2:]), axis=0)
        if junctions_new.shape[0] == 0:
            junctions_new = np.zeros([0, 2])
            line_map = np.zeros([0, 0])
            warped_heatmap = np.zeros(self.image_size)
        else:
            junctions_new = np.unique(junctions_new, axis=0)

            # Generate line map from points and segments
            line_map = get_line_map(junctions_new,
                                    warped_segments).astype(np.int)
            # Compute the heatmap
            warped_heatmap = get_line_heatmap(np.flip(junctions_new, axis=1),
                                              line_map, self.image_size)

        return {
            "junctions": junctions_new,
            "warped_image": warped_image,
            "valid_mask": valid_mask,
            "line_map": line_map,
            "warped_heatmap": warped_heatmap,
            "homo": homo,
            "scale": scale
        }
