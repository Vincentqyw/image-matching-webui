import numpy as np


def sample_homography(
        shape, perspective=True, scaling=True, rotation=True,
        translation=True, n_scales=5, n_angles=25, scaling_amplitude=0.1,
        perspective_amplitude_x=0.1, perspective_amplitude_y=0.1,
        patch_ratio=0.5, max_angle=1.57, allow_artifacts=False,
        translation_overflow=0.):
    """ Computes the homography transformation from a random patch in the
        original image to a warped projection with the same image size. The
        original patch, which is initialized with a simple half-size centered
        crop, is iteratively projected, scaled, rotated and translated.

    Args:
        shape: A tuple specifying the height and width of the original image.
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
        An np.array of shape `[3, 3]` corresponding to the homography.
    """
    # Convert shape to ndarry
    if not isinstance(shape, np.ndarray):
        shape = np.array(shape)

    # Corners of the output patch
    margin = (1 - patch_ratio) / 2
    pts1 = margin + np.array([[0, 0], [0, patch_ratio],
                             [patch_ratio, patch_ratio], [patch_ratio, 0]])
    # Corners of the intput image
    pts2 = pts1.copy()

    # Random perspective and affine perturbations
    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)

        # normal distribution with mean=0, std=perspective_amplitude_y/2
        perspective_displacement = np.random.normal(
            0., perspective_amplitude_y/2, [1])
        h_displacement_left = np.random.normal(0., perspective_amplitude_x/2,
                                               [1])
        h_displacement_right = np.random.normal(0., perspective_amplitude_x/2,
                                                [1])
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
            valid = np.arange(n_scales)
        else:
            valid = np.where(np.all((scaled >= 0.) & (scaled < 1.), (1, 2)))[0]
        idx = valid[np.random.uniform(0., valid.shape[0], ()).astype(np.int32)]
        pts2 = scaled[idx]

    # Random translation
    if translation:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += (np.stack([np.random.uniform(-t_min[0], t_max[0], ()),
                           np.random.uniform(-t_min[1], t_max[1], ())]))[None, ...]

    # Random rotation: sample several rotations, check collision with borders,
    # randomly pick a valid one
    if rotation:
        angles = np.linspace(-max_angle, max_angle, n_angles)
        # in case no rotation is valid
        angles = np.concatenate([[0.], angles], axis=0)
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack(
            [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)],
            axis=1), [-1, 2, 2])
        rotated = np.matmul(
                np.tile((pts2 - center)[None, ...], [n_angles+1, 1, 1]),
                rot_mat) + center
        if allow_artifacts:
            valid = np.array(range(n_angles))  # all angles are valid, except angle=0
        else:
            valid = np.where(np.all((rotated >= 0.) & (rotated < 1.), axis=(1, 2)))[0]
        idx = valid[np.random.uniform(0., valid.shape[0], ()).astype(np.int32)]
        pts2 = rotated[idx]

    # Rescale to actual size
    shape = shape[::-1].astype(np.float32)  # different convention [y, x]
    pts1 *= shape[None, ...]
    pts2 *= shape[None, ...]

    def ax(p, q): return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q): return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = np.transpose(np.stack([[pts2[i][j] for i in range(4) for j in range(2)]], axis=0))
    homo_vec, _, _, _ = np.linalg.lstsq(a_mat, p_mat, rcond=None)

    # Compose the homography vector back to matrix
    homo_mat = np.concatenate([homo_vec[0:3, 0][None, ...],
                                homo_vec[3:6, 0][None, ...],
                                np.concatenate((homo_vec[6], homo_vec[7], [1]),
                                axis=0)[None, ...]], axis=0)

    return homo_mat


def warp_points(points, H):
    """ Warp 2D points by an homography H. """
    n_points = points.shape[0]
    reproj_points = points.copy()[:, [1, 0]]
    reproj_points = np.concatenate([reproj_points, np.ones((n_points, 1))],
                                   axis=1)
    reproj_points = H.dot(reproj_points.transpose()).transpose()
    reproj_points = reproj_points[:, :2] / reproj_points[:, 2:]
    reproj_points = reproj_points[:, [1, 0]]
    return reproj_points


def warp_lines(lines, H):
    """ Warp lines of the shape [N, 2, 2] by an homography H. """
    return warp_points(lines.reshape(-1, 2), H).reshape(-1, 2, 2)
