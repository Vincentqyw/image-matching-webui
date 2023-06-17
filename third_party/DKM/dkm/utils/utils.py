import numpy as np
import cv2
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Code taken from https://github.com/PruneTruong/DenseMatching/blob/40c29a6b5c35e86b9509e65ab0cd12553d998e5f/validation/utils_pose_estimation.py
# --- GEOMETRY ---
def estimate_pose(kpts0, kpts1, K0, K1, norm_thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    K0inv = np.linalg.inv(K0[:2,:2])
    K1inv = np.linalg.inv(K1[:2,:2])

    kpts0 = (K0inv @ (kpts0-K0[None,:2,2]).T).T 
    kpts1 = (K1inv @ (kpts1-K1[None,:2,2]).T).T

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf, method=cv2.RANSAC
    )

    ret = None
    if E is not None:
        best_num_inliers = 0

        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t, mask.ravel() > 0)
    return ret


def rotate_intrinsic(K, n):
    base_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    rot = np.linalg.matrix_power(base_rot, n)
    return rot @ K


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array(
            [
                [np.cos(r), -np.sin(r), 0.0, 0.0],
                [np.sin(r), np.cos(r), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    scales = np.diag([1.0 / scales[0], 1.0 / scales[1], 1.0])
    return np.dot(scales, K)


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t.squeeze(), t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs


# From Patch2Pix https://github.com/GrumpyZhou/patch2pix
def get_depth_tuple_transform_ops(resize=None, normalize=True, unscale=False):
    ops = []
    if resize:
        ops.append(TupleResize(resize, mode=InterpolationMode.BILINEAR))
    return TupleCompose(ops)


def get_tuple_transform_ops(resize=None, normalize=True, unscale=False):
    ops = []
    if resize:
        ops.append(TupleResize(resize))
    if normalize:
        ops.append(TupleToTensorScaled())
        ops.append(
            TupleNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )  # Imagenet mean/std
    else:
        if unscale:
            ops.append(TupleToTensorUnscaled())
        else:
            ops.append(TupleToTensorScaled())
    return TupleCompose(ops)


class ToTensorScaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor, scale the range to [0, 1]"""

    def __call__(self, im):
        if not isinstance(im, torch.Tensor):
            im = np.array(im, dtype=np.float32).transpose((2, 0, 1))
            im /= 255.0
            return torch.from_numpy(im)
        else:
            return im

    def __repr__(self):
        return "ToTensorScaled(./255)"


class TupleToTensorScaled(object):
    def __init__(self):
        self.to_tensor = ToTensorScaled()

    def __call__(self, im_tuple):
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self):
        return "TupleToTensorScaled(./255)"


class ToTensorUnscaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor"""

    def __call__(self, im):
        return torch.from_numpy(np.array(im, dtype=np.float32).transpose((2, 0, 1)))

    def __repr__(self):
        return "ToTensorUnscaled()"


class TupleToTensorUnscaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor"""

    def __init__(self):
        self.to_tensor = ToTensorUnscaled()

    def __call__(self, im_tuple):
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self):
        return "TupleToTensorUnscaled()"


class TupleResize(object):
    def __init__(self, size, mode=InterpolationMode.BICUBIC):
        self.size = size
        self.resize = transforms.Resize(size, mode)

    def __call__(self, im_tuple):
        return [self.resize(im) for im in im_tuple]

    def __repr__(self):
        return "TupleResize(size={})".format(self.size)


class TupleNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, im_tuple):
        return [self.normalize(im) for im in im_tuple]

    def __repr__(self):
        return "TupleNormalize(mean={}, std={})".format(self.mean, self.std)


class TupleCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, im_tuple):
        for t in self.transforms:
            im_tuple = t(im_tuple)
        return im_tuple

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    # https://github.com/zju3dv/LoFTR/blob/94e98b695be18acb43d5d3250f52226a8e36f839/src/loftr/utils/geometry.py adapted from here
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>, should be normalized in (-1,1)
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    (
        n,
        h,
        w,
    ) = depth0.shape
    kpts0_depth = F.grid_sample(depth0[:, None], kpts0[:, :, None], mode="bilinear")[
        :, 0, :, 0
    ]
    kpts0 = torch.stack(
        (w * (kpts0[..., 0] + 1) / 2, h * (kpts0[..., 1] + 1) / 2), dim=-1
    )  # [-1+1/h, 1-1/h] -> [0.5, h-0.5]
    # Sample depth, get calculable_mask on depth != 0
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = (
        torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1)
        * kpts0_depth[..., None]
    )  # (N, L, 3)
    kpts0_n = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)
    kpts0_cam = kpts0_n

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (
        w_kpts0_h[:, :, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (
        (w_kpts0[:, :, 0] > 0)
        * (w_kpts0[:, :, 0] < w - 1)
        * (w_kpts0[:, :, 1] > 0)
        * (w_kpts0[:, :, 1] < h - 1)
    )
    w_kpts0 = torch.stack(
        (2 * w_kpts0[..., 0] / w - 1, 2 * w_kpts0[..., 1] / h - 1), dim=-1
    )  # from [0.5,h-0.5] -> [-1+1/h, 1-1/h]
    # w_kpts0[~covisible_mask, :] = -5 # xd

    w_kpts0_depth = F.grid_sample(
        depth1[:, None], w_kpts0[:, :, None], mode="bilinear"
    )[:, 0, :, 0]
    consistent_mask = (
        (w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth
    ).abs() < 0.05
    valid_mask = nonzero_mask * covisible_mask * consistent_mask

    return valid_mask, w_kpts0


imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


def numpy_to_pil(x: np.ndarray):
    """
    Args:
        x: Assumed to be of shape (h,w,c)
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.max() <= 1.01:
        x *= 255
    x = x.astype(np.uint8)
    return Image.fromarray(x)


def tensor_to_pil(x, unnormalize=False):
    if unnormalize:
        x = x * imagenet_std[:, None, None] + imagenet_mean[:, None, None]
    x = x.detach().permute(1, 2, 0).cpu().numpy()
    x = np.clip(x, 0.0, 1.0)
    return numpy_to_pil(x)


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def to_cpu(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cpu()
    return batch


def get_pose(calib):
    w, h = np.array(calib["imsize"])[0]
    return np.array(calib["K"]), np.array(calib["R"]), np.array(calib["T"]).T, h, w


def compute_relative_pose(R1, t1, R2, t2):
    rots = R2 @ (R1.T)
    trans = -rots @ t1 + t2
    return rots, trans
