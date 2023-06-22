import numpy as np
from skimage import measure as skmeasure
from torch._six import string_classes
import collections.abc as collections
import torch
import cv2


def map_tensor(input_, func):
    if isinstance(input_, torch.Tensor):
        return func(input_)
    elif isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    else:
        raise TypeError(
            f'input must be tensor, dict or list; found {type(input_)}')


def batch_to_numpy(batch):
    return map_tensor(batch, lambda tensor: tensor.cpu().numpy())


def batch_to_device(batch, device, non_blocking=False):
    def _func(tensor):
        return tensor.to(device=device, non_blocking=non_blocking)

    return map_tensor(batch, _func)


def create_pairwise_conv_kernel(kernel_size,
                                center_size=16,
                                dia_stride=4,
                                random_dia=False, random_tot_1=1024,
                                return_neiCnt=False, diff_kernel=True):
    '''
    @func: create 1 x fH x fW x num_ch kernel for conv2D to compute pairwise-diff
    @param: kernel_size --[filter_ht, filter_wd]
            center_size -- scalar, neighbours in this window is all considered.
            dia_stride -- scalar, consider dialated pairs on further sides.
            random_dia -- perform random choice to assign neighbour
            random_tot_1 -- # of 1 in random result
            return_neiCnt -- return # of selected neighbour
    '''
    # selected neighbours for computing pairwise difference
    selected_neiI  = np.zeros(kernel_size)
    axis_x, axis_y = np.meshgrid(range(kernel_size[1]), range(kernel_size[0]))
    cy, cx         = kernel_size[0]//2, kernel_size[1]//2
    dist_mat_x     = np.abs(axis_x-cx)
    dist_mat_y     = np.abs(axis_y-cy)
    selected_neiI[(dist_mat_x+dist_mat_y)<center_size] = 1
    if random_dia == False:
        flagI = (np.mod(dist_mat_x, dia_stride) + np.mod(dist_mat_y, dia_stride)) == 0
        #flagI = (np.mod(dist_mat_x+dist_mat_y, dia_stride)==0)
        selected_neiI[flagI] = 1
    else:
        prob_1 = float(random_tot_1) / (kernel_size[0]*kernel_size[1])
        np.random.seed(17929104)
        random_neiI = np.random.choice([0,1], size=kernel_size, p=[1-prob_1, prob_1])
        selected_neiI[random_neiI==1] = 1
    selected_neiI[cy, cx:] = 1
    selected_neiI[cy:, cx] = 1
    selected_neiI[cy, cx]  = 0

    # label each neighbour with continuous indices
    identity_label = axis_y * kernel_size[1] + axis_x
    selected_neiI  = identity_label * selected_neiI

    # remove duplicate pairwise
    selected_neiI[:cy, :]       = 0
    selected_neiI[:cy+1, :cx+1] = 0

    # convert to one hot kernel
    label_neiI = skmeasure.label(selected_neiI)
    label_neiI = np.reshape(label_neiI, [-1])
    kernel     = np.eye(label_neiI.max()+1)[label_neiI]
    kernel     = np.reshape(kernel, [1, kernel_size[0], kernel_size[1], -1])

    if diff_kernel:
        kernel     = 0 - kernel[..., 1:]
        kernel[0, cy, cx, :] = 1

    if return_neiCnt:
        return kernel, label_neiI.max()
    else:
        return kernel


def check_nan_or_inf(t):
    return torch.any(find_nan_or_inf(t))


def find_nan_or_inf(t):
    return torch.logical_or(torch.isnan(t), torch.isinf(t))


def dot(a, b, dim=-1):
    """ Dot product of a and b along the specified dimension. """
    return (a * b).sum(dim)


def bilinear_interpolate_numpy(im, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return (Ia.T * wa).T + (Ib.T * wb).T + (Ic.T * wc).T + (Id.T * wd).T


def nn_interpolate_numpy(img, x, y):
    xi = np.clip(np.round(x).astype(int), 0, img.shape[1] - 1)
    yi = np.clip(np.round(y).astype(int), 0, img.shape[0] - 1)
    return img[yi, xi]


def compute_image_grad(img, ksize=7):
    blur_img = cv2.GaussianBlur(img, (ksize, ksize), 1).astype(np.float32)
    dx = np.zeros_like(blur_img)
    dy = np.zeros_like(blur_img)
    dx[:, 1:] = (blur_img[:, 1:] - blur_img[:, :-1]) / 2
    dx[1:, 1:] = dx[:-1, 1:] + dx[1:, 1:]
    dy[1:] = (blur_img[1:] - blur_img[:-1]) / 2
    dy[1:, 1:] = dy[1:, :-1] + dy[1:, 1:]
    gradnorm = np.sqrt(dx ** 2 + dy ** 2)
    gradangle = np.arctan2(dy, dx)
    return dx, dy, gradnorm, gradangle


def align_with_grad_angle(angle, img):
    """ Starting from an angle in [0, pi], find the sign of the angle based on
        the image gradient of the corresponding pixel. """
    # Image gradient
    img_grad_angle = compute_image_grad(img)[3]
    
    # Compute the distance of the image gradient to the angle
    # and angle - pi
    pred_grad = np.mod(angle, np.pi)  # in [0, pi]
    pos_dist = np.minimum(np.abs(img_grad_angle - pred_grad),
                          2 * np.pi - np.abs(img_grad_angle - pred_grad))
    neg_dist = np.minimum(
        np.abs(img_grad_angle - pred_grad + np.pi),
        2 * np.pi - np.abs(img_grad_angle - pred_grad + np.pi))
    
    # Assign the new grad angle to the closest of the two
    is_pos_closest = np.argmin(np.stack([neg_dist, pos_dist],
                                        axis=-1), axis=-1).astype(bool)
    new_grad_angle = np.where(is_pos_closest, pred_grad, pred_grad - np.pi)
    return new_grad_angle, img_grad_angle


def preprocess_angle(angle, img, mask=False):
    """ Convert a grad angle field into a line level angle, using
        the image gradient to get the right orientation. """
    oriented_grad_angle, img_grad_angle = align_with_grad_angle(angle, img)
    oriented_grad_angle = np.mod(oriented_grad_angle - np.pi / 2, 2 * np.pi)
    if mask:
        oriented_grad_angle[0] = -1024
        oriented_grad_angle[:, 0] = -1024
    return oriented_grad_angle.astype(np.float64), img_grad_angle
