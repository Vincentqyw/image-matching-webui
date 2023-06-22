import numpy as np
import torch
from kornia.geometry.transform import warp_perspective
from kornia.morphology import erosion

from ..datasets.utils.homographies import sample_homography


default_H_params = {
    'translation': True,
    'rotation': True,
    'scaling': True,
    'perspective': True,
    'scaling_amplitude': 0.2,
    'perspective_amplitude_x': 0.2,
    'perspective_amplitude_y': 0.2,
    'patch_ratio': 0.85,
    'max_angle': 1.57,
    'allow_artifacts': True
}

erosion_kernel = torch.tensor(
    [[0, 0, 1, 0, 0],
     [0, 1, 1, 1, 0],
     [1, 1, 1, 1, 1],
     [0, 1, 1, 1, 0],
     [0, 0, 1, 0, 0]],
    dtype=torch.float
)


def torch_homography_adaptation(img, net, num_H=10, H_params=default_H_params,
                                aggregation='median'):
    """ Perform homography adaptation at test time using Pytorch.
        Only works with a batch size of 1. """
    assert len(img) == 1, "torch_homography_adaptation only with a batch size of 1."
    bs = 10
    device = img.device
    h, w = img.shape[2:4]
    
    # Generate homographies and warp the image
    Hs = []
    for i in range(num_H):
        if i == 0:
            # Always include at least the identity
            Hs.append(torch.eye(3, dtype=torch.float, device=device))
        else:
            Hs.append(torch.tensor(
                sample_homography((h, w), **default_H_params),
                dtype=torch.float, device=device))
    Hs = torch.stack(Hs, dim=0)
        
    # Loop through all mini batches
    n_mini_batch = int(np.ceil(num_H / bs))
    dfs = torch.empty((0, h, w), dtype=torch.float, device=device)
    angles = torch.empty((0, h, w), dtype=torch.float, device=device)
    offsets = torch.empty((0, h, w, 2), dtype=torch.float, device=device)
    counts = torch.empty((0, h, w), dtype=torch.float, device=device)
    for i in range(n_mini_batch):
        H = Hs[i*bs:(i+1)*bs]

        # Warp the image
        warped_imgs = warp_perspective(
            img.repeat(len(H), 1, 1, 1), H, (h, w), mode='bilinear')

        # Forward pass
        with torch.no_grad():
            outs = net({'image': warped_imgs})

            # Warp back the results
            df, angle, offset, count = warp_afm(
                outs['df'], outs['line_level'],
                outs['offset'], torch.inverse(H))

        # Aggregate the results
        dfs = torch.cat([dfs, df], dim=0)
        angles = torch.cat([angles, angle], dim=0)
        offsets = torch.cat([offsets, offset], dim=0)
        counts = torch.cat([counts, count], dim=0)

    # Aggregate the results
    if aggregation == 'mean':
        df = (dfs * counts).sum(dim=0) / counts.sum(dim=0)
        offset = ((offsets * counts.unsqueeze(-1)).sum(dim=0)
                  / counts.sum(dim=0).unsqueeze(-1))
    elif aggregation == 'median':
        df[counts == 0] = np.nan
        df = np.nanmedian(df, dim=0)[0]
        offset[counts == 0] = np.nan
        offset = np.nanmedian(offset, dim=0)[0]
        # df = masked_median(dfs, counts)
        # offset = masked_median(offsets, counts[..., None].repeat(1, 1, 1, 2))
    else:
        raise ValueError("Unknown aggregation method: " + aggregation)

    # Median of the angle
    angles = angles.reshape(num_H, h * w)
    counts = counts.reshape(num_H, h * w)
    circ_bound = (torch.min(np.pi - angles, angles)
                  * counts).sum(0) / counts.sum(0) < 0.3
    angles[:, circ_bound] -= torch.where(
        angles[:, circ_bound] > np.pi /2,
        torch.ones_like(angles[:, circ_bound]) * np.pi,
        torch.zeros_like(angles[:, circ_bound]))
    # angle = torch.remainder(masked_median(angles, counts),
    #                         np.pi).reshape(h, w)
    angle[counts == 0] = np.nan
    angle = torch.remainder(torch.nanmedian(angle, dim=0)[0],
                            np.pi).reshape(h, w)

    return df, angle, offset


def warp_points(points, H):
    """ Warp batched 2D points by a batched homography H:
        points is [bs, ..., 2] and H is [bs, 3, 3]. """
    shape = points.shape
    bs = len(points)
    reproj_points = points.reshape(bs, -1, 2)[:, :, [1, 0]].transpose(1, 2)
    reproj_points = torch.cat(
        [reproj_points, torch.ones_like(reproj_points[:, :1])], dim=1)
    reproj_points = (H @ reproj_points).transpose(1, 2)
    reproj_points = reproj_points[..., :2] / reproj_points[..., 2:]
    reproj_points = reproj_points[..., [1, 0]]
    return reproj_points.reshape(shape)


def warp_afm(df, angle, offset, H):
    """ Warp an attraction field defined by a DF, line level angle and offset
        field, with a set of homographies. All tensors are batched. """
    b_size, h, w = df.shape
    device = df.device

    # Warp the closest point on a line
    pix_loc = torch.stack(torch.meshgrid(
        torch.arange(h, dtype=torch.float, device=device),
        torch.arange(w, dtype=torch.float, device=device),
        indexing='ij'), dim=-1)[None].repeat(b_size, 1, 1, 1)
    closest = pix_loc + offset
    warped_closest = warp_points(closest, H)
    warped_pix_loc = warp_points(pix_loc, H)
    offset_norm = torch.norm(offset, dim=-1)
    zero_offset = offset_norm < 1e-3
    offset_norm[zero_offset] = 1
    scaling = (torch.norm(warped_closest - warped_pix_loc, dim=-1)
               / offset_norm)
    scaling[zero_offset] = 0
    warped_closest = warp_perspective(
        warped_closest.permute(0, 3, 1, 2), H, (h, w),
        mode='nearest').permute(0, 2, 3, 1)
    warped_offset = warped_closest - pix_loc
    
    # Warp the DF
    warped_df = warp_perspective(df.unsqueeze(1), H, (h, w),
                                 mode='bilinear')[:, 0]
    warped_scaling = warp_perspective(scaling.unsqueeze(1), H, (h, w),
                                      mode='bilinear')[:, 0]
    warped_df *= warped_scaling

    # Warp the angle
    closest = pix_loc + torch.stack([torch.sin(angle), torch.cos(angle)],
                                    dim=-1)
    warped_closest = warp_points(closest, H)
    warped_angle = torch.remainder(torch.atan2(
        warped_closest[..., 0] - warped_pix_loc[..., 0],
        warped_closest[..., 1] - warped_pix_loc[..., 1]), np.pi)
    warped_angle = warp_perspective(warped_angle.unsqueeze(1), H, (h, w),
                                    mode='nearest')[:, 0]
    
    # Compute the counts of valid pixels
    H_inv = torch.inverse(H)
    counts = warp_perspective(torch.ones_like(df).unsqueeze(1), H_inv, (h, w),
                              mode='nearest')
    counts = erosion(counts, erosion_kernel.to(device))
    counts = warp_perspective(counts, H, (h, w), mode='nearest')[:, 0]

    return warped_df, warped_angle, warped_offset, counts


def masked_median(arr, mask):
    """ Compute the median of a batched tensor arr, taking into account a
        mask of valid pixels. We assume the batch size to be small. """
    b_size = len(arr)
    arr_shape = arr.shape[1:]
    flat_arr = arr.reshape(b_size, -1)
    flat_mask = mask.reshape(b_size, -1)
    counts = flat_mask.sum(dim=0)
    out_median = torch.zeros_like(flat_arr[0])
    for i in range(1, b_size + 1):
        curr_mask = counts == i
        curr_val = flat_arr.t()[curr_mask]
        curr_val = curr_val[flat_mask.t()[curr_mask] == 1].reshape(-1, i)
        out_median[curr_mask] = torch.quantile(curr_val, 0.5, dim=1)
    return out_median.reshape(arr_shape)
