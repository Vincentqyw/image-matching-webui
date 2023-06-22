"""
Loss function implementations.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry import warp_perspective

from ..misc.geometry_utils import (keypoints_to_grid, get_dist_mask,
                                   get_common_line_mask)


def get_loss_and_weights(model_cfg, device=torch.device("cuda")):
    """ Get loss functions and either static or dynamic weighting. """
    # Get the global weighting policy
    w_policy = model_cfg.get("weighting_policy", "static")
    if not w_policy in ["static", "dynamic"]:
        raise ValueError("[Error] Not supported weighting policy.")
    
    loss_func = {}
    loss_weight = {}
    # Get junction loss function and weight
    w_junc, junc_loss_func = get_junction_loss_and_weight(model_cfg, w_policy)
    loss_func["junc_loss"] = junc_loss_func.to(device)
    loss_weight["w_junc"] = w_junc

    # Get heatmap loss function and weight
    w_heatmap, heatmap_loss_func = get_heatmap_loss_and_weight(
        model_cfg, w_policy, device)
    loss_func["heatmap_loss"] = heatmap_loss_func.to(device)
    loss_weight["w_heatmap"] = w_heatmap

    # [Optionally] get descriptor loss function and weight
    if model_cfg.get("descriptor_loss_func", None) is not None:
        w_descriptor, descriptor_loss_func = get_descriptor_loss_and_weight(
            model_cfg, w_policy)
        loss_func["descriptor_loss"] = descriptor_loss_func.to(device)
        loss_weight["w_desc"] = w_descriptor

    return loss_func, loss_weight


def get_junction_loss_and_weight(model_cfg, global_w_policy):
    """ Get the junction loss function and weight. """
    junction_loss_cfg = model_cfg.get("junction_loss_cfg", {})
    
    # Get the junction loss weight
    w_policy = junction_loss_cfg.get("policy", global_w_policy)
    if w_policy == "static":
        w_junc = torch.tensor(model_cfg["w_junc"], dtype=torch.float32)
    elif w_policy == "dynamic":
        w_junc = nn.Parameter(
            torch.tensor(model_cfg["w_junc"], dtype=torch.float32),
            requires_grad=True)
    else:
        raise ValueError(
    "[Error] Unknown weighting policy for junction loss weight.")

    # Get the junction loss function
    junc_loss_name = model_cfg.get("junction_loss_func", "superpoint")
    if junc_loss_name == "superpoint":
        junc_loss_func = JunctionDetectionLoss(model_cfg["grid_size"],
                                               model_cfg["keep_border_valid"])
    else:
        raise ValueError("[Error] Not supported junction loss function.")

    return w_junc, junc_loss_func


def get_heatmap_loss_and_weight(model_cfg, global_w_policy, device):
    """ Get the heatmap loss function and weight. """
    heatmap_loss_cfg = model_cfg.get("heatmap_loss_cfg", {})

    # Get the heatmap loss weight
    w_policy = heatmap_loss_cfg.get("policy", global_w_policy)
    if w_policy == "static":
        w_heatmap = torch.tensor(model_cfg["w_heatmap"], dtype=torch.float32)
    elif w_policy == "dynamic":
        w_heatmap = nn.Parameter(
            torch.tensor(model_cfg["w_heatmap"], dtype=torch.float32), 
            requires_grad=True)
    else:
        raise ValueError(
    "[Error] Unknown weighting policy for junction loss weight.")

    # Get the corresponding heatmap loss based on the config
    heatmap_loss_name = model_cfg.get("heatmap_loss_func", "cross_entropy")
    if heatmap_loss_name == "cross_entropy":
        # Get the heatmap class weight (always static)
        heatmap_class_w = model_cfg.get("w_heatmap_class", 1.)
        class_weight = torch.tensor(
            np.array([1., heatmap_class_w])).to(torch.float).to(device)
        heatmap_loss_func = HeatmapLoss(class_weight=class_weight)
    else:
        raise ValueError("[Error] Not supported heatmap loss function.")

    return w_heatmap, heatmap_loss_func


def get_descriptor_loss_and_weight(model_cfg, global_w_policy):
    """ Get the descriptor loss function and weight. """
    descriptor_loss_cfg = model_cfg.get("descriptor_loss_cfg", {})
    
    # Get the descriptor loss weight
    w_policy = descriptor_loss_cfg.get("policy", global_w_policy)
    if w_policy == "static":
        w_descriptor = torch.tensor(model_cfg["w_desc"], dtype=torch.float32)
    elif w_policy == "dynamic":
        w_descriptor = nn.Parameter(torch.tensor(model_cfg["w_desc"],
                                    dtype=torch.float32), requires_grad=True)
    else:
        raise ValueError(
    "[Error] Unknown weighting policy for descriptor loss weight.")

    # Get the descriptor loss function
    descriptor_loss_name = model_cfg.get("descriptor_loss_func",
                                         "regular_sampling")
    if descriptor_loss_name == "regular_sampling":
        descriptor_loss_func = TripletDescriptorLoss(
            descriptor_loss_cfg["grid_size"],
            descriptor_loss_cfg["dist_threshold"],
            descriptor_loss_cfg["margin"])
    else:
        raise ValueError("[Error] Not supported descriptor loss function.")

    return w_descriptor, descriptor_loss_func


def space_to_depth(input_tensor, grid_size):
    """ PixelUnshuffle for pytorch. """
    N, C, H, W = input_tensor.size()
    # (N, C, H//bs, bs, W//bs, bs)
    x = input_tensor.view(N, C, H // grid_size, grid_size, W // grid_size, grid_size)
    # (N, bs, bs, C, H//bs, W//bs)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
    # (N, C*bs^2, H//bs, W//bs)
    x = x.view(N, C * (grid_size ** 2), H // grid_size, W // grid_size)
    return x


def junction_detection_loss(junction_map, junc_predictions, valid_mask=None,
                            grid_size=8, keep_border=True):
    """ Junction detection loss. """
    # Convert junc_map to channel tensor
    junc_map = space_to_depth(junction_map, grid_size)
    map_shape = junc_map.shape[-2:]
    batch_size = junc_map.shape[0]
    dust_bin_label = torch.ones(
        [batch_size, 1, map_shape[0],
         map_shape[1]]).to(junc_map.device).to(torch.int)
    junc_map = torch.cat([junc_map*2, dust_bin_label], dim=1)
    labels = torch.argmax(
        junc_map.to(torch.float) +
        torch.distributions.Uniform(0, 0.1).sample(junc_map.shape).to(junc_map.device),
        dim=1)

    # Also convert the valid mask to channel tensor
    valid_mask = (torch.ones(junction_map.shape) if valid_mask is None
                  else valid_mask)
    valid_mask = space_to_depth(valid_mask, grid_size)
    
    # Compute junction loss on the border patch or not
    if keep_border:
        valid_mask = torch.sum(valid_mask.to(torch.bool).to(torch.int),
                               dim=1, keepdim=True) > 0
    else:
        valid_mask = torch.sum(valid_mask.to(torch.bool).to(torch.int),
                               dim=1, keepdim=True) >= grid_size * grid_size

    # Compute the classification loss
    loss_func = nn.CrossEntropyLoss(reduction="none")
    # The loss still need NCHW format
    loss = loss_func(input=junc_predictions,
                     target=labels.to(torch.long))
    
    # Weighted sum by the valid mask
    loss_ = torch.sum(loss * torch.squeeze(valid_mask.to(torch.float),
                                           dim=1), dim=[0, 1, 2])
    loss_final = loss_ / torch.sum(torch.squeeze(valid_mask.to(torch.float),
                                                 dim=1))

    return loss_final


def heatmap_loss(heatmap_gt, heatmap_pred, valid_mask=None,
                 class_weight=None):
    """ Heatmap prediction loss. """
    # Compute the classification loss on each pixel
    if class_weight is None:
        loss_func = nn.CrossEntropyLoss(reduction="none")
    else:
        loss_func = nn.CrossEntropyLoss(class_weight, reduction="none")

    loss = loss_func(input=heatmap_pred,
                     target=torch.squeeze(heatmap_gt.to(torch.long), dim=1))

    # Weighted sum by the valid mask
    # Sum over H and W
    loss_spatial_sum = torch.sum(loss * torch.squeeze(
        valid_mask.to(torch.float), dim=1), dim=[1, 2])
    valid_spatial_sum = torch.sum(torch.squeeze(valid_mask.to(torch.float32),
                                                dim=1), dim=[1, 2])
    # Mean to single scalar over batch dimension
    loss = torch.sum(loss_spatial_sum) / torch.sum(valid_spatial_sum)

    return loss


class JunctionDetectionLoss(nn.Module):
    """ Junction detection loss. """
    def __init__(self, grid_size, keep_border):
        super(JunctionDetectionLoss, self).__init__()
        self.grid_size = grid_size
        self.keep_border = keep_border

    def forward(self, prediction, target, valid_mask=None):
        return junction_detection_loss(target, prediction, valid_mask,
                                       self.grid_size, self.keep_border)


class HeatmapLoss(nn.Module):
    """ Heatmap prediction loss. """
    def __init__(self, class_weight):
        super(HeatmapLoss, self).__init__()
        self.class_weight = class_weight

    def forward(self, prediction, target, valid_mask=None):
        return heatmap_loss(target, prediction, valid_mask, self.class_weight)


class RegularizationLoss(nn.Module):
    """ Module for regularization loss. """
    def __init__(self):
        super(RegularizationLoss, self).__init__()
        self.name = "regularization_loss"
        self.loss_init = torch.zeros([])

    def forward(self, loss_weights):
        # Place it to the same device
        loss = self.loss_init.to(loss_weights["w_junc"].device)
        for _, val in loss_weights.items():
            if isinstance(val, nn.Parameter):
                loss += val
        
        return loss


def triplet_loss(desc_pred1, desc_pred2, points1, points2, line_indices,
                 epoch, grid_size=8, dist_threshold=8,
                 init_dist_threshold=64, margin=1):
    """ Regular triplet loss for descriptor learning. """
    b_size, _, Hc, Wc = desc_pred1.size()
    img_size = (Hc * grid_size, Wc * grid_size)
    device = desc_pred1.device

    # Extract valid keypoints
    n_points = line_indices.size()[1]
    valid_points = line_indices.bool().flatten()
    n_correct_points = torch.sum(valid_points).item()
    if n_correct_points == 0:
        return torch.tensor(0., dtype=torch.float, device=device)

    # Check which keypoints are too close to be matched
    # dist_threshold is decreased at each epoch for easier training
    dist_threshold = max(dist_threshold,
                         2 * init_dist_threshold // (epoch + 1))
    dist_mask = get_dist_mask(points1, points2, valid_points, dist_threshold)

    # Additionally ban negative mining along the same line
    common_line_mask = get_common_line_mask(line_indices, valid_points)
    dist_mask = dist_mask | common_line_mask

    # Convert the keypoints to a grid suitable for interpolation
    grid1 = keypoints_to_grid(points1, img_size)
    grid2 = keypoints_to_grid(points2, img_size)

    # Extract the descriptors
    desc1 = F.grid_sample(desc_pred1, grid1).permute(
        0, 2, 3, 1).reshape(b_size * n_points, -1)[valid_points]
    desc1 = F.normalize(desc1, dim=1)
    desc2 = F.grid_sample(desc_pred2, grid2).permute(
        0, 2, 3, 1).reshape(b_size * n_points, -1)[valid_points]
    desc2 = F.normalize(desc2, dim=1)
    desc_dists = 2 - 2 * (desc1 @ desc2.t())

    # Positive distance loss
    pos_dist = torch.diag(desc_dists)

    # Negative distance loss
    max_dist = torch.tensor(4., dtype=torch.float, device=device)
    desc_dists[
        torch.arange(n_correct_points, dtype=torch.long),
        torch.arange(n_correct_points, dtype=torch.long)] = max_dist
    desc_dists[dist_mask] = max_dist
    neg_dist = torch.min(torch.min(desc_dists, dim=1)[0],
                         torch.min(desc_dists, dim=0)[0])

    triplet_loss = F.relu(margin + pos_dist - neg_dist)
    return triplet_loss, grid1, grid2, valid_points


class TripletDescriptorLoss(nn.Module):
    """ Triplet descriptor loss. """
    def __init__(self, grid_size, dist_threshold, margin):
        super(TripletDescriptorLoss, self).__init__()
        self.grid_size = grid_size
        self.init_dist_threshold = 64
        self.dist_threshold = dist_threshold
        self.margin = margin

    def forward(self, desc_pred1, desc_pred2, points1,
                points2, line_indices, epoch):
        return self.descriptor_loss(desc_pred1, desc_pred2, points1,
                                    points2, line_indices, epoch)

    # The descriptor loss based on regularly sampled points along the lines
    def descriptor_loss(self, desc_pred1, desc_pred2, points1,
                        points2, line_indices, epoch):
        return torch.mean(triplet_loss(
            desc_pred1, desc_pred2, points1, points2, line_indices, epoch,
            self.grid_size, self.dist_threshold, self.init_dist_threshold,
            self.margin)[0])


class TotalLoss(nn.Module):
    """ Total loss summing junction, heatma, descriptor
        and regularization losses. """
    def __init__(self, loss_funcs, loss_weights, weighting_policy):
        super(TotalLoss, self).__init__()
        # Whether we need to compute the descriptor loss
        self.compute_descriptors = "descriptor_loss" in loss_funcs.keys()

        self.loss_funcs = loss_funcs
        self.loss_weights = loss_weights
        self.weighting_policy = weighting_policy

        # Always add regularization loss (it will return zero if not used)
        self.loss_funcs["reg_loss"] = RegularizationLoss().cuda()

    def forward(self, junc_pred, junc_target, heatmap_pred,
                heatmap_target, valid_mask=None):
        """ Detection only loss. """
        # Compute the junction loss
        junc_loss = self.loss_funcs["junc_loss"](junc_pred, junc_target,
                                                 valid_mask)
        # Compute the heatmap loss
        heatmap_loss = self.loss_funcs["heatmap_loss"](
            heatmap_pred, heatmap_target, valid_mask)

        # Compute the total loss.
        if self.weighting_policy == "dynamic":
            reg_loss = self.loss_funcs["reg_loss"](self.loss_weights)
            total_loss = junc_loss * torch.exp(-self.loss_weights["w_junc"]) + \
                         heatmap_loss * torch.exp(-self.loss_weights["w_heatmap"]) + \
                         reg_loss
            
            return {
                "total_loss": total_loss,
                "junc_loss": junc_loss,
                "heatmap_loss": heatmap_loss,
                "reg_loss": reg_loss,
                "w_junc": torch.exp(-self.loss_weights["w_junc"]).item(),
                "w_heatmap": torch.exp(-self.loss_weights["w_heatmap"]).item(),
            }
        
        elif self.weighting_policy == "static":
            total_loss = junc_loss * self.loss_weights["w_junc"] + \
                         heatmap_loss * self.loss_weights["w_heatmap"]
            
            return {
                "total_loss": total_loss,
                "junc_loss": junc_loss,
                "heatmap_loss": heatmap_loss
            }

        else:
            raise ValueError("[Error] Unknown weighting policy.")
    
    def forward_descriptors(self, 
            junc_map_pred1, junc_map_pred2, junc_map_target1,
            junc_map_target2, heatmap_pred1, heatmap_pred2, heatmap_target1,
            heatmap_target2, line_points1, line_points2, line_indices,
            desc_pred1, desc_pred2, epoch, valid_mask1=None,
            valid_mask2=None):
        """ Loss for detection + description. """
        # Compute junction loss
        junc_loss = self.loss_funcs["junc_loss"](
            torch.cat([junc_map_pred1, junc_map_pred2], dim=0), 
            torch.cat([junc_map_target1, junc_map_target2], dim=0),
            torch.cat([valid_mask1, valid_mask2], dim=0)
        )
        # Get junction loss weight (dynamic or not)
        if isinstance(self.loss_weights["w_junc"], nn.Parameter):
            w_junc = torch.exp(-self.loss_weights["w_junc"])
        else:
            w_junc = self.loss_weights["w_junc"]

        # Compute heatmap loss
        heatmap_loss = self.loss_funcs["heatmap_loss"](
            torch.cat([heatmap_pred1, heatmap_pred2], dim=0), 
            torch.cat([heatmap_target1, heatmap_target2], dim=0),
            torch.cat([valid_mask1, valid_mask2], dim=0)
        )
        # Get heatmap loss weight (dynamic or not)
        if isinstance(self.loss_weights["w_heatmap"], nn.Parameter):
            w_heatmap = torch.exp(-self.loss_weights["w_heatmap"])
        else:
            w_heatmap = self.loss_weights["w_heatmap"]

        # Compute the descriptor loss
        descriptor_loss = self.loss_funcs["descriptor_loss"](
            desc_pred1, desc_pred2, line_points1,
            line_points2, line_indices, epoch)
        # Get descriptor loss weight (dynamic or not)
        if isinstance(self.loss_weights["w_desc"], nn.Parameter):
            w_descriptor = torch.exp(-self.loss_weights["w_desc"])
        else:
            w_descriptor = self.loss_weights["w_desc"]

        # Update the total loss
        total_loss = (junc_loss * w_junc
                      + heatmap_loss * w_heatmap
                      + descriptor_loss * w_descriptor)
        outputs = {
            "junc_loss": junc_loss,
            "heatmap_loss": heatmap_loss,
            "w_junc": w_junc.item() \
                if isinstance(w_junc, nn.Parameter) else w_junc,
            "w_heatmap": w_heatmap.item() \
                if isinstance(w_heatmap, nn.Parameter) else w_heatmap,
            "descriptor_loss": descriptor_loss,
            "w_desc": w_descriptor.item() \
                if isinstance(w_descriptor, nn.Parameter) else w_descriptor
        }
        
        # Compute the regularization loss
        reg_loss = self.loss_funcs["reg_loss"](self.loss_weights)
        total_loss += reg_loss
        outputs.update({
            "reg_loss": reg_loss,
            "total_loss": total_loss
        })

        return outputs
