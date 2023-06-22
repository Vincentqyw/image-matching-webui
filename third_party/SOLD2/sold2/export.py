import numpy as np
import copy
import cv2
import h5py
import math
from tqdm import tqdm
import torch
from torch.nn.functional import pixel_shuffle, softmax
from torch.utils.data import DataLoader
from kornia.geometry import warp_perspective

from .dataset.dataset_util import get_dataset
from .model.model_util import get_model
from .misc.train_utils import get_latest_checkpoint
from .train import convert_junc_predictions
from .dataset.transforms.homographic_transforms import sample_homography


def restore_weights(model, state_dict):
    """ Restore weights in compatible mode. """
    # Try to directly load state dict
    try:
        model.load_state_dict(state_dict)
    except:
        err = model.load_state_dict(state_dict, strict=False)
        # missing keys are those in model but not in state_dict
        missing_keys = err.missing_keys
        # Unexpected keys are those in state_dict but not in model
        unexpected_keys = err.unexpected_keys

        # Load mismatched keys manually
        model_dict = model.state_dict()
        for idx, key in enumerate(missing_keys):
            dict_keys = [_ for _ in unexpected_keys if not "tracked" in _]
            model_dict[key] = state_dict[dict_keys[idx]]
        model.load_state_dict(model_dict)
    return model


def get_padded_filename(num_pad, idx):
    """ Get the filename padded with 0. """
    file_len = len("%d" % (idx))
    filename = "0" * (num_pad - file_len) + "%d" % (idx)
    return filename


def export_predictions(args, dataset_cfg, model_cfg, output_path,
                       export_dataset_mode):
    """ Export predictions. """
    # Get the test configuration
    test_cfg = model_cfg["test"]

    # Create the dataset and dataloader based on the export_dataset_mode
    print("\t Initializing dataset and dataloader")
    batch_size = 4
    export_dataset, collate_fn = get_dataset(export_dataset_mode, dataset_cfg)
    export_loader = DataLoader(export_dataset, batch_size=batch_size,
                               num_workers=test_cfg.get("num_workers", 4),
                               shuffle=False, pin_memory=False,
                               collate_fn=collate_fn)
    print("\t Successfully intialized dataset and dataloader.")

    # Initialize model and load the checkpoint
    model = get_model(model_cfg, mode="test")
    checkpoint = get_latest_checkpoint(args.resume_path, args.checkpoint_name)
    model = restore_weights(model, checkpoint["model_state_dict"])
    model = model.cuda()
    model.eval()
    print("\t Successfully initialized model")

    # Start the export process
    print("[Info] Start exporting predictions")
    output_dataset_path = output_path + ".h5"
    filename_idx = 0
    with h5py.File(output_dataset_path, "w", libver="latest", swmr=True) as f:
        # Iterate through all the data in dataloader
        for data in tqdm(export_loader, ascii=True):
            # Fetch the data
            junc_map = data["junction_map"]
            heatmap = data["heatmap"]
            valid_mask = data["valid_mask"]
            input_images = data["image"].cuda()

            # Run the forward pass
            with torch.no_grad():
                outputs = model(input_images)

            # Convert predictions
            junc_np = convert_junc_predictions(
                outputs["junctions"], model_cfg["grid_size"],
                model_cfg["detection_thresh"], 300)
            junc_map_np = junc_map.numpy().transpose(0, 2, 3, 1)
            heatmap_np = softmax(outputs["heatmap"].detach(),
                                 dim=1).cpu().numpy().transpose(0, 2, 3, 1)
            heatmap_gt_np = heatmap.numpy().transpose(0, 2, 3, 1)
            valid_mask_np = valid_mask.numpy().transpose(0, 2, 3, 1)

            # Data entries to save
            current_batch_size = input_images.shape[0]
            for batch_idx in range(current_batch_size):
                output_data = {
                    "image": input_images.cpu().numpy().transpose(0, 2, 3, 1)[batch_idx],
                    "junc_gt": junc_map_np[batch_idx],
                    "junc_pred": junc_np["junc_pred"][batch_idx],
                    "junc_pred_nms": junc_np["junc_pred_nms"][batch_idx].astype(np.float32),
                    "heatmap_gt": heatmap_gt_np[batch_idx],
                    "heatmap_pred": heatmap_np[batch_idx],
                    "valid_mask": valid_mask_np[batch_idx],
                    "junc_points": data["junctions"][batch_idx].numpy()[0].round().astype(np.int32),
                    "line_map": data["line_map"][batch_idx].numpy()[0].astype(np.int32)
                }

                # Save data to h5 dataset
                num_pad = math.ceil(math.log10(len(export_loader))) + 1
                output_key = get_padded_filename(num_pad, filename_idx)
                f_group = f.create_group(output_key)

                # Store data
                for key, output_data in output_data.items():
                    f_group.create_dataset(key, data=output_data,
                                           compression="gzip")
                filename_idx += 1


def export_homograpy_adaptation(args, dataset_cfg, model_cfg, output_path,
                                export_dataset_mode, device):
    """ Export homography adaptation results. """
    # Check if the export_dataset_mode is supported
    supported_modes = ["train", "test"]
    if not export_dataset_mode in supported_modes:
        raise ValueError(
            "[Error] The specified export_dataset_mode is not supported.")

    # Get the test configuration
    test_cfg = model_cfg["test"]

    # Get the homography adaptation configurations
    homography_cfg = dataset_cfg.get("homography_adaptation", None)
    if homography_cfg is None:
        raise ValueError(
            "[Error] Empty homography_adaptation entry in config.")

    # Create the dataset and dataloader based on the export_dataset_mode
    print("\t Initializing dataset and dataloader")
    batch_size = args.export_batch_size

    export_dataset, collate_fn = get_dataset(export_dataset_mode, dataset_cfg)
    export_loader = DataLoader(export_dataset, batch_size=batch_size,
                               num_workers=test_cfg.get("num_workers", 4),
                               shuffle=False, pin_memory=False,
                               collate_fn=collate_fn)
    print("\t Successfully intialized dataset and dataloader.")

    # Initialize model and load the checkpoint
    model = get_model(model_cfg, mode="test")
    checkpoint = get_latest_checkpoint(args.resume_path, args.checkpoint_name,
                                       device)
    model = restore_weights(model, checkpoint["model_state_dict"])
    model = model.to(device).eval()
    print("\t Successfully initialized model")

    # Start the export process
    print("[Info] Start exporting predictions")    
    output_dataset_path = output_path + ".h5"
    with h5py.File(output_dataset_path, "w", libver="latest") as f:
        f.swmr_mode=True
        for _, data in enumerate(tqdm(export_loader, ascii=True)):
            input_images = data["image"].to(device)
            file_keys = data["file_key"]
            batch_size = input_images.shape[0]
            
            # Run the homograpy adaptation
            outputs = homography_adaptation(input_images, model,
                                            model_cfg["grid_size"],
                                            homography_cfg)

            # Save the entries
            for batch_idx in range(batch_size):
                # Get the save key
                save_key = file_keys[batch_idx]
                output_data = {
                    "image": input_images.cpu().numpy().transpose(0, 2, 3, 1)[batch_idx],
                    "junc_prob_mean": outputs["junc_probs_mean"].cpu().numpy().transpose(0, 2, 3, 1)[batch_idx],
                    "junc_prob_max": outputs["junc_probs_max"].cpu().numpy().transpose(0, 2, 3, 1)[batch_idx],
                    "junc_count": outputs["junc_counts"].cpu().numpy().transpose(0, 2, 3, 1)[batch_idx],
                    "heatmap_prob_mean": outputs["heatmap_probs_mean"].cpu().numpy().transpose(0, 2, 3, 1)[batch_idx],
                    "heatmap_prob_max": outputs["heatmap_probs_max"].cpu().numpy().transpose(0, 2, 3, 1)[batch_idx],
                    "heatmap_cout": outputs["heatmap_counts"].cpu().numpy().transpose(0, 2, 3, 1)[batch_idx]
                }

                # Create group and write data
                f_group = f.create_group(save_key)
                for key, output_data in output_data.items():
                    f_group.create_dataset(key, data=output_data,
                                           compression="gzip")


def homography_adaptation(input_images, model, grid_size, homography_cfg):
    """ The homography adaptation process.
    Arguments:
        input_images: The images to be evaluated.
        model: The pytorch model in evaluation mode.
        grid_size: Grid size of the junction decoder.
        homography_cfg: Homography adaptation configurations.
    """
    # Get the device of the current model
    device = next(model.parameters()).device

    # Define some constants and placeholder
    batch_size, _, H, W = input_images.shape
    num_iter = homography_cfg["num_iter"]
    junc_probs = torch.zeros([batch_size, num_iter, H, W], device=device)
    junc_counts = torch.zeros([batch_size, 1, H, W], device=device)
    heatmap_probs = torch.zeros([batch_size, num_iter, H, W], device=device)
    heatmap_counts = torch.zeros([batch_size, 1, H, W], device=device)
    margin = homography_cfg["valid_border_margin"]

    # Keep a config with no artifacts
    homography_cfg_no_artifacts = copy.copy(homography_cfg["homographies"])
    homography_cfg_no_artifacts["allow_artifacts"] = False

    for idx in range(num_iter):
        if idx <= num_iter // 5:
            # Ensure that 20% of the homographies have no artifact
            H_mat_lst = [sample_homography(
                [H,W], **homography_cfg_no_artifacts)[0][None]
                         for _ in range(batch_size)]
        else:
            H_mat_lst = [sample_homography(
                [H,W], **homography_cfg["homographies"])[0][None]
                         for _ in range(batch_size)]

        H_mats = np.concatenate(H_mat_lst, axis=0)
        H_tensor = torch.tensor(H_mats, dtype=torch.float, device=device)
        H_inv_tensor = torch.inverse(H_tensor)

        # Perform the homography warp
        images_warped = warp_perspective(input_images, H_tensor, (H, W),
                                         flags="bilinear")
        
        # Warp the mask
        masks_junc_warped = warp_perspective(
            torch.ones([batch_size, 1, H, W], device=device),
            H_tensor, (H, W), flags="nearest")
        masks_heatmap_warped = warp_perspective(
            torch.ones([batch_size, 1, H, W], device=device),
            H_tensor, (H, W), flags="nearest")

        # Run the network forward pass
        with torch.no_grad():
            outputs = model(images_warped)
        
        # Unwarp and mask the junction prediction
        junc_prob_warped = pixel_shuffle(softmax(
            outputs["junctions"], dim=1)[:, :-1, :, :], grid_size)
        junc_prob = warp_perspective(junc_prob_warped, H_inv_tensor,
                                     (H, W), flags="bilinear")

        # Create the out of boundary mask
        out_boundary_mask = warp_perspective(
            torch.ones([batch_size, 1, H, W], device=device),
            H_inv_tensor, (H, W), flags="nearest")
        out_boundary_mask = adjust_border(out_boundary_mask, device, margin)

        junc_prob = junc_prob * out_boundary_mask
        junc_count = warp_perspective(masks_junc_warped * out_boundary_mask,
                                      H_inv_tensor, (H, W), flags="nearest")

        # Unwarp the mask and heatmap prediction
        # Always fetch only one channel
        if outputs["heatmap"].shape[1] == 2:
            # Convert to single channel directly from here
            heatmap_prob_warped = softmax(outputs["heatmap"],
                                          dim=1)[:, 1:, :, :]
        else:
            heatmap_prob_warped = torch.sigmoid(outputs["heatmap"])
        
        heatmap_prob_warped = heatmap_prob_warped * masks_heatmap_warped
        heatmap_prob = warp_perspective(heatmap_prob_warped, H_inv_tensor,
                                        (H, W), flags="bilinear")
        heatmap_count = warp_perspective(masks_heatmap_warped, H_inv_tensor,
                                         (H, W), flags="nearest")

        # Record the results
        junc_probs[:, idx:idx+1, :, :] = junc_prob
        heatmap_probs[:, idx:idx+1, :, :] = heatmap_prob
        junc_counts += junc_count
        heatmap_counts += heatmap_count

    # Perform the accumulation operation
    if homography_cfg["min_counts"] > 0:
        min_counts = homography_cfg["min_counts"]
        junc_count_mask = (junc_counts < min_counts)
        heatmap_count_mask = (heatmap_counts < min_counts)
        junc_counts[junc_count_mask] = 0
        heatmap_counts[heatmap_count_mask] = 0
    else:
        junc_count_mask = np.zeros_like(junc_counts, dtype=bool)
        heatmap_count_mask = np.zeros_like(heatmap_counts, dtype=bool)
    
    # Compute the mean accumulation
    junc_probs_mean = torch.sum(junc_probs, dim=1, keepdim=True) / junc_counts
    junc_probs_mean[junc_count_mask] = 0.
    heatmap_probs_mean = (torch.sum(heatmap_probs, dim=1, keepdim=True)
                          / heatmap_counts)
    heatmap_probs_mean[heatmap_count_mask] = 0.

    # Compute the max accumulation
    junc_probs_max = torch.max(junc_probs, dim=1, keepdim=True)[0]
    junc_probs_max[junc_count_mask] = 0.
    heatmap_probs_max = torch.max(heatmap_probs, dim=1, keepdim=True)[0]
    heatmap_probs_max[heatmap_count_mask] = 0.

    return {"junc_probs_mean": junc_probs_mean,
            "junc_probs_max": junc_probs_max,
            "junc_counts": junc_counts,
            "heatmap_probs_mean": heatmap_probs_mean,
            "heatmap_probs_max": heatmap_probs_max,
            "heatmap_counts": heatmap_counts}


def adjust_border(input_masks, device, margin=3):
    """ Adjust the border of the counts and valid_mask. """
    # Convert the mask to numpy array
    dtype = input_masks.dtype
    input_masks = np.squeeze(input_masks.cpu().numpy(), axis=1)

    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (margin*2, margin*2))
    batch_size = input_masks.shape[0]
    
    output_mask_lst = []
    # Erode all the masks
    for i in range(batch_size):
        output_mask = cv2.erode(input_masks[i, ...], erosion_kernel)

        output_mask_lst.append(
            torch.tensor(output_mask, dtype=dtype, device=device)[None])
    
    # Concat back along the batch dimension.
    output_masks = torch.cat(output_mask_lst, dim=0)
    return output_masks.unsqueeze(dim=1)
