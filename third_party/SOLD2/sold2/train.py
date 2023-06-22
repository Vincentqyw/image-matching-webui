"""
This file implements the training process and all the summaries
"""
import os
import numpy as np
import cv2
import torch
from torch.nn.functional import pixel_shuffle, softmax
from torch.utils.data import DataLoader
import torch.utils.data.dataloader as torch_loader
from tensorboardX import SummaryWriter

from .dataset.dataset_util import get_dataset
from .model.model_util import get_model
from .model.loss import TotalLoss, get_loss_and_weights
from .model.metrics import AverageMeter, Metrics, super_nms
from .model.lr_scheduler import get_lr_scheduler
from .misc.train_utils import (convert_image, get_latest_checkpoint,
                               remove_old_checkpoints)


def customized_collate_fn(batch):
    """ Customized collate_fn. """
    batch_keys = ["image", "junction_map", "heatmap", "valid_mask"]
    list_keys = ["junctions", "line_map"]

    outputs = {}
    for key in batch_keys:
        outputs[key] = torch_loader.default_collate([b[key] for b in batch])
    for key in list_keys:
        outputs[key] = [b[key] for b in batch]

    return outputs


def restore_weights(model, state_dict, strict=True):
    """ Restore weights in compatible mode. """
    # Try to directly load state dict
    try:
        model.load_state_dict(state_dict, strict=strict)
    # Deal with some version compatibility issue (catch version incompatible)
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


def train_net(args, dataset_cfg, model_cfg, output_path):
    """ Main training function. """
    # Add some version compatibility check
    if model_cfg.get("weighting_policy") is None:
        # Default to static
        model_cfg["weighting_policy"] = "static"

    # Get the train, val, test config
    train_cfg = model_cfg["train"]
    test_cfg = model_cfg["test"]

    # Create train and test dataset
    print("\t Initializing dataset...")
    train_dataset, train_collate_fn = get_dataset("train", dataset_cfg)
    test_dataset, test_collate_fn = get_dataset("test", dataset_cfg)

    # Create the dataloader
    train_loader = DataLoader(train_dataset,
                              batch_size=train_cfg["batch_size"],
                              num_workers=8,
                              shuffle=True, pin_memory=True,
                              collate_fn=train_collate_fn)
    test_loader = DataLoader(test_dataset,
                             batch_size=test_cfg.get("batch_size", 1),
                             num_workers=test_cfg.get("num_workers", 1),
                             shuffle=False, pin_memory=False,
                             collate_fn=test_collate_fn)
    print("\t Successfully intialized dataloaders.")


    # Get the loss function and weight first
    loss_funcs, loss_weights = get_loss_and_weights(model_cfg)

    # If resume.
    if args.resume:
        # Create model and load the state dict
        checkpoint = get_latest_checkpoint(args.resume_path,
                                           args.checkpoint_name)
        model = get_model(model_cfg, loss_weights)
        model = restore_weights(model, checkpoint["model_state_dict"])
        model = model.cuda()
        optimizer = torch.optim.Adam(
            [{"params": model.parameters(),
              "initial_lr": model_cfg["learning_rate"]}], 
            model_cfg["learning_rate"], 
            amsgrad=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Optionally get the learning rate scheduler
        scheduler = get_lr_scheduler(
            lr_decay=model_cfg.get("lr_decay", False),
            lr_decay_cfg=model_cfg.get("lr_decay_cfg", None),
            optimizer=optimizer)
        # If we start to use learning rate scheduler from the middle
        if ((scheduler is not None)
            and (checkpoint.get("scheduler_state_dict", None) is not None)):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    # Initialize all the components.
    else:
        # Create model and optimizer
        model = get_model(model_cfg, loss_weights)
        # Optionally get the pretrained wieghts
        if args.pretrained:
            print("\t [Debug] Loading pretrained weights...")
            checkpoint = get_latest_checkpoint(args.pretrained_path,
                                               args.checkpoint_name)
            # If auto weighting restore from non-auto weighting
            model = restore_weights(model, checkpoint["model_state_dict"],
                                    strict=False)
            print("\t [Debug] Finished loading pretrained weights!")
        
        model = model.cuda()
        optimizer = torch.optim.Adam(
            [{"params": model.parameters(),
              "initial_lr": model_cfg["learning_rate"]}], 
            model_cfg["learning_rate"], 
            amsgrad=True)
        # Optionally get the learning rate scheduler
        scheduler = get_lr_scheduler(
            lr_decay=model_cfg.get("lr_decay", False),
            lr_decay_cfg=model_cfg.get("lr_decay_cfg", None),
            optimizer=optimizer)
        start_epoch = 0
    
    print("\t Successfully initialized model")

    # Define the total loss
    policy = model_cfg.get("weighting_policy", "static")
    loss_func = TotalLoss(loss_funcs, loss_weights, policy).cuda()
    if "descriptor_decoder" in model_cfg:
        metric_func = Metrics(model_cfg["detection_thresh"],
                              model_cfg["prob_thresh"],
                              model_cfg["descriptor_loss_cfg"]["grid_size"],
                              desc_metric_lst='all')
    else:
        metric_func = Metrics(model_cfg["detection_thresh"],
                              model_cfg["prob_thresh"],
                              model_cfg["grid_size"])

    # Define the summary writer
    logdir = os.path.join(output_path, "log")
    writer = SummaryWriter(logdir=logdir)

    # Start the training loop
    for epoch in range(start_epoch, model_cfg["epochs"]):
        # Record the learning rate
        current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        writer.add_scalar("LR/lr", current_lr, epoch)

        # Train for one epochs
        print("\n\n================== Training ====================")
        train_single_epoch(
            model=model,
            model_cfg=model_cfg,
            optimizer=optimizer,
            loss_func=loss_func,
            metric_func=metric_func,
            train_loader=train_loader,
            writer=writer,
            epoch=epoch)

        # Do the validation
        print("\n\n================== Validation ==================")
        validate(
            model=model,
            model_cfg=model_cfg,
            loss_func=loss_func,
            metric_func=metric_func,
            val_loader=test_loader,
            writer=writer,
            epoch=epoch)

        # Update the scheduler
        if scheduler is not None:
            scheduler.step()

        # Save checkpoints
        file_name = os.path.join(output_path,
                                 "checkpoint-epoch%03d-end.tar"%(epoch))
        print("[Info] Saving checkpoint %s ..." % file_name)
        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_cfg": model_cfg}
        if scheduler is not None:
            save_dict.update({"scheduler_state_dict": scheduler.state_dict()})
        torch.save(save_dict, file_name)

        # Remove the outdated checkpoints
        remove_old_checkpoints(output_path, model_cfg.get("max_ckpt", 15))


def train_single_epoch(model, model_cfg, optimizer, loss_func, metric_func,
                       train_loader, writer, epoch):
    """ Train for one epoch. """
    # Switch the model to training mode
    model.train()

    # Initialize the average meter
    compute_descriptors = loss_func.compute_descriptors
    if compute_descriptors:
        average_meter = AverageMeter(is_training=True, desc_metric_lst='all')
    else:
        average_meter = AverageMeter(is_training=True)

    # The training loop
    for idx, data in enumerate(train_loader):
        if compute_descriptors:
            junc_map = data["ref_junction_map"].cuda()
            junc_map2 = data["target_junction_map"].cuda()
            heatmap = data["ref_heatmap"].cuda()
            heatmap2 = data["target_heatmap"].cuda()
            line_points = data["ref_line_points"].cuda()
            line_points2 = data["target_line_points"].cuda()
            line_indices = data["ref_line_indices"].cuda()
            valid_mask = data["ref_valid_mask"].cuda()
            valid_mask2 = data["target_valid_mask"].cuda()
            input_images = data["ref_image"].cuda()
            input_images2 = data["target_image"].cuda()

            # Run the forward pass
            outputs = model(input_images)
            outputs2 = model(input_images2)

            # Compute losses
            losses = loss_func.forward_descriptors(
                outputs["junctions"], outputs2["junctions"],
                junc_map, junc_map2, outputs["heatmap"], outputs2["heatmap"],
                heatmap, heatmap2, line_points, line_points2,
                line_indices, outputs['descriptors'], outputs2['descriptors'],
                epoch, valid_mask, valid_mask2)
        else:
            junc_map = data["junction_map"].cuda()
            heatmap = data["heatmap"].cuda()
            valid_mask = data["valid_mask"].cuda()
            input_images = data["image"].cuda()

            # Run the forward pass
            outputs = model(input_images)

            # Compute losses
            losses = loss_func(
                outputs["junctions"], junc_map,
                outputs["heatmap"], heatmap,
                valid_mask)
        
        total_loss = losses["total_loss"]

        # Update the model
        optimizer.zero_grad()
        total_loss.backward()                     
        optimizer.step()

        # Compute the global step
        global_step = epoch * len(train_loader) + idx
        ############## Measure the metric error #########################
        # Only do this when needed
        if (((idx % model_cfg["disp_freq"]) == 0)
            or ((idx % model_cfg["summary_freq"]) == 0)):
            junc_np = convert_junc_predictions(
                outputs["junctions"], model_cfg["grid_size"],
                model_cfg["detection_thresh"], 300)
            junc_map_np = junc_map.cpu().numpy().transpose(0, 2, 3, 1)

            # Always fetch only one channel (compatible with L1, L2, and CE)
            if outputs["heatmap"].shape[1] == 2:
                heatmap_np = softmax(outputs["heatmap"].detach(),
                                     dim=1).cpu().numpy()
                heatmap_np = heatmap_np.transpose(0, 2, 3, 1)[:, :, :, 1:]
            else:
                heatmap_np = torch.sigmoid(outputs["heatmap"].detach())
                heatmap_np = heatmap_np.cpu().numpy().transpose(0, 2, 3, 1)
            
            heatmap_gt_np = heatmap.cpu().numpy().transpose(0, 2, 3, 1)
            valid_mask_np = valid_mask.cpu().numpy().transpose(0, 2, 3, 1)

            # Evaluate metric results
            if compute_descriptors:
                metric_func.evaluate(
                    junc_np["junc_pred"], junc_np["junc_pred_nms"],
                    junc_map_np, heatmap_np, heatmap_gt_np, valid_mask_np,
                    line_points, line_points2, outputs["descriptors"],
                    outputs2["descriptors"], line_indices)
            else:
                metric_func.evaluate(
                    junc_np["junc_pred"], junc_np["junc_pred_nms"],
                    junc_map_np, heatmap_np, heatmap_gt_np, valid_mask_np)
            # Update average meter
            junc_loss = losses["junc_loss"].item()
            heatmap_loss = losses["heatmap_loss"].item()
            loss_dict = {
                "junc_loss": junc_loss,
                "heatmap_loss": heatmap_loss,
                "total_loss": total_loss.item()}
            if compute_descriptors:
                descriptor_loss = losses["descriptor_loss"].item()
                loss_dict["descriptor_loss"] = losses["descriptor_loss"].item()

            average_meter.update(metric_func, loss_dict, num_samples=junc_map.shape[0])

        # Display the progress
        if (idx % model_cfg["disp_freq"]) == 0:
            results = metric_func.metric_results
            average = average_meter.average()
            # Get gpu memory usage in GB
            gpu_mem_usage = torch.cuda.max_memory_allocated() / (1024 ** 3)
            if compute_descriptors:
                print("Epoch [%d / %d] Iter [%d / %d] loss=%.4f (%.4f), junc_loss=%.4f (%.4f), heatmap_loss=%.4f (%.4f), descriptor_loss=%.4f (%.4f), gpu_mem=%.4fGB"
                      % (epoch, model_cfg["epochs"], idx, len(train_loader),
                         total_loss.item(), average["total_loss"], junc_loss,
                         average["junc_loss"], heatmap_loss,
                         average["heatmap_loss"], descriptor_loss,
                         average["descriptor_loss"], gpu_mem_usage))
            else:
                print("Epoch [%d / %d] Iter [%d / %d] loss=%.4f (%.4f), junc_loss=%.4f (%.4f), heatmap_loss=%.4f (%.4f), gpu_mem=%.4fGB"
                      % (epoch, model_cfg["epochs"], idx, len(train_loader),
                         total_loss.item(), average["total_loss"],
                         junc_loss, average["junc_loss"], heatmap_loss,
                         average["heatmap_loss"], gpu_mem_usage))
            print("\t Junction     precision=%.4f (%.4f) / recall=%.4f (%.4f)"
                  % (results["junc_precision"], average["junc_precision"],
                     results["junc_recall"], average["junc_recall"]))
            print("\t Junction nms precision=%.4f (%.4f) / recall=%.4f (%.4f)"
                  % (results["junc_precision_nms"],
                     average["junc_precision_nms"],
                     results["junc_recall_nms"], average["junc_recall_nms"]))
            print("\t Heatmap      precision=%.4f (%.4f) / recall=%.4f (%.4f)"
                  %(results["heatmap_precision"],
                    average["heatmap_precision"],
                    results["heatmap_recall"], average["heatmap_recall"]))
            if compute_descriptors:
                print("\t Descriptors  matching score=%.4f (%.4f)"
                      %(results["matching_score"], average["matching_score"]))

        # Record summaries
        if (idx % model_cfg["summary_freq"]) == 0:
            results = metric_func.metric_results
            average = average_meter.average()
            # Add the shared losses
            scalar_summaries = {
                "junc_loss": junc_loss,
                "heatmap_loss": heatmap_loss,
                "total_loss": total_loss.detach().cpu().numpy(),
                "metrics": results,
                "average": average}
            # Add descriptor terms
            if compute_descriptors:
                scalar_summaries["descriptor_loss"] = descriptor_loss
                scalar_summaries["w_desc"] = losses["w_desc"]

            # Add weighting terms (even for static terms)
            scalar_summaries["w_junc"] = losses["w_junc"]
            scalar_summaries["w_heatmap"] = losses["w_heatmap"]
            scalar_summaries["reg_loss"] = losses["reg_loss"].item()

            num_images = 3
            junc_pred_binary = (junc_np["junc_pred"][:num_images, ...]
                                > model_cfg["detection_thresh"])
            junc_pred_nms_binary = (junc_np["junc_pred_nms"][:num_images, ...]
                                    > model_cfg["detection_thresh"])
            image_summaries = {
                "image": input_images.cpu().numpy()[:num_images, ...],
                "valid_mask": valid_mask_np[:num_images, ...],
                "junc_map_pred": junc_pred_binary,
                "junc_map_pred_nms": junc_pred_nms_binary,
                "junc_map_gt": junc_map_np[:num_images, ...],
                "junc_prob_map": junc_np["junc_prob"][:num_images, ...],
                "heatmap_pred": heatmap_np[:num_images, ...],
                "heatmap_gt": heatmap_gt_np[:num_images, ...]}
            # Record the training summary
            record_train_summaries(
                writer, global_step, scalars=scalar_summaries,
                images=image_summaries)


def validate(model, model_cfg, loss_func, metric_func, val_loader, writer, epoch):
    """ Validation. """
    # Switch the model to eval mode
    model.eval()

    # Initialize the average meter
    compute_descriptors = loss_func.compute_descriptors
    if compute_descriptors:
        average_meter = AverageMeter(is_training=True, desc_metric_lst='all')
    else:
        average_meter = AverageMeter(is_training=True)

    # The validation loop
    for idx, data in enumerate(val_loader):
        if compute_descriptors:
            junc_map = data["ref_junction_map"].cuda()
            junc_map2 = data["target_junction_map"].cuda()
            heatmap = data["ref_heatmap"].cuda()
            heatmap2 = data["target_heatmap"].cuda()
            line_points = data["ref_line_points"].cuda()
            line_points2 = data["target_line_points"].cuda()
            line_indices = data["ref_line_indices"].cuda()
            valid_mask = data["ref_valid_mask"].cuda()
            valid_mask2 = data["target_valid_mask"].cuda()
            input_images = data["ref_image"].cuda()
            input_images2 = data["target_image"].cuda()

            # Run the forward pass
            with torch.no_grad():
                outputs = model(input_images)
                outputs2 = model(input_images2)

                # Compute losses
                losses = loss_func.forward_descriptors(
                    outputs["junctions"], outputs2["junctions"],
                    junc_map, junc_map2, outputs["heatmap"],
                    outputs2["heatmap"], heatmap, heatmap2, line_points,
                    line_points2, line_indices, outputs['descriptors'],
                    outputs2['descriptors'], epoch, valid_mask, valid_mask2)
        else:
            junc_map = data["junction_map"].cuda()
            heatmap = data["heatmap"].cuda()
            valid_mask = data["valid_mask"].cuda()
            input_images = data["image"].cuda()

            # Run the forward pass
            with torch.no_grad():
                outputs = model(input_images)

                # Compute losses
                losses = loss_func(
                    outputs["junctions"], junc_map,
                    outputs["heatmap"], heatmap,
                    valid_mask)
        total_loss = losses["total_loss"]

        ############## Measure the metric error #########################
        junc_np = convert_junc_predictions(
            outputs["junctions"], model_cfg["grid_size"],
            model_cfg["detection_thresh"], 300)
        junc_map_np = junc_map.cpu().numpy().transpose(0, 2, 3, 1)
        # Always fetch only one channel (compatible with L1, L2, and CE)
        if outputs["heatmap"].shape[1] == 2:
            heatmap_np = softmax(outputs["heatmap"].detach(),
                                 dim=1).cpu().numpy().transpose(0, 2, 3, 1)
            heatmap_np = heatmap_np[:, :, :, 1:]
        else:
            heatmap_np = torch.sigmoid(outputs["heatmap"].detach())
            heatmap_np = heatmap_np.cpu().numpy().transpose(0, 2, 3, 1)


        heatmap_gt_np = heatmap.cpu().numpy().transpose(0, 2, 3, 1)
        valid_mask_np = valid_mask.cpu().numpy().transpose(0, 2, 3, 1)

        # Evaluate metric results
        if compute_descriptors:
            metric_func.evaluate(
                junc_np["junc_pred"], junc_np["junc_pred_nms"],
                junc_map_np, heatmap_np, heatmap_gt_np, valid_mask_np,
                line_points, line_points2, outputs["descriptors"],
                outputs2["descriptors"], line_indices)
        else:
            metric_func.evaluate(
                junc_np["junc_pred"], junc_np["junc_pred_nms"], junc_map_np,
                heatmap_np, heatmap_gt_np, valid_mask_np)
        # Update average meter
        junc_loss = losses["junc_loss"].item()
        heatmap_loss = losses["heatmap_loss"].item()
        loss_dict = {
            "junc_loss": junc_loss,
            "heatmap_loss": heatmap_loss,
            "total_loss": total_loss.item()}
        if compute_descriptors:
            descriptor_loss = losses["descriptor_loss"].item()
            loss_dict["descriptor_loss"] = losses["descriptor_loss"].item()
        average_meter.update(metric_func, loss_dict, num_samples=junc_map.shape[0])

        # Display the progress
        if (idx % model_cfg["disp_freq"]) == 0:
            results = metric_func.metric_results
            average = average_meter.average()
            if compute_descriptors:
                print("Iter [%d / %d] loss=%.4f (%.4f), junc_loss=%.4f (%.4f), heatmap_loss=%.4f (%.4f), descriptor_loss=%.4f (%.4f)"
                      % (idx, len(val_loader),
                         total_loss.item(), average["total_loss"],
                         junc_loss, average["junc_loss"],
                         heatmap_loss, average["heatmap_loss"],
                         descriptor_loss, average["descriptor_loss"]))
            else:
                print("Iter [%d / %d] loss=%.4f (%.4f), junc_loss=%.4f (%.4f), heatmap_loss=%.4f (%.4f)"
                      % (idx, len(val_loader),
                         total_loss.item(), average["total_loss"],
                         junc_loss, average["junc_loss"],
                         heatmap_loss, average["heatmap_loss"]))
            print("\t Junction     precision=%.4f (%.4f) / recall=%.4f (%.4f)"
                  % (results["junc_precision"], average["junc_precision"],
                     results["junc_recall"], average["junc_recall"]))
            print("\t Junction nms precision=%.4f (%.4f) / recall=%.4f (%.4f)"
                  % (results["junc_precision_nms"],
                     average["junc_precision_nms"],
                     results["junc_recall_nms"], average["junc_recall_nms"]))
            print("\t Heatmap      precision=%.4f (%.4f) / recall=%.4f (%.4f)"
                  % (results["heatmap_precision"],
                     average["heatmap_precision"],
                     results["heatmap_recall"], average["heatmap_recall"]))
            if compute_descriptors:
                print("\t Descriptors  matching score=%.4f (%.4f)"
                      %(results["matching_score"], average["matching_score"]))

    # Record summaries
    average = average_meter.average()
    scalar_summaries = {"average": average}
    # Record the training summary
    record_test_summaries(writer, epoch, scalar_summaries)


def convert_junc_predictions(predictions, grid_size,
                             detect_thresh=1/65, topk=300):
    """ Convert torch predictions to numpy arrays for evaluation. """
    # Convert to probability outputs first
    junc_prob = softmax(predictions.detach(), dim=1).cpu()
    junc_pred = junc_prob[:, :-1, :, :]

    junc_prob_np = junc_prob.numpy().transpose(0, 2, 3, 1)[:, :, :, :-1]
    junc_prob_np = np.sum(junc_prob_np, axis=-1)
    junc_pred_np = pixel_shuffle(
        junc_pred, grid_size).cpu().numpy().transpose(0, 2, 3, 1)
    junc_pred_np_nms = super_nms(junc_pred_np, grid_size, detect_thresh, topk)
    junc_pred_np = junc_pred_np.squeeze(-1)

    return {"junc_pred": junc_pred_np, "junc_pred_nms": junc_pred_np_nms,
            "junc_prob": junc_prob_np}


def record_train_summaries(writer, global_step, scalars, images):
    """ Record training summaries. """
    # Record the scalar summaries
    results = scalars["metrics"]
    average = scalars["average"]

    # GPU memory part
    # Get gpu memory usage in GB
    gpu_mem_usage = torch.cuda.max_memory_allocated() / (1024 ** 3)
    writer.add_scalar("GPU/GPU_memory_usage", gpu_mem_usage, global_step)

    # Loss part
    writer.add_scalar("Train_loss/junc_loss", scalars["junc_loss"],
                      global_step)
    writer.add_scalar("Train_loss/heatmap_loss", scalars["heatmap_loss"],
                      global_step)
    writer.add_scalar("Train_loss/total_loss", scalars["total_loss"],
                      global_step)
    # Add regularization loss
    if "reg_loss" in scalars.keys():
        writer.add_scalar("Train_loss/reg_loss", scalars["reg_loss"],
                          global_step)
    # Add descriptor loss
    if "descriptor_loss" in scalars.keys():
        key = "descriptor_loss"
        writer.add_scalar("Train_loss/%s"%(key), scalars[key], global_step)
        writer.add_scalar("Train_loss_average/%s"%(key), average[key],
                          global_step)
    
    # Record weighting
    for key in scalars.keys():
        if "w_" in key:
            writer.add_scalar("Train_weight/%s"%(key), scalars[key],
                              global_step)
    
    # Smoothed loss
    writer.add_scalar("Train_loss_average/junc_loss", average["junc_loss"],
                      global_step)
    writer.add_scalar("Train_loss_average/heatmap_loss",
                      average["heatmap_loss"], global_step)
    writer.add_scalar("Train_loss_average/total_loss", average["total_loss"],
                      global_step)
    # Add smoothed descriptor loss
    if "descriptor_loss" in average.keys():
        writer.add_scalar("Train_loss_average/descriptor_loss",
                          average["descriptor_loss"], global_step)

    # Metrics part
    writer.add_scalar("Train_metrics/junc_precision",
                      results["junc_precision"], global_step)
    writer.add_scalar("Train_metrics/junc_precision_nms",
                      results["junc_precision_nms"], global_step)
    writer.add_scalar("Train_metrics/junc_recall",
                      results["junc_recall"], global_step)
    writer.add_scalar("Train_metrics/junc_recall_nms",
                      results["junc_recall_nms"], global_step)
    writer.add_scalar("Train_metrics/heatmap_precision",
                      results["heatmap_precision"], global_step)
    writer.add_scalar("Train_metrics/heatmap_recall",
                      results["heatmap_recall"], global_step)
    # Add descriptor metric
    if "matching_score" in results.keys():
        writer.add_scalar("Train_metrics/matching_score",
                          results["matching_score"], global_step)

    # Average part
    writer.add_scalar("Train_metrics_average/junc_precision",
                      average["junc_precision"], global_step)
    writer.add_scalar("Train_metrics_average/junc_precision_nms",
                      average["junc_precision_nms"], global_step)
    writer.add_scalar("Train_metrics_average/junc_recall",
                      average["junc_recall"], global_step)
    writer.add_scalar("Train_metrics_average/junc_recall_nms",
                      average["junc_recall_nms"], global_step)
    writer.add_scalar("Train_metrics_average/heatmap_precision",
                      average["heatmap_precision"], global_step)
    writer.add_scalar("Train_metrics_average/heatmap_recall",
                      average["heatmap_recall"], global_step)
    # Add smoothed descriptor metric
    if "matching_score" in average.keys():
        writer.add_scalar("Train_metrics_average/matching_score",
                          average["matching_score"], global_step)

    # Record the image summary
    # Image part
    image_tensor = convert_image(images["image"], 1)
    valid_masks = convert_image(images["valid_mask"], -1)
    writer.add_images("Train/images", image_tensor, global_step,
                      dataformats="NCHW")
    writer.add_images("Train/valid_map", valid_masks, global_step,
                      dataformats="NHWC")

    # Heatmap part
    writer.add_images("Train/heatmap_gt",
                      convert_image(images["heatmap_gt"], -1), global_step,
                      dataformats="NHWC")
    writer.add_images("Train/heatmap_pred",
                      convert_image(images["heatmap_pred"], -1), global_step,
                      dataformats="NHWC")

    # Junction prediction part
    junc_plots = plot_junction_detection(
        image_tensor, images["junc_map_pred"],
        images["junc_map_pred_nms"], images["junc_map_gt"])
    writer.add_images("Train/junc_gt", junc_plots["junc_gt_plot"] / 255.,
                      global_step, dataformats="NHWC")
    writer.add_images("Train/junc_pred", junc_plots["junc_pred_plot"] / 255.,
                      global_step, dataformats="NHWC")
    writer.add_images("Train/junc_pred_nms",
                      junc_plots["junc_pred_nms_plot"] / 255., global_step,
                      dataformats="NHWC")
    writer.add_images(
        "Train/junc_prob_map",
        convert_image(images["junc_prob_map"][..., None], axis=-1),
        global_step, dataformats="NHWC")


def record_test_summaries(writer, epoch, scalars):
    """ Record testing summaries. """
    average = scalars["average"]

    # Average loss
    writer.add_scalar("Val_loss/junc_loss", average["junc_loss"], epoch)
    writer.add_scalar("Val_loss/heatmap_loss", average["heatmap_loss"], epoch)
    writer.add_scalar("Val_loss/total_loss", average["total_loss"], epoch)
    # Add descriptor loss
    if "descriptor_loss" in average.keys():
        key = "descriptor_loss"
        writer.add_scalar("Val_loss/%s"%(key), average[key], epoch)

    # Average metrics
    writer.add_scalar("Val_metrics/junc_precision", average["junc_precision"],
                      epoch)
    writer.add_scalar("Val_metrics/junc_precision_nms",
                      average["junc_precision_nms"], epoch)
    writer.add_scalar("Val_metrics/junc_recall",
                      average["junc_recall"], epoch)
    writer.add_scalar("Val_metrics/junc_recall_nms",
                      average["junc_recall_nms"], epoch)
    writer.add_scalar("Val_metrics/heatmap_precision",
                      average["heatmap_precision"], epoch)
    writer.add_scalar("Val_metrics/heatmap_recall",
                      average["heatmap_recall"], epoch)
    # Add descriptor metric
    if "matching_score" in average.keys():
        writer.add_scalar("Val_metrics/matching_score",
                          average["matching_score"], epoch)


def plot_junction_detection(image_tensor, junc_pred_tensor,
                            junc_pred_nms_tensor, junc_gt_tensor):
    """ Plot the junction points on images. """
    # Get the batch_size
    batch_size = image_tensor.shape[0]

    # Process through batch dimension
    junc_pred_lst = []
    junc_pred_nms_lst = []
    junc_gt_lst = []
    for i in range(batch_size):
        # Convert image to 255 uint8
        image = (image_tensor[i, :, :, :]
                 * 255.).astype(np.uint8).transpose(1,2,0)

        # Plot groundtruth onto image
        junc_gt = junc_gt_tensor[i, ...]
        coord_gt = np.where(junc_gt.squeeze() > 0)
        points_gt = np.concatenate((coord_gt[0][..., None],
                                    coord_gt[1][..., None]),
                                    axis=1)
        plot_gt = image.copy()
        for id in range(points_gt.shape[0]):
            cv2.circle(plot_gt, tuple(np.flip(points_gt[id, :])), 3,
                       color=(255, 0, 0), thickness=2)
        junc_gt_lst.append(plot_gt[None, ...])

        # Plot junc_pred
        junc_pred = junc_pred_tensor[i, ...]
        coord_pred = np.where(junc_pred > 0)
        points_pred = np.concatenate((coord_pred[0][..., None],
                                      coord_pred[1][..., None]),
                                      axis=1)
        plot_pred = image.copy()
        for id in range(points_pred.shape[0]):
            cv2.circle(plot_pred, tuple(np.flip(points_pred[id, :])), 3,
                       color=(0, 255, 0), thickness=2)
        junc_pred_lst.append(plot_pred[None, ...])

        # Plot junc_pred_nms
        junc_pred_nms = junc_pred_nms_tensor[i, ...]
        coord_pred_nms = np.where(junc_pred_nms > 0)
        points_pred_nms = np.concatenate((coord_pred_nms[0][..., None],
                                          coord_pred_nms[1][..., None]),
                                          axis=1)
        plot_pred_nms = image.copy()
        for id in range(points_pred_nms.shape[0]):
            cv2.circle(plot_pred_nms, tuple(np.flip(points_pred_nms[id, :])),
                       3, color=(0, 255, 0), thickness=2)
        junc_pred_nms_lst.append(plot_pred_nms[None, ...])

    return {"junc_gt_plot": np.concatenate(junc_gt_lst, axis=0),
            "junc_pred_plot": np.concatenate(junc_pred_lst, axis=0),
            "junc_pred_nms_plot": np.concatenate(junc_pred_nms_lst, axis=0)}
