"""
Implements the full pipeline from raw images to line matches.
"""
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import softmax

from .model_util import get_model
from .loss import get_loss_and_weights
from .metrics import super_nms
from .line_detection import LineSegmentDetectionModule
from .line_matching import WunschLineMatcher
from ..train import convert_junc_predictions
from ..misc.train_utils import adapt_checkpoint
from .line_detector import line_map_to_segments


class LineMatcher(object):
    """ Full line matcher including line detection and matching
        with the Needleman-Wunsch algorithm. """
    def __init__(self, model_cfg, ckpt_path, device, line_detector_cfg,
                 line_matcher_cfg, multiscale=False, scales=[1., 2.]):
        # Get loss weights if dynamic weighting
        _, loss_weights = get_loss_and_weights(model_cfg, device)
        self.device = device
        
        # Initialize the cnn backbone
        self.model = get_model(model_cfg, loss_weights)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        checkpoint = adapt_checkpoint(checkpoint["model_state_dict"])
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        self.grid_size = model_cfg["grid_size"]
        self.junc_detect_thresh = model_cfg["detection_thresh"]
        self.max_num_junctions = model_cfg.get("max_num_junctions", 300)

        # Initialize the line detector
        self.line_detector = LineSegmentDetectionModule(**line_detector_cfg)
        self.multiscale = multiscale
        self.scales = scales

        # Initialize the line matcher
        self.line_matcher = WunschLineMatcher(**line_matcher_cfg)
        
        # Print some debug messages
        for key, val in line_detector_cfg.items():
            print(f"[Debug] {key}: {val}")
        # print("[Debug] detect_thresh: %f" % (line_detector_cfg["detect_thresh"]))
        # print("[Debug] num_samples: %d" % (line_detector_cfg["num_samples"]))
        


    # Perform line detection and descriptor inference on a single image
    def line_detection(self, input_image, valid_mask=None,
                       desc_only=False, profile=False):
        # Restrict input_image to 4D torch tensor
        if ((not len(input_image.shape) == 4)
            or (not isinstance(input_image, torch.Tensor))):
            raise ValueError(
                "[Error] the input image should be a 4D torch tensor")

        # Move the input to corresponding device
        input_image = input_image.to(self.device)

        # Forward of the CNN backbone
        start_time = time.time()
        with torch.no_grad():
            net_outputs = self.model(input_image)

        outputs = {"descriptor": net_outputs["descriptors"]}

        if not desc_only:
            junc_np = convert_junc_predictions(
                net_outputs["junctions"], self.grid_size,
                self.junc_detect_thresh, self.max_num_junctions)
            if valid_mask is None:
                junctions = np.where(junc_np["junc_pred_nms"].squeeze())
            else:
                junctions = np.where(
                    junc_np["junc_pred_nms"].squeeze() * valid_mask)
            junctions = np.concatenate([junctions[0][..., None],
                                        junctions[1][..., None]], axis=-1)

            if net_outputs["heatmap"].shape[1] == 2:
                # Convert to single channel directly from here
                heatmap = softmax(
                    net_outputs["heatmap"],
                    dim=1)[:, 1:, :, :].cpu().numpy().transpose(0, 2, 3, 1)
            else:
                heatmap = torch.sigmoid(
                    net_outputs["heatmap"]).cpu().numpy().transpose(0, 2, 3, 1)
            heatmap = heatmap[0, :, :, 0]

            # Run the line detector.
            line_map, junctions, heatmap = self.line_detector.detect(
                junctions, heatmap, device=self.device)
            if isinstance(line_map, torch.Tensor):
                line_map = line_map.cpu().numpy()
            if isinstance(junctions, torch.Tensor):
                junctions = junctions.cpu().numpy()
            outputs["heatmap"] = heatmap.cpu().numpy()
            outputs["junctions"] = junctions

            # If it's a line map with multiple detect_thresh and inlier_thresh
            if len(line_map.shape) > 2:
                num_detect_thresh = line_map.shape[0]
                num_inlier_thresh = line_map.shape[1]
                line_segments = []
                for detect_idx in range(num_detect_thresh):
                    line_segments_inlier = []
                    for inlier_idx in range(num_inlier_thresh):
                        line_map_tmp = line_map[detect_idx, inlier_idx, :, :]
                        line_segments_tmp = line_map_to_segments(junctions, line_map_tmp)
                        line_segments_inlier.append(line_segments_tmp)
                    line_segments.append(line_segments_inlier)
            else:
                line_segments = line_map_to_segments(junctions, line_map)

            outputs["line_segments"] = line_segments

        end_time = time.time()

        if profile:
            outputs["time"] = end_time - start_time
        
        return outputs

    # Perform line detection and descriptor inference at multiple scales
    def multiscale_line_detection(self, input_image, valid_mask=None,
                                  desc_only=False, profile=False,
                                  scales=[1., 2.], aggregation='mean'):
        # Restrict input_image to 4D torch tensor
        if ((not len(input_image.shape) == 4)
            or (not isinstance(input_image, torch.Tensor))):
            raise ValueError(
                "[Error] the input image should be a 4D torch tensor")

        # Move the input to corresponding device
        input_image = input_image.to(self.device)
        img_size = input_image.shape[2:4]
        desc_size = tuple(np.array(img_size) // 4)

        # Run the inference at multiple image scales
        start_time = time.time()
        junctions, heatmaps, descriptors = [], [], []
        for s in scales:
            # Resize the image
            resized_img = F.interpolate(input_image, scale_factor=s,
                                        mode='bilinear')

            # Forward of the CNN backbone
            with torch.no_grad():
                net_outputs = self.model(resized_img)

            descriptors.append(F.interpolate(
                net_outputs["descriptors"], size=desc_size, mode="bilinear"))

            if not desc_only:
                junc_prob = convert_junc_predictions(
                    net_outputs["junctions"], self.grid_size)["junc_pred"]
                junctions.append(cv2.resize(junc_prob.squeeze(),
                                 (img_size[1], img_size[0]),
                                 interpolation=cv2.INTER_LINEAR))

                if net_outputs["heatmap"].shape[1] == 2:
                    # Convert to single channel directly from here
                    heatmap = softmax(net_outputs["heatmap"],
                                      dim=1)[:, 1:, :, :]
                else:
                    heatmap = torch.sigmoid(net_outputs["heatmap"])
                heatmaps.append(F.interpolate(heatmap, size=img_size,
                                              mode="bilinear"))

        # Aggregate the results
        if aggregation == 'mean':
            # Aggregation through the mean activation
            descriptors = torch.stack(descriptors, dim=0).mean(0)
        else:
            # Aggregation through the max activation
            descriptors = torch.stack(descriptors, dim=0).max(0)[0]
        outputs = {"descriptor": descriptors}

        if not desc_only:
            if aggregation == 'mean':
                junctions = np.stack(junctions, axis=0).mean(0)[None]
                heatmap = torch.stack(heatmaps, dim=0).mean(0)[0, 0, :, :]
                heatmap = heatmap.cpu().numpy()
            else:
                junctions = np.stack(junctions, axis=0).max(0)[None]
                heatmap = torch.stack(heatmaps, dim=0).max(0)[0][0, 0, :, :]
                heatmap = heatmap.cpu().numpy()

            # Extract junctions
            junc_pred_nms = super_nms(
                junctions[..., None], self.grid_size,
                self.junc_detect_thresh, self.max_num_junctions)
            if valid_mask is None:
                junctions = np.where(junc_pred_nms.squeeze())
            else:
                junctions = np.where(junc_pred_nms.squeeze() * valid_mask)
            junctions = np.concatenate([junctions[0][..., None],
                                        junctions[1][..., None]], axis=-1)

            # Run the line detector.
            line_map, junctions, heatmap = self.line_detector.detect(
                junctions, heatmap, device=self.device)
            if isinstance(line_map, torch.Tensor):
                line_map = line_map.cpu().numpy()
            if isinstance(junctions, torch.Tensor):
                junctions = junctions.cpu().numpy()
            outputs["heatmap"] = heatmap.cpu().numpy()
            outputs["junctions"] = junctions

            # If it's a line map with multiple detect_thresh and inlier_thresh
            if len(line_map.shape) > 2:
                num_detect_thresh = line_map.shape[0]
                num_inlier_thresh = line_map.shape[1]
                line_segments = []
                for detect_idx in range(num_detect_thresh):
                    line_segments_inlier = []
                    for inlier_idx in range(num_inlier_thresh):
                        line_map_tmp = line_map[detect_idx, inlier_idx, :, :]
                        line_segments_tmp = line_map_to_segments(
                            junctions, line_map_tmp)
                        line_segments_inlier.append(line_segments_tmp)
                    line_segments.append(line_segments_inlier)
            else:
                line_segments = line_map_to_segments(junctions, line_map)

            outputs["line_segments"] = line_segments

        end_time = time.time()

        if profile:
            outputs["time"] = end_time - start_time
        
        return outputs
    
    def __call__(self, images, valid_masks=[None, None], profile=False):
        # Line detection and descriptor inference on both images
        if self.multiscale:
            forward_outputs = [
                self.multiscale_line_detection(
                    images[0], valid_masks[0], profile=profile,
                    scales=self.scales),
                self.multiscale_line_detection(
                    images[1], valid_masks[1], profile=profile,
                    scales=self.scales)]
        else:
            forward_outputs = [
                self.line_detection(images[0], valid_masks[0],
                                    profile=profile),
                self.line_detection(images[1], valid_masks[1],
                                    profile=profile)]
        line_seg1 = forward_outputs[0]["line_segments"]
        line_seg2 = forward_outputs[1]["line_segments"]
        desc1 = forward_outputs[0]["descriptor"]
        desc2 = forward_outputs[1]["descriptor"]

        # Match the lines in both images
        start_time = time.time()
        matches = self.line_matcher.forward(line_seg1, line_seg2,
                                            desc1, desc2)
        end_time = time.time()

        outputs = {"line_segments": [line_seg1, line_seg2],
                   "matches": matches}

        if profile:
            outputs["line_detection_time"] = (forward_outputs[0]["time"]
                                              + forward_outputs[1]["time"])
            outputs["line_matching_time"] = end_time - start_time
        
        return outputs
