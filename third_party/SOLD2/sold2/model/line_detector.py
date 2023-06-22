"""
Line segment detection from raw images.
"""
import time
import numpy as np
import torch
from torch.nn.functional import softmax

from .model_util import get_model
from .loss import get_loss_and_weights
from .line_detection import LineSegmentDetectionModule
from ..train import convert_junc_predictions
from ..misc.train_utils import adapt_checkpoint


def line_map_to_segments(junctions, line_map):
    """ Convert a line map to a Nx2x2 list of segments. """ 
    line_map_tmp = line_map.copy()

    output_segments = np.zeros([0, 2, 2])
    for idx in range(junctions.shape[0]):
        # if no connectivity, just skip it
        if line_map_tmp[idx, :].sum() == 0:
            continue
        # Record the line segment
        else:
            for idx2 in np.where(line_map_tmp[idx, :] == 1)[0]:
                p1 = junctions[idx, :]  # HW format
                p2 = junctions[idx2, :]
                single_seg = np.concatenate([p1[None, ...], p2[None, ...]],
                                            axis=0)
                output_segments = np.concatenate(
                    (output_segments, single_seg[None, ...]), axis=0)
                
                # Update line_map
                line_map_tmp[idx, idx2] = 0
                line_map_tmp[idx2, idx] = 0
    
    return output_segments


class LineDetector(object):
    def __init__(self, model_cfg, ckpt_path, device, line_detector_cfg,
                 junc_detect_thresh=None):
        """ SOLD² line detector taking raw images as input.
        Parameters:
            model_cfg: config for CNN model
            ckpt_path: path to the weights
            line_detector_cfg: config file for the line detection module
        """
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

        if junc_detect_thresh is not None:
            self.junc_detect_thresh = junc_detect_thresh
        else:
            self.junc_detect_thresh = model_cfg.get("detection_thresh", 1/65)
        self.max_num_junctions = model_cfg.get("max_num_junctions", 300)

        # Initialize the line detector
        self.line_detector_cfg = line_detector_cfg
        self.line_detector = LineSegmentDetectionModule(**line_detector_cfg)
    
    def __call__(self, input_image, valid_mask=None,
                 return_heatmap=False, profile=False):
        # Now we restrict input_image to 4D torch tensor
        if ((not len(input_image.shape) == 4)
            or (not isinstance(input_image, torch.Tensor))):
            raise ValueError(
        "[Error] the input image should be a 4D torch tensor.")

        # Move the input to corresponding device
        input_image = input_image.to(self.device)

        # Forward of the CNN backbone
        start_time = time.time()
        with torch.no_grad():
            net_outputs = self.model(input_image)

        junc_np = convert_junc_predictions(
            net_outputs["junctions"], self.grid_size,
            self.junc_detect_thresh, self.max_num_junctions)
        if valid_mask is None:
            junctions = np.where(junc_np["junc_pred_nms"].squeeze())
        else:
            junctions = np.where(junc_np["junc_pred_nms"].squeeze()
                                 * valid_mask)
        junctions = np.concatenate(
            [junctions[0][..., None], junctions[1][..., None]], axis=-1)

        if net_outputs["heatmap"].shape[1] == 2:
            # Convert to single channel directly from here
            heatmap = softmax(net_outputs["heatmap"], dim=1)[:, 1:, :, :]
        else:
            heatmap = torch.sigmoid(net_outputs["heatmap"])
        heatmap = heatmap.cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, 0]

        # Run the line detector.
        line_map, junctions, heatmap = self.line_detector.detect(
            junctions, heatmap, device=self.device)
        heatmap = heatmap.cpu().numpy()
        if isinstance(line_map, torch.Tensor):
            line_map = line_map.cpu().numpy()
        if isinstance(junctions, torch.Tensor):
            junctions = junctions.cpu().numpy()
        line_segments = line_map_to_segments(junctions, line_map)
        end_time = time.time()

        outputs = {"line_segments": line_segments}

        if return_heatmap:
            outputs["heatmap"] = heatmap
        if profile:
            outputs["time"] = end_time - start_time
        
        return outputs
