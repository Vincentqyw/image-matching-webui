"""
Convert the aggregation results from the homography adaptation to GT labels.
"""
import sys
sys.path.append("../")
import os
import yaml
import argparse
import numpy as np
import h5py
import torch
from tqdm import tqdm

from config.project_config import Config as cfg
from model.line_detection import LineSegmentDetectionModule
from model.metrics import super_nms
from misc.train_utils import parse_h5_data


def convert_raw_exported_predictions(input_data, grid_size=8,
                                     detect_thresh=1/65, topk=300):
    """ Convert the exported junctions and heatmaps predictions
        to a standard format.
    Arguments:
        input_data: the raw data (dict) decoded from the hdf5 dataset
        outputs: dict containing required entries including:
            junctions_pred: Nx2 ndarray containing nms junction predictions.
            heatmap_pred: HxW ndarray containing predicted heatmaps
            valid_mask: HxW ndarray containing the valid mask
    """
    # Check the input_data is from (1) single prediction,
    # or (2) homography adaptation.
    # Homography adaptation raw predictions
    if (("junc_prob_mean" in input_data.keys())
        and ("heatmap_prob_mean" in input_data.keys())):
        # Get the junction predictions and convert if to Nx2 format
        junc_prob = input_data["junc_prob_mean"]
        junc_pred_np = junc_prob[None, ...]
        junc_pred_np_nms = super_nms(junc_pred_np, grid_size,
                                     detect_thresh, topk)
        junctions = np.where(junc_pred_np_nms.squeeze())
        junc_points_pred = np.concatenate([junctions[0][..., None],
                                           junctions[1][..., None]], axis=-1)

        # Get the heatmap predictions
        heatmap_pred = input_data["heatmap_prob_mean"].squeeze()
        valid_mask = np.ones(heatmap_pred.shape, dtype=np.int32)
        
    # Single predictions
    else:
        # Get the junction point predictions and convert to Nx2 format
        junc_points_pred = np.where(input_data["junc_pred_nms"])
        junc_points_pred = np.concatenate(
            [junc_points_pred[0][..., None],
             junc_points_pred[1][..., None]], axis=-1)

        # Get the heatmap predictions
        heatmap_pred = input_data["heatmap_pred"]
        valid_mask = input_data["valid_mask"]

    return {
        "junctions_pred": junc_points_pred,
        "heatmap_pred": heatmap_pred,
        "valid_mask": valid_mask
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dataset", type=str,
                        help="Name of the exported dataset.")
    parser.add_argument("output_dataset", type=str,
                        help="Name of the output dataset.")
    parser.add_argument("config", type=str,
                        help="Path to the model config.")
    args = parser.parse_args()    
    
    # Define the path to the input exported dataset
    exported_dataset_path = os.path.join(cfg.export_dataroot,
                                         args.input_dataset)
    if not os.path.exists(exported_dataset_path):
        raise ValueError("Missing input dataset: " + exported_dataset_path)
    exported_dataset = h5py.File(exported_dataset_path, "r")

    # Define the output path for the results
    output_dataset_path = os.path.join(cfg.export_dataroot,
                                       args.output_dataset)

    device = torch.device("cuda")
    nms_device = torch.device("cuda")
    
    # Read the config file
    if not os.path.exists(args.config):
        raise ValueError("Missing config file: " + args.config)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_cfg = config["model_cfg"]
    line_detector_cfg = config["line_detector_cfg"]
    
    # Initialize the line detection module
    line_detector = LineSegmentDetectionModule(**line_detector_cfg)

    # Iterate through all the dataset keys
    with h5py.File(output_dataset_path, "w") as output_dataset:
        for idx, output_key in enumerate(tqdm(list(exported_dataset.keys()),
                                              ascii=True)):
            # Get the data
            data = parse_h5_data(exported_dataset[output_key])

            # Preprocess the data
            converted_data = convert_raw_exported_predictions(
                data, grid_size=model_cfg["grid_size"],
                detect_thresh=model_cfg["detection_thresh"])
            junctions_pred_raw = converted_data["junctions_pred"]
            heatmap_pred = converted_data["heatmap_pred"]
            valid_mask = converted_data["valid_mask"]

            line_map_pred, junctions_pred, heatmap_pred = line_detector.detect(
                junctions_pred_raw, heatmap_pred, device=device)
            if isinstance(line_map_pred, torch.Tensor):
                line_map_pred = line_map_pred.cpu().numpy()
            if isinstance(junctions_pred, torch.Tensor):
                junctions_pred = junctions_pred.cpu().numpy()
            if isinstance(heatmap_pred, torch.Tensor):
                heatmap_pred = heatmap_pred.cpu().numpy()
            
            output_data = {"junctions": junctions_pred,
                           "line_map": line_map_pred}

            # Record it to the h5 dataset
            f_group = output_dataset.create_group(output_key)

            # Store data
            for key, output_data in output_data.items():
                f_group.create_dataset(key, data=output_data,
                                       compression="gzip")
