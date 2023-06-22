"""
    Export line detections and descriptors given a list of input images.
"""
import os
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm

from .experiment import load_config
from .model.line_matcher import LineMatcher


def export_descriptors(images_list, ckpt_path, config, device, extension,
                       output_folder, multiscale=False):
    # Extract the image paths
    with open(images_list, 'r') as f:
        image_files = f.readlines()
    image_files = [path.strip('\n') for path in image_files]

    # Initialize the line matcher
    line_matcher = LineMatcher(
        config["model_cfg"], ckpt_path, device, config["line_detector_cfg"],
        config["line_matcher_cfg"], multiscale)
    print("\t Successfully initialized model")

    # Run the inference on each image and write the output on disk
    for img_path in tqdm(image_files):
        img = cv2.imread(img_path, 0)
        img = torch.tensor(img[None, None] / 255., dtype=torch.float,
                           device=device)

        # Run the line detection and description
        ref_detection = line_matcher.line_detection(img)
        ref_line_seg = ref_detection["line_segments"]
        ref_descriptors = ref_detection["descriptor"][0].cpu().numpy()

        # Write the output on disk
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        output_file = os.path.join(output_folder, img_name + extension)
        np.savez_compressed(output_file, line_seg=ref_line_seg,
                            descriptors=ref_descriptors)


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_list", type=str, required=True,
                        help="List of input images in a text file.")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to the output folder.")
    parser.add_argument("--config", type=str,
                        default="config/export_line_features.yaml")
    parser.add_argument("--checkpoint_path", type=str,
                        default="pretrained_models/sold2_wireframe.tar")
    parser.add_argument("--multiscale", action="store_true", default=False)
    parser.add_argument("--extension", type=str, default=None)
    args = parser.parse_args()

    # Get the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Get the model config, extension and checkpoint path
    config = load_config(args.config)
    ckpt_path = os.path.abspath(args.checkpoint_path)
    extension = 'sold2' if args.extension is None else args.extension
    extension = "." + extension

    export_descriptors(args.img_list, ckpt_path, config, device, extension,
                       args.output_folder, args.multiscale)
