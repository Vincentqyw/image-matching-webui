"""
Refine the lines matching an extension in a given folder.
"""

import os
import argparse
import numpy as np
import h5py
import cv2
from tqdm import tqdm
from pathlib import Path
import cv2
import torch

from ..models.line_refiner import LineRefiner
from ..utils.experiments import get_best_checkpoint


line_refinement_conf = {
    'line_detection_params': {
        'use_vps': True,
        'optimize_vps': True,
        'filtering': False,
        'lambda_df': 1.,
        'lambda_grad': 1.,
        'lambda_vp': 0.2,
        'threshold': 1.,
        'max_iters': 100000,
        'minimum_point_number': 2,
        'maximum_model_number': -1,
        'scoring_exponent': 1,
    },
}


def refine_lines(img, lines, refiner):
    # Refine the lines (the line refiner uses the xy convention for lines)
    torch_img = img.astype(np.float32) / 255
    torch_img = torch.tensor(torch_img, dtype=torch.float,
                             device='cuda')[None, None]
    inputs = {'image': torch_img, 'lines': lines[None]}
    with torch.no_grad():
        lines = refiner(inputs)['refined_lines'][0]
    return lines[:, :, [1, 0]]


def export(image_folder, line_folder, ckpt):
    # Load the image paths
    img_paths = list(Path(image_folder).iterdir())
    img_paths.sort()
    img_names = [path.stem for path in img_paths]
    line_paths = list(Path(line_folder).iterdir())
    line_paths.sort()
    num_imgs = len(img_paths)

    # Load the line refiner
    model_name = 'refined'
    refiner = LineRefiner(line_refinement_conf)
    ckpt = torch.load(str(ckpt), map_location='cpu')
    refiner.load_state_dict(ckpt['model'], strict=False)
    refiner = refiner.eval().cuda()

    for i in tqdm(range(num_imgs)):
        img = cv2.imread(str(img_paths[i]), 0)

        # Load the pre-computed lines matching the extension
        # Input lines are assumed to be in row-col coordinates
        pred_lines = np.load(str(line_paths[i]))
        lines = pred_lines['lines'][:, :, [1, 0]]

        # Refine the lines
        outputs = {}
        if len(lines) == 0:
            outputs['lines'] = lines[:, :, [1, 0]]
        else:
            outputs['lines'] = refine_lines(img, lines, refiner)

        # Save the results on disk
        filename = img_names[i] + '_' + model_name + '.npz'
        path = os.path.join(line_folder, filename)
        with open(path, 'wb') as output_file:
            np.savez(output_file, **outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'image_folder', type=str,
        help="Path to the original image folder.")
    parser.add_argument(
        'line_folder', type=str,
        help="Path to the folder containing the line detections. Lines are expected in .npz files, with same root name as the images,  with a dict {'lines': my_lines}, where my_lines is an array [n_lines, 2, 2] (row-col convention).")
    parser.add_argument(
        'ckpt', type=str,
        help="Path to the DeepLSD checkpoint.")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        sys.exit('No model found in: ' + args.ckpt)

    export(args.image_folder, args.line_folder, args.ckpt)
