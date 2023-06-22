"""
Evaluate line detections with low-level metrics.
"""

import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from ..datasets.wireframe_eval import WireframeEval
from ..datasets.hpatches import HPatches
from ..datasets.rdnim import RDNIM
from ..datasets.york_urban_lines import YorkUrbanLines
from ..datasets.utils.homographies import warp_lines
from ..evaluation.ls_evaluation import (
    match_segments_1_to_1, compute_repeatability,
    compute_loc_error, H_estimation, match_segments_lbd)


wireframe_config = {
    'dataset_dir': 'Wireframe_raw',
    'resize': None,
}

hpatches_config = {
    'dataset_dir': 'HPatches_sequences',
    'alteration': 'all',
    'max_side': 1200,
}

rdnim_config = {
    'dataset_dir': 'RDNIM',
    'reference': 'night',
}

yorkurban_config = {
    'dataset_dir': 'YorkUrbanDB',
}

num_lines_thresholds = [10, 25, 50, 100, 300]
thresholds = [1, 2, 3, 4, 5]


def get_dataloader(dataset):
    if dataset == 'wireframe':
        data = WireframeEval(wireframe_config)
    elif dataset == 'hpatches':
        data = HPatches(hpatches_config)
    elif dataset == 'rdnim':
        data = RDNIM(rdnim_config)
    elif dataset == 'york_urban':
        data = YorkUrbanLines(yorkurban_config)
    else:
        sys.exit(f"Unknown dataset: {dataset}")
    return data.get_data_loader('test')


def evaluate(dataset, line_folder, output_folder, method, thresh):
    # Get the dataloader
    dataloader = get_dataloader(dataset)
    min_length = 20

    # Gather all metrics across all line detections
    (struct_rep, struct_loc_error, orth_rep, orth_loc_error,
     H_estim, num_lines) = [], [], [], [], [], []
    for data in tqdm(dataloader):
        img = (data['image'].numpy()[0, 0] * 255).astype(np.uint8)
        img_size = img.shape
        H = data['H'][0].numpy()
        if dataset == 'hpatches':
            img_name = data['warped_name'][0]
        elif dataset == 'rdnim':
            img_name = str(Path(data['warped_image_path'][0]).stem)
        else:
            img_name = data['name'][0]

        pred_file = os.path.join(line_folder, img_name + '_' + method + '.npz')
        with open(pred_file, 'rb') as f:
            data = np.load(f)
            pred_lines0 = data['lines']
            pred_lines1 = data['warped_lines']

        # Filter out small lines
        pred_lines0 = pred_lines0[
            np.linalg.norm(pred_lines0[:, 1] - pred_lines0[:, 0], axis=1) > min_length]
        pred_lines1 = pred_lines1[
            np.linalg.norm(pred_lines1[:, 1] - pred_lines1[:, 0], axis=1) > min_length]

        # Compute the average number of lines
        num_lines.append((len(pred_lines0) + len(pred_lines1)) / 2)

        # Compute the structural metrics
        segs1, segs2, matched_idx1, matched_idx2, distances = match_segments_1_to_1(
            pred_lines0, pred_lines1, H, img_size, line_dist='struct', dist_thresh=5)
        if len(matched_idx1) == 0:
            struct_rep.append([0] * len(thresholds))
        else:
            struct_rep.append(compute_repeatability(segs1, segs2, matched_idx1, matched_idx2,
                                                    distances, thresholds, rep_type='num'))
            struct_loc_error.append(compute_loc_error(distances, num_lines_thresholds))

        # Compute the orthogonal metrics
        segs1, segs2, matched_idx1, matched_idx2, distances = match_segments_1_to_1(
            pred_lines0, pred_lines1, H, img_size, line_dist='orth', dist_thresh=5)
        if len(matched_idx1) == 0:
            orth_rep.append([0] * len(thresholds))
        else:
            orth_rep.append(compute_repeatability(segs1, segs2, matched_idx1, matched_idx2,
                                                  distances, thresholds, rep_type='num'))
            orth_loc_error.append(compute_loc_error(distances, num_lines_thresholds))

        # Homography estimation
        segs1, segs2, matched_idx1, matched_idx2 = match_segments_lbd(
            img, pred_lines0, pred_lines1, H, img_size)
        if len(matched_idx1) < 3:
            H_estim.append(0)
        else:
            matched_seg1 = segs1[matched_idx1]
            matched_seg2 = warp_lines(segs2, H)[matched_idx2]
            score = H_estimation(matched_seg1, matched_seg2, H,
                                 img_size, reproj_thresh=3)[0]
            H_estim.append(score)

    num_lines = np.mean(num_lines)
    struct_rep = np.mean(np.stack(struct_rep, axis=0), axis=0)
    struct_loc_error = np.mean(np.stack(struct_loc_error, axis=0), axis=0)
    orth_rep = np.mean(np.stack(orth_rep, axis=0), axis=0)
    orth_loc_error = np.mean(np.stack(orth_loc_error, axis=0), axis=0)
    H_estim = np.mean(H_estim)

    # Write the results on disk
    file_path = os.path.join(output_folder, method + '.npz')
    with open(file_path, 'wb') as f:
        np.savez(f, struct_rep=struct_rep, struct_loc_error=struct_loc_error,
                 orth_rep=orth_rep, orth_loc_error=orth_loc_error,
                 H_estim=H_estim, num_lines=num_lines)

    # Print the results for the requested threshold
    print(f"Results for {method}:")
    print(f'Num lines: {np.round(num_lines * 1000) / 1000}')
    print()
    print(f'Struct-repeatability: {np.round(struct_rep[thresh - 1] * 1000) / 1000}')
    print()
    print(f'Struct-loc: {np.round(struct_loc_error[2] * 1000) / 1000}')
    print()
    print(f'Orth-repeatability: {np.round(orth_rep[thresh - 1] * 1000) / 1000}')
    print()
    print(f'Orth-loc: {np.round(orth_loc_error[2] * 1000) / 1000}')
    print()
    print(f'H estimation: {np.round(H_estim * 1000) / 1000}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str,
                         help="Dataset to evaluate on ('wireframe', 'hpatches', 'rdnim', 'york_urban').")
    parser.add_argument('line_folder', type=str,
                         help="Path to the fodler containing all line detections.")
    parser.add_argument('output_folder', type=str,
                        help="Path to the output folder.")
    parser.add_argument('method', type=str,
                        help="Name of the method (should match with the file extension, e.g. 'deeplsd' if the file ends with 'deeplsd.npz').")
    parser.add_argument('--thresh', type=int, default=3,
                        help="Threshold for repeatability and homography estimation (from 1 to 5, default: 3).")
    args = parser.parse_args()

    if not os.path.exists(args.line_folder):
        sys.exit('No folder found in: ' + args.line_folder)

    if args.thresh not in [1, 2, 3, 4, 5]:
        sys.exit('Invalid threshold, should be 1, 2, 3, 4, or 5.')

    evaluate(args.dataset, args.line_folder, args.output_folder,
             args.method, args.thresh)
