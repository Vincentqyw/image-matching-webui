"""
Export vanishing points (VP).
"""

import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from ..datasets.york_urban import YorkUrban
from ..datasets.nyu import NYU
from ..evaluation.ls_evaluation import vp_consistency_check, get_vp_error, get_recall_AUC


yorkurban_config = {
    'dataset_dir': 'YorkUrbanDB',
}

nyu_config = {
    'dataset_dir': 'NYU_depth_v2',
}


def get_dataloader(dataset):
    if dataset == 'york_urban':
        data = YorkUrban(yorkurban_config)
    elif dataset == 'nyu':
        data = NYU(nyu_config)
    else:
        sys.exit(f"Unknown dataset: {dataset}")
    return data.get_data_loader('test')


def plot_vp_consistency(method_names, x, vp_consistency, lw=2):
    """ Plot the VP consistency of different methods. """
    n_models = len(method_names)
    colors = sns.color_palette(n_colors=n_models)
    for m, y, c in zip(method_names, vp_consistency, colors):
        plt.plot(x, y, label=m, color=c, linewidth=lw)
    plt.legend(loc='lower right', fontsize=16, ncol=2)
    plt.xlabel('Error threshold (in px)', fontsize=18)
    plt.ylabel('VP consistency (in %)', fontsize=18)
    plt.grid()
    plt.savefig('vp_consistency.pdf', bbox_inches='tight', pad_inches=0)


def evaluate(dataset, vp_folder, output_folder, method):
    # Get the dataloader
    dataloader = get_dataloader(dataset)
    thresholds = list(np.arange(1, 9))

    # Gather all metrics across all VP detections
    vp_consistency, vp_error, AUC =  [], [], []
    for data in tqdm(dataloader):
        # GT data
        img_name = data['name'][0]
        K = data['K'][0].numpy()
        if dataset == 'york_urban':
            gt_lines = data['gt_lines'][0].numpy()
            vp_association = data['vp_association'][0].numpy()
            gt_vp = data['updated_vps'][0].numpy()
        else:
            gt_vp = data['vps'][0].numpy()

        # Regress line segments, VPs and the associated VPs
        pred_file = os.path.join(vp_folder, img_name + '_' + method + '.npz')
        with open(pred_file, 'rb') as f:
            vp = np.load(f)['vps']

        # VP consistency
        if dataset == 'york_urban':
            vp_consistency.append(vp_consistency_check(
                gt_lines, vp_association, vp, tol=thresholds))

        # VP error
        vp_error.append(get_vp_error(gt_vp, vp, K))

        # VP recall AUC
        AUC.append(get_recall_AUC(gt_vp, vp, K)[1])

    if dataset == 'york_urban':
        vp_consistency = np.stack(vp_consistency, axis=0).mean(axis=0)
    vp_error = np.stack(vp_error, axis=0).mean(axis=0)
    AUC = np.stack(AUC, axis=0).mean(axis=0)

    # Write the results on disk
    file_path = os.path.join(output_folder, method + '.npz')
    with open(file_path, 'wb') as f:
        if dataset == 'york_urban':
            np.savez(f, vp_consistency=vp_consistency, vp_error=vp_error, AUC=AUC)
        else:
            np.savez(f, vp_error=vp_error, AUC=AUC)

    # Print the results for the requested threshold
    print(f"Results for {method}:")
    print(f'VP error: {np.round(vp_error * 1000) / 1000}')
    print()
    print(f'VP recall AUC: {np.round(AUC * 100) / 1000}')
    if dataset == 'york_urban':
        plot_vp_consistency([method], thresholds, [vp_consistency], lw=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str,
                         help="Dataset to evaluate on ('york_urban' or 'nyu').")
    parser.add_argument('vp_folder', type=str,
                         help="Path to the folder containing all VP detections.")
    parser.add_argument('output_folder', type=str,
                        help="Path to the output folder.")
    parser.add_argument('method', type=str,
                        help="Name of the method (should match with the file extension, e.g. 'deeplsd' if the file ends with 'deeplsd.npz').")
    args = parser.parse_args()

    if not os.path.exists(args.vp_folder):
        sys.exit('No folder found in: ' + args.vp_folder)

    evaluate(args.dataset, args.vp_folder, args.output_folder, args.method)
