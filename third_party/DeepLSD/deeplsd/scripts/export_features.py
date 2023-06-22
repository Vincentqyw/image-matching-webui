"""
Export line detections for all images of a given dataset.
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import cv2
from tqdm import tqdm
from pathlib import Path
from pyprogressivex import findVanishingPoints

from ..models import get_model
from ..datasets import get_dataset
from ..utils.experiments import get_best_checkpoint
from ..settings import EXPER_PATH


default_vp_params = {
    'threshold': 1.5,
    'conf': 0.99,
    'spatial_coherence_weight': 0.0,
    'neighborhood_ball_radius': 1.0,
    'maximum_tanimoto_similarity': 1.0,
    'max_iters': 100000,
    'minimum_point_number': 2,
    'maximum_model_number': -1,
    'sampler_id': 0,
    'scoring_exponent': 1.0,
    'do_logging': False,
}


def compute_vp_progressive_x(lines, h, w, vp_params=default_vp_params):
    """ Compute vanishing points with progressive-X. """
    # Order lines by decreasing length
    order = np.argsort(np.linalg.norm(lines[:, 0] - lines[:, 1], axis=1))[::-1]
    sorted_lines = lines[order]
    
    # Compute weights based on the line length
    weights = np.linalg.norm(sorted_lines[:, 0] - sorted_lines[:, 1], axis=1)
    weights /= np.amax(weights)
    # weights = np.ones(len(sorted_lines))

    # Compute VPs
    vp, ord_label = findVanishingPoints(np.ascontiguousarray(sorted_lines.reshape(-1, 4)),
                                        np.ascontiguousarray(weights), w, h, **vp_params)

    # Put back in the right order
    label = np.zeros_like(ord_label)
    label[order] = ord_label
    label[label == label.max()] = -1  # Last value is the outlier class

    return vp, label


def export(conf, ckpt, output_folder, extension, pred_vps):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the dataset
    dataset = get_dataset(conf.data.name)(conf.data)
    split = 'export' if (conf.data.name == 'nyu'
                         or conf.data.name == 'eth3d') else 'test'
    dataloader = dataset.get_data_loader(split)

    # Load the model
    ckpt = torch.load(str(ckpt), map_location='cpu')
    net = get_model(conf.model.name)(conf.model)
    net.load_state_dict(ckpt['model'])
    net = net.to(device).eval()

    pred_lines = {}
    for data in tqdm(dataloader):
        input0 = {'image': data['image'].to(device)}
        h, w = data['image'].shape[2:4]
        with torch.no_grad():
            out0 = net(input0)
            pred_lines['lines'] = out0['lines'][0][:, :, [1, 0]]
            if pred_vps:
                if conf.model.optimize:
                    pred_lines['vps'] = out0['vps'][0]
                    pred_lines['vp_labels'] = out0['vp_labels'][0]
                else:
                    # Detect VPs with ProgressiveX
                    pred_lines['vps'], pred_lines['vp_labels'] = compute_vp_progressive_x(
                        out0['lines'][0], h, w)
            if 'warped_image' in data:
                input1 = {'image': data['warped_image'].to(device)}
                pred_lines['warped_lines'] = net(
                    input1)['lines'][0][:, :, [1, 0]]

        # Save the results on disk
        if conf.data.name == 'rdnim':
            img_name = str(Path(data['warped_image_path'][0]).stem)
        elif conf.data.name == 'hpatches':
            img_name = data['warped_name'][0]
        else:
            img_name = data['name'][0]
        filename = img_name + '_deeplsd' + (
            '_'+ extension if extension != '' else '') + '.npz'
        path = os.path.join(output_folder, filename)
        with open(path, 'wb') as output_file:
            np.savez(output_file, **pred_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type=str,
                         help='Path to the config file.')
    parser.add_argument('ckpt', type=str,
                         help="Path to model checkpoint.")
    parser.add_argument('output_folder', type=str,
                        help="Path to the output folder.")
    parser.add_argument('--extension', type=str, default='',
                        help="Extension to the path.")
    parser.add_argument('--pred_vps', action='store_true',
                        help="Add the VP prediction to the prediction.")
    parser.add_argument('dotlist', nargs='*')
    args = parser.parse_intermixed_args()

    conf = OmegaConf.from_cli(args.dotlist)
    if args.conf:
        conf = OmegaConf.merge(OmegaConf.load(args.conf), conf)

    if not os.path.exists(args.ckpt):
        sys.exit('No model found in: ' + args.ckpt)

    export(conf, args.ckpt, args.output_folder, args.extension, args.pred_vps)
