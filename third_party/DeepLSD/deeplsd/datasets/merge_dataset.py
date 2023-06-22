"""
Merge multiple datasets for line distance function prediction.
"""

import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from . import get_dataset
from .base_dataset import BaseDataset, worker_init_fn


class MergeDataset(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        'names': ['minidepth', 'wireframe_ha'],
        'dataset_dir': ['MiniDepth', 'Wireframe_raw'],
        'gt_dir': ['export_datasets/minidepth_ha3',
                   'export_datasets/wireframe_ha5'],
        'weights': [0.5, 0.5],
        'resize': [512, 512],
        'photometric_augmentation': {
            'enable': True,
            'primitives': [
                'random_brightness', 'random_contrast',
                'additive_speckle_noise', 'additive_gaussian_noise',
                'additive_shade', 'motion_blur'],
            'params': {
                'random_brightness': {'brightness': 0.5},
                'random_contrast': {'strength_range': [0.5, 1.5]},
                'additive_gaussian_noise': {'stddev_range': [5, 95]},
                'additive_speckle_noise': {'prob_range': [0, 0.01]},
                'additive_shade': {
                    'transparency_range': [-0.8, 0.8],
                    'kernel_size_range': [100, 150]
                },
                'motion_blur': {'max_kernel_size': 3}
            }
        },
        'warped_pair': False,
        'homographic_augmentation': False,
        'homography': {
            'params': {
                'translation': True,
                'rotation': True,
                'scaling': True,
                'perspective': True,
                'scaling_amplitude': 0.2,
                'perspective_amplitude_x': 0.2,
                'perspective_amplitude_y': 0.2,
                'patch_ratio': 0.85,
                'max_angle': 1.57,
                'allow_artifacts': True
            }
        },
        'seed': 0
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        return _Dataset(self.conf, split)
    
    def collate_fn(self, batch):
        """ Customized collate_fn for non-batchable data. """
        batch_keys = ['name', 'image', 'ref_valid_mask', 'df', 'line_level',
                      'offset', 'bg_mask', 'H_ref']
        list_keys = []

        outputs = {}
        for data_key in batch[0].keys():
            batch_match = sum([_ in data_key for _ in batch_keys])
            list_match = sum([_ in data_key for _ in list_keys])
            if batch_match > 0 and list_match == 0:
                outputs[data_key] = torch.utils.data.dataloader.default_collate(
                    [b[data_key] for b in batch])
            elif batch_match == 0 and list_match > 0:
                outputs[data_key] = [b[data_key] for b in batch]
            elif batch_match == 0 and list_match == 0:
                continue
            else:
                raise ValueError(
                    "A key matches batch keys and list keys simultaneously.")
        return outputs

    # Overwrite the parent data loader to handle custom collate_fn
    def get_data_loader(self, split, shuffle=False):
        """Return a data loader for a given split."""
        assert split in ['train', 'val'], "Merge not available in test mode"
        batch_size = self.conf.get(split+'_batch_size')
        num_workers = self.conf.get('num_workers', batch_size)
        return DataLoader(self.get_dataset(split), batch_size=batch_size,
                          shuffle=shuffle or split == 'train',
                          pin_memory=True, num_workers=num_workers,
                          worker_init_fn=worker_init_fn,
                          collate_fn=self.collate_fn)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        assert split in ['train', 'val'], "Merge not available in test mode"
        self.datasets = []
        self.weights = conf.weights
        for i, (name, data_dir, gt_dir) in enumerate(zip(
            conf.names, conf.dataset_dir, conf.gt_dir)):
            if split == 'val' and i > 0:
                # Use only the first dataset for val
                self.weights = [1]
                break
            curr_conf = OmegaConf.to_container(conf, resolve=True)
            curr_conf['dataset_dir'] = data_dir
            curr_conf['gt_dir'] = gt_dir
            del curr_conf['weights']
            del curr_conf['names']
            if name == 'hypersim':
                curr_conf['gt_lines'] = 'pytlsd_reflectance'
                curr_conf['min_perc'] = 0.2
                curr_conf['H_params'] = conf.homography.params
                del curr_conf['homography']
                curr_conf['warped_pair'] = {'enable': conf.warped_pair}
            curr_conf = OmegaConf.create(curr_conf)
            self.datasets.append(
                get_dataset(name)(curr_conf).get_dataset(split))

    def __getitem__(self, idx):
        dataset = self.datasets[np.random.choice(
            range(len(self.datasets)), p=self.weights)]
        return dataset[np.random.randint(len(dataset))]

    def __len__(self):
        return np.sum([len(d) for d in self.datasets])
