"""
York Urban DB dataset to evaluate basic line detection metrics.
"""

import os
from pathlib import Path
import cv2
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader

from .base_dataset import BaseDataset, worker_init_fn
from .utils.homographies import sample_homography
from ..settings import DATA_PATH


class YorkUrbanLines(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        'dataset_dir': 'YorkUrbanDB',
        'grayscale': True,
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
        assert split == 'test', "YorkUrbanLines only available in test mode."
        return _Dataset(self.conf, split)

    # Overwrite the parent data loader to handle custom collate_fn
    def get_data_loader(self, split, shuffle=False):
        """Return a data loader for a given split."""
        assert split == 'test', "YorkUrbanLines only available in test mode."
        batch_size = self.conf.get(split+'_batch_size')
        num_workers = self.conf.get('num_workers', batch_size)
        return DataLoader(self.get_dataset(split), batch_size=batch_size,
                          shuffle=shuffle or split == 'train',
                          pin_memory=True, num_workers=num_workers,
                          worker_init_fn=worker_init_fn)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        torch.manual_seed(conf.seed)
        np.random.seed(conf.seed)

        # Extract the image names
        self.root_dir = os.path.join(DATA_PATH, conf.dataset_dir)
        self.img_names = [name for name in os.listdir(self.root_dir)
                          if os.path.isdir(os.path.join(self.root_dir, name))]
        self.grayscale = conf.grayscale
        assert len(self.img_names) == 102
        
        # Pre-generate all the homographies to ensure reproducibility
        self.H = []
        self.w, self.h = 640, 480
        img_size = (self.h, self.w)
        for _ in range(len(self.img_names)):
            self.H.append(sample_homography(img_size, **conf.homography.params))

    def get_dataset(self, split):
        return self

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_names[idx],
                                f'{self.img_names[idx]}.jpg')
        name = str(Path(img_path).stem)
        img = cv2.imread(img_path)
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Warp the image
        warped_img = cv2.warpPerspective(img, self.H[idx], (self.w, self.h),
                                         flags=cv2.INTER_LINEAR)

        # Normalize the images in [0, 1]
        img = img.astype(np.float32) / 255.
        warped_img = warped_img.astype(np.float32) / 255.

        # Convert all data to torch tensors
        if self.grayscale:
            img = torch.tensor(img[None], dtype=torch.float)
            warped_img = torch.tensor(warped_img[None], dtype=torch.float)
        else:
            img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)
            warped_img = torch.tensor(warped_img,
                                      dtype=torch.float).permute(2, 0, 1)

        return {
            'name': name,
            'image': img,
            'warped_image': warped_img,
            'H': self.H[idx],
            'image_path': img_path,
        }

    def __len__(self):
        return len(self.img_names)
