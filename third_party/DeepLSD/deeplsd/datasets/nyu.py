""" NYU dataset for VP estimation evaluation. """

import os
import csv
import numpy as np
import torch
import cv2
import scipy.io
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from .base_dataset import BaseDataset
from ..evaluation.ls_evaluation import unproject_vp_to_world
from ..settings import DATA_PATH


class NYU(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        'dataset_dir': 'NYU_depth_v2',
        'val_size': 49,
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        assert split in ['val', 'test', 'export']
        return _Dataset(self.conf, split)

    # Overwrite the parent data loader to handle custom split
    def get_data_loader(self, split, shuffle=False):
        """Return a data loader for a given split."""
        assert split in ['val', 'test', 'export']
        batch_size = self.conf.get(split+'_batch_size')
        num_workers = self.conf.get('num_workers', batch_size)
        return DataLoader(self.get_dataset(split), batch_size=batch_size,
                          shuffle=False, pin_memory=True,
                          num_workers=num_workers)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        # Extract the image names
        num_imgs = 1449
        root_dir = os.path.join(DATA_PATH, conf.dataset_dir)
        self.img_paths = [os.path.join(root_dir, 'images', str(i) + '.jpg')
                          for i in range(num_imgs)]
        self.vps_paths = [
            os.path.join(root_dir, 'vps', 'vps_' + str(i).zfill(4) + '.csv')
            for i in range(num_imgs)]
        self.img_names = [str(i).zfill(4) for i in range(num_imgs)]

        # Separate validation and test
        if split == 'val':
            self.img_paths = self.img_paths[-conf.val_size:]
            self.vps_paths = self.vps_paths[-conf.val_size:]
            self.img_names = self.img_names[-conf.val_size:]
        elif split == 'test':
            self.img_paths = self.img_paths[:-conf.val_size]
            self.vps_paths = self.vps_paths[:-conf.val_size]
            self.img_names = self.img_names[:-conf.val_size]

        # Load the intrinsics
        fx_rgb = 5.1885790117450188e+02
        fy_rgb = 5.1946961112127485e+02
        cx_rgb = 3.2558244941119034e+02
        cy_rgb = 2.5373616633400465e+02
        self.K = torch.tensor([[fx_rgb, 0, cx_rgb],
                               [0, fy_rgb, cy_rgb],
                               [0, 0, 1]])

    def get_dataset(self, split):
        return self

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx], 0)

        # Load the GT VPs
        vps = []
        with open(self.vps_paths[idx]) as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            for ri, row in enumerate(reader):
                if ri == 0:
                    continue
                vps.append([float(row[1]), float(row[2]), 1.])
        vps = unproject_vp_to_world(np.array(vps), self.K.numpy())

        # Normalize the images in [0, 1]
        img = img.astype(float) / 255.

        # Convert to torch tensors
        img = torch.tensor(img[None], dtype=torch.float)
        vps = torch.tensor(vps, dtype=torch.float)

        return {'image': img, 'image_path': self.img_paths[idx],
                'name': self.img_names[idx], 'vps': vps, 'K': self.K}        

    def __len__(self):
        return len(self.img_paths)
