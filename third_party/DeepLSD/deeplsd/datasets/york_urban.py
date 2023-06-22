""" YorkUrban dataset for VP estimation evaluation. """

import os
import numpy as np
import torch
import cv2
import scipy.io
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from .base_dataset import BaseDataset
from ..settings import DATA_PATH


class YorkUrban(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        'dataset_dir': 'YorkUrbanDB',
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        assert split in ['val', 'test']
        return _Dataset(self.conf, split)

    # Overwrite the parent data loader to handle custom collate_fn
    def get_data_loader(self, split, shuffle=False):
        """Return a data loader for a given split."""
        assert split in ['val', 'test']
        batch_size = self.conf.get(split+'_batch_size')
        num_workers = self.conf.get('num_workers', batch_size)
        return DataLoader(self.get_dataset(split), batch_size=batch_size,
                          shuffle=False, pin_memory=True,
                          num_workers=num_workers)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        # Extract the image names
        self.root_dir = os.path.join(DATA_PATH, conf.dataset_dir)
        self.img_names = [name for name in os.listdir(self.root_dir)
                          if os.path.isdir(os.path.join(self.root_dir, name))]
        assert len(self.img_names) == 102

        # Separate validation and test
        split_file = os.path.join(self.root_dir,
                                    'ECCV_TrainingAndTestImageNumbers.mat')
        split_mat = scipy.io.loadmat(split_file)
        if split == 'val':
            valid_set = split_mat['trainingSetIndex'][:, 0] - 1
        else:
            valid_set = split_mat['testSetIndex'][:, 0] - 1
        self.img_names = np.array(self.img_names)[valid_set]
        assert len(self.img_names) == 51

        # Load the intrinsics
        K_file = os.path.join(self.root_dir, 'cameraParameters.mat')
        K_mat = scipy.io.loadmat(K_file)
        f = K_mat['focal'][0, 0] / K_mat['pixelSize'][0, 0]
        p_point = K_mat['pp'][0] - 1  # -1 to convert to 0-based conv
        self.K = torch.tensor([[f, 0, p_point[0]],
                               [0, f, p_point[1]],
                               [0, 0, 1]])

    def get_dataset(self, split):
        return self

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_names[idx],
                                f'{self.img_names[idx]}.jpg')
        name = str(Path(img_path).stem)
        img = cv2.imread(img_path, 0)

        # Load the GT lines and VP association
        lines_file = os.path.join(self.root_dir, self.img_names[idx],
                                  f'{self.img_names[idx]}LinesAndVP.mat')
        lines_mat = scipy.io.loadmat(lines_file)
        lines = lines_mat['lines'].reshape(-1, 2, 2)[:, :, [1, 0]] - 1
        vp_association = lines_mat['vp_association'][:, 0] - 1

        # Load the VPs (non orthogonal ones)
        vp_file = os.path.join(
            self.root_dir, self.img_names[idx],
            f'{self.img_names[idx]}GroundTruthVP_CamParams.mat')
        vps = scipy.io.loadmat(vp_file)['vp'].T
        
        # Keep only the relevant VPs
        unique_vps = np.unique(vp_association)
        vps = vps[unique_vps]
        for i, index in enumerate(unique_vps):
            vp_association[vp_association == index] = i

        # Load the extended VPs of YUD+
        vp_file = os.path.join(
            self.root_dir, self.img_names[idx],
            f'{self.img_names[idx]}UpdatedGroundTruthVP_CamParams.mat')
        updated_vps = scipy.io.loadmat(vp_file)['vp'].T

        # Normalize the images in [0, 1]
        img = img.astype(float) / 255.

        # Convert to torch tensors
        img = torch.tensor(img[None], dtype=torch.float)
        lines = torch.tensor(lines.astype(float), dtype=torch.float)
        vps = torch.tensor(vps, dtype=torch.float)
        updated_vps = torch.tensor(updated_vps, dtype=torch.float)
        vp_association = torch.tensor(vp_association, dtype=torch.int)

        return {'image': img, 'image_path': img_path, 'name': name,
                'gt_lines': lines, 'vps': vps, 'updated_vps': updated_vps,
                'vp_association': vp_association, 'K': self.K}        

    def __len__(self):
        return len(self.img_names)
