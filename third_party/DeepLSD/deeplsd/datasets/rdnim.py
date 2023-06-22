""" Rotated Day-Night Image Matching dataset. """

import os
import numpy as np
import torch
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from .base_dataset import BaseDataset
from .utils.preprocessing import read_timestamps
from ..settings import DATA_PATH


class RDNIM(BaseDataset, Dataset):
    default_conf = {
        'dataset_dir': 'RDNIM',
        'reference': 'day',
    }

    def _init(self, conf):
        self._root_dir = Path(DATA_PATH, conf.dataset_dir)
        ref = conf.reference

        # Extract the timestamps
        timestamp_files = [p for p
                           in Path(self._root_dir, 'time_stamps').iterdir()]
        timestamps = {}
        for f in timestamp_files:
            id = f.stem
            timestamps[id] = read_timestamps(str(f))

        # Extract the reference images paths
        references = {}
        seq_paths = [p for p in Path(self._root_dir, 'references').iterdir()]
        for seq in seq_paths:
            id = seq.stem
            references[id] = str(Path(seq, ref + '.jpg'))

        # Extract the images paths and the homographies
        seq_path = [p for p in Path(self._root_dir, 'images').iterdir()]
        self._files = []
        for seq in seq_path:
            id = seq.stem
            images_path = [x for x in seq.iterdir() if x.suffix == '.jpg']
            for img in images_path:
                timestamp = timestamps[id]['time'][
                    timestamps[id]['name'].index(img.name)]
                H = np.loadtxt(str(img)[:-4] + '.txt').astype(float)
                self._files.append({
                    'img': str(img),
                    'ref': str(references[id]),
                    'H': H,
                    'timestamp': timestamp})

    def __getitem__(self, item):
        img0_path = self._files[item]['ref']
        img0 = cv2.imread(img0_path, 0)
        img1_path = self._files[item]['img']
        img1 = cv2.imread(img1_path, 0)
        img_size = img0.shape[:2]
        H = self._files[item]['H']

        # Normalize the images in [0, 1]
        img0 = img0.astype(float) / 255.
        img1 = img1.astype(float) / 255.

        img0 = torch.tensor(img0[None], dtype=torch.float)
        img1 = torch.tensor(img1[None], dtype=torch.float)
        H = torch.tensor(H, dtype=torch.float)

        return {'image': img0, 'warped_image': img1, 'H': H,
                'timestamp': self._files[item]['timestamp'],
                'image_path': img0_path, 'warped_image_path': img1_path}

    def __len__(self):
        return len(self._files)

    def get_dataset(self, split):
        assert split in ['test']
        return self

    # Overwrite the parent data loader to handle custom collate_fn
    def get_data_loader(self, split, shuffle=False):
        """Return a data loader for a given split."""
        assert split in ['test']
        batch_size = self.conf.get(split+'_batch_size')
        num_workers = self.conf.get('num_workers', batch_size)
        return DataLoader(self, batch_size=batch_size,
                          shuffle=shuffle or split == 'train',
                          pin_memory=True, num_workers=num_workers)
