"""
HPatches sequences dataset, to perform homography estimation and
evaluate basic line detection metrics.
"""
import os
import numpy as np
import torch
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from .base_dataset import BaseDataset
from ..settings import DATA_PATH



class HPatches(BaseDataset, Dataset):
    default_conf = {
        'dataset_dir': 'HPatches_sequences',
        'alteration': 'all',  # 'i', 'v' or 'all'
        'max_side': 1200,
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        assert split in ['test', 'export']
        return _Dataset(self.conf, split)

    # Overwrite the parent data loader to handle export mode
    def get_data_loader(self, split, shuffle=False):
        """Return a data loader for a given split."""
        assert split in ['test', 'export']
        batch_size = self.conf.get(split+'_batch_size')
        num_workers = self.conf.get('num_workers', batch_size)
        return DataLoader(
            self.get_dataset(split), batch_size=batch_size,
            shuffle=False, pin_memory=True, num_workers=num_workers)


class _Dataset(Dataset):
    def __init__(self, conf, split):
        self.conf = conf
        self.root_dir = Path(DATA_PATH, conf.dataset_dir)
        folder_paths = [x for x in self.root_dir.iterdir() if x.is_dir()]
        self.data = []
        for path in folder_paths:
            if conf.alteration == 'i' and path.stem[0] != 'i':
                continue
            if conf.alteration == 'v' and path.stem[0] != 'v':
                continue
            if split == 'test':
                for i in range(2, 7):
                    ref_path = Path(path, "1.ppm")
                    target_path = Path(path, str(i) + '.ppm')
                    self.data += [{
                        "ref_name": str(ref_path.parent.stem + "_" + ref_path.stem),
                        "ref_img_path": str(ref_path),
                        "target_name": str(target_path.parent.stem + "_" + target_path.stem),
                        "target_img_path": str(target_path),
                        "H": np.loadtxt(str(Path(path, "H_1_" + str(i)))),
                    }]
            else:
                for i in range(1, 7):
                    ref_path = Path(path, str(i) + '.ppm')
                    self.data += [{
                        "ref_name": str(ref_path.parent.stem + "_" + ref_path.stem),
                        "ref_img_path": str(ref_path)}]
        
    def get_dataset(self, split):
        return self

    def __getitem__(self, idx):
        img0_path = self.data[idx]['ref_img_path']
        img0 = cv2.imread(img0_path, 0)
        img_size = img0.shape

        if max(img_size) > self.conf.max_side:
            s = self.conf.max_side / max(img_size)
            h_s = int(img_size[0] * s)
            w_s = int(img_size[1] * s)
            img0 = cv2.resize(img0, (w_s, h_s), interpolation=cv2.INTER_AREA)

        # Normalize the image in [0, 1]
        img0 = img0.astype(float) / 255.
        img0 = torch.tensor(img0[None], dtype=torch.float)
        outputs = {'image': img0, 'image_path': img0_path,
                   'name': self.data[idx]['ref_name']}

        if 'target_name' in self.data[idx]:
            img1_path = self.data[idx]['target_img_path']
            img1 = cv2.imread(img1_path, 0)
            H = self.data[idx]['H']

            if max(img_size) > self.conf.max_side:
                img1 = cv2.resize(img1, (w_s, h_s),
                                  interpolation=cv2.INTER_AREA)
                H = self.adapt_homography_to_preprocessing(
                    H, img_size, img_size, (h_s, w_s))

            # Normalize the image in [0, 1]
            img1 = img1.astype(float) / 255.
            img1 = torch.tensor(img1[None], dtype=torch.float)
            H = torch.tensor(H, dtype=torch.float)

            outputs['warped_image'] = img1
            outputs['warped_image_path'] = img1_path
            outputs['warped_name'] = self.data[idx]['target_name']
            outputs['H'] = H

        return outputs

    def __len__(self):
        return len(self.data)

    def adapt_homography_to_preprocessing(self, H, img_shape1, img_shape2,
                                          target_size):
        source_size1 = np.array(img_shape1, dtype=float)
        source_size2 = np.array(img_shape2, dtype=float)
        target_size = np.array(target_size)

        # Get the scaling factor in resize
        scale1 = np.amax(target_size / source_size1)
        scaling1 = np.diag([1. / scale1, 1. / scale1, 1.]).astype(float)
        scale2 = np.amax(target_size / source_size2)
        scaling2 = np.diag([scale2, scale2, 1.]).astype(float)

        # Get the translation params in crop
        pad_y1 = (source_size1[0] * scale1 - target_size[0]) / 2.
        pad_x1 = (source_size1[1] * scale1 - target_size[1]) / 2.
        translation1 = np.array([[1., 0., pad_x1],
                                 [0., 1., pad_y1],
                                 [0., 0., 1.]], dtype=float)
        pad_y2 = (source_size2[0] * scale2 - target_size[0]) / 2.
        pad_x2 = (source_size2[1] * scale2 - target_size[1]) / 2.
        translation2 = np.array([[1., 0., -pad_x2],
                                 [0., 1., -pad_y2],
                                 [0., 0., 1.]], dtype=float)

        return translation2 @ scaling2 @ H @ scaling1 @ translation1
