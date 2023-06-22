"""
Wireframe dataset to evaluate basic line detection metrics.
"""

from pathlib import Path
import logging
import cv2
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader

from .base_dataset import BaseDataset, worker_init_fn
from .utils.preprocessing import resize_and_crop
from .utils.homographies import sample_homography
from ..settings import DATA_PATH


class WireframeEval(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        'dataset_dir': 'Wireframe_raw',
        'resize': None,
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
        assert split == 'test', "WireframeEval only available in test mode."
        return _Dataset(self.conf, split)

    # Overwrite the parent data loader to handle custom collate_fn
    def get_data_loader(self, split, shuffle=False):
        """Return a data loader for a given split."""
        assert split == 'test', "WireframeEval only available in test mode."
        batch_size = self.conf.get(split+'_batch_size')
        num_workers = self.conf.get('num_workers', batch_size)
        return DataLoader(self.get_dataset(split), batch_size=batch_size,
                          shuffle=shuffle or split == 'train',
                          pin_memory=True, num_workers=num_workers,
                          worker_init_fn=worker_init_fn)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        self.conf, self.split = conf, split
        torch.manual_seed(conf.seed)
        np.random.seed(conf.seed)
        folder = 'test'

        # Extract the images paths
        self.images = Path(DATA_PATH, conf.dataset_dir, folder)
        self.images = [img for img in self.images.iterdir()
                       if str(img)[-3:] == 'png' or str(img)[-3:] == 'jpg']
        if len(self.images) == 0:
            raise ValueError(
                f'Could not find any image in folder: {conf.dataset_dir}.')
        logging.info(f'Found {len(self.images)} in image folder.')
        self.images.sort()
        
        # Pre-generate all the homographies to ensure reproducibility
        self.H = []
        self.w, self.h = 640, 480
        img_size = (self.h, self.w)
        for _ in range(len(self.images)):
            self.H.append(sample_homography(img_size, **conf.homography.params))

    def get_dataset(self, split):
        return self

    def __getitem__(self, idx):
        # Read the image
        path = self.images[idx]
        img = cv2.imread(str(path), 0)
        img_size = np.array(img.shape)
        h, w = img_size

        # Warp the image
        H = self.H[idx]
        warped_img = cv2.warpPerspective(img, H, (w, h))

        # Resize the image and GT if necessary
        if self.conf.resize is not None:
            H = self.adapt_homography_to_preprocessing(H, (h, w), (h, w))
            img_size = self.conf.resize
            h, w = img_size
            img = resize_and_crop(img, img_size)
            warped_img = resize_and_crop(warped_img, img_size)

        # Normalize the images in [0, 1]
        img = img.astype(np.float32) / 255.
        warped_img = warped_img.astype(np.float32) / 255.

        # Convert all data to torch tensors
        img = torch.tensor(img[None], dtype=torch.float)
        H = torch.tensor(H, dtype=torch.float)
        warped_img = torch.tensor(warped_img[None], dtype=torch.float)

        data = {
            'name': path.stem,
            'image': img,
            'warped_image': warped_img,
            'H': H,
        }
        return data

    def __len__(self):
        return len(self.images)

    def adapt_homography_to_preprocessing(self, H, img_shape1, img_shape2):
        source_size1 = np.array(img_shape1, dtype=float)
        source_size2 = np.array(img_shape2, dtype=float)
        target_size = np.array(self.conf.resize, dtype=float)

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
