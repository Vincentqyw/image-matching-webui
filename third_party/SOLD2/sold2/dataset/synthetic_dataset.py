"""
This file implements the synthetic shape dataset object for pytorch
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import math
import h5py
import pickle
import torch
import numpy as np
import cv2
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
import torch.utils.data.dataloader as torch_loader

from ..config.project_config import Config as cfg
from . import synthetic_util
from .transforms import photometric_transforms as photoaug
from .transforms import homographic_transforms as homoaug
from ..misc.train_utils import parse_h5_data


def synthetic_collate_fn(batch):
    """ Customized collate_fn. """
    batch_keys = ["image", "junction_map", "heatmap",
                  "valid_mask", "homography"]
    list_keys = ["junctions", "line_map", "file_key"]

    outputs = {}
    for data_key in batch[0].keys():
        batch_match = sum([_ in data_key for _ in batch_keys])
        list_match = sum([_ in data_key for _ in list_keys])
        # print(batch_match, list_match)
        if batch_match > 0 and list_match == 0:
            outputs[data_key] = torch_loader.default_collate([b[data_key]
                                                             for b in batch])
        elif batch_match == 0 and list_match > 0:
            outputs[data_key] = [b[data_key] for b in batch]
        elif batch_match == 0 and list_match == 0:
            continue
        else:
            raise ValueError(
        "[Error] A key matches batch keys and list keys simultaneously.")

    return outputs


class SyntheticShapes(Dataset):
    """ Dataset of synthetic shapes. """
    # Initialize the dataset
    def __init__(self, mode="train", config=None):
        super(SyntheticShapes, self).__init__()
        if not mode in ["train", "val", "test"]:
            raise ValueError(
        "[Error] Supported dataset modes are 'train', 'val', and 'test'.")
        self.mode = mode

        # Get configuration
        if config is None:
            self.config = self.get_default_config()
        else:
            self.config = config

        # Set all available primitives
        self.available_primitives = [
            'draw_lines',
            'draw_polygon',
            'draw_multiple_polygons',
            'draw_star',
            'draw_checkerboard_multiseg',
            'draw_stripes_multiseg',
            'draw_cube',
            'gaussian_noise'
        ]

        # Some cache setting
        self.dataset_name = self.get_dataset_name()
        self.cache_name = self.get_cache_name()
        self.cache_path = cfg.synthetic_cache_path

        # Check if export dataset exists
        print("===============================================")
        self.filename_dataset, self.datapoints = self.construct_dataset()
        self.print_dataset_info()

        # Initialize h5 file handle
        self.dataset_path = os.path.join(cfg.synthetic_dataroot, self.dataset_name + ".h5")
        
        # Fix the random seed for torch and numpy in testing mode
        if ((self.mode == "val" or self.mode == "test")
            and self.config["add_augmentation_to_all_splits"]):
            seed = self.config.get("test_augmentation_seed", 200)
            np.random.seed(seed)
            torch.manual_seed(seed)
            # For CuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    ##########################################
    ## Dataset construction related methods ##
    ##########################################
    def construct_dataset(self):
        """ Dataset constructor. """
        # Check if the filename cache exists
        # If cache exists, load from cache
        if self._check_dataset_cache():
            print("[Info]: Found filename cache at ...")
            print("\t Load filename cache...")
            filename_dataset, datapoints = self.get_filename_dataset_from_cache()
            print("\t Check if all file exists...")
            # If all file exists, continue
            if self._check_file_existence(filename_dataset):
                print("\t All files exist!")
            # If not, need to re-export the synthetic dataset
            else:
                print("\t Some files are missing. Re-export the synthetic shape dataset.")
                self.export_synthetic_shapes()
                print("\t Initialize filename dataset")
                filename_dataset, datapoints = self.get_filename_dataset()
                print("\t Create filename dataset cache...")
                self.create_filename_dataset_cache(filename_dataset,
                                                   datapoints)

        # If not, initialize dataset from scratch
        else:
            print("[Info]: Can't find filename cache ...")
            print("\t First check export dataset exists.")
            # If export dataset exists, then just update the filename_dataset
            if self._check_export_dataset():
                print("\t Synthetic dataset exists. Initialize the dataset ...")

            # If export dataset does not exist, export from scratch
            else:
                print("\t Synthetic dataset does not exist. Export the synthetic dataset.")
                self.export_synthetic_shapes()
                print("\t Initialize filename dataset")

            filename_dataset, datapoints = self.get_filename_dataset()
            print("\t Create filename dataset cache...")
            self.create_filename_dataset_cache(filename_dataset, datapoints)

        return filename_dataset, datapoints

    def get_cache_name(self):
        """ Get cache name from dataset config / default config. """
        if self.config["dataset_name"] is None:
            dataset_name = self.default_config["dataset_name"] + "_%s" % self.mode
        else:
            dataset_name = self.config["dataset_name"] + "_%s" % self.mode
        # Compose cache name
        cache_name = dataset_name + "_cache.pkl"

        return cache_name

    def get_dataset_name(self):
        """Get dataset name from dataset config / default config. """
        if self.config["dataset_name"] is None:
            dataset_name = self.default_config["dataset_name"] + "_%s" % self.mode
        else:
            dataset_name = self.config["dataset_name"] + "_%s" % self.mode

        return dataset_name

    def get_filename_dataset_from_cache(self):
        """ Get filename dataset from cache. """
        # Load from the pkl cache
        cache_file_path = os.path.join(self.cache_path, self.cache_name)
        with open(cache_file_path, "rb") as f:
            data = pickle.load(f)

        return data["filename_dataset"], data["datapoints"]

    def get_filename_dataset(self):
        """ Get filename dataset from scratch. """
        # Path to the exported dataset
        dataset_path = os.path.join(cfg.synthetic_dataroot,
                                    self.dataset_name + ".h5")

        filename_dataset = {}
        datapoints = []
        # Open the h5 dataset
        with h5py.File(dataset_path, "r") as f:
            # Iterate through all the primitives
            for prim_name in f.keys():
                filenames = sorted(f[prim_name].keys())
                filenames_full = [os.path.join(prim_name, _)
                                  for _ in filenames]

                filename_dataset[prim_name] = filenames_full
                datapoints += filenames_full

        return filename_dataset, datapoints

    def create_filename_dataset_cache(self, filename_dataset, datapoints):
        """ Create filename dataset cache for faster initialization. """
        # Check cache path exists
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        cache_file_path = os.path.join(self.cache_path, self.cache_name)
        data = {
            "filename_dataset": filename_dataset,
            "datapoints": datapoints
        }
        with open(cache_file_path, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def export_synthetic_shapes(self):
        """ Export synthetic shapes to disk. """
        # Set the global random state for data generation
        synthetic_util.set_random_state(np.random.RandomState(
            self.config["generation"]["random_seed"]))

        # Define the export path
        dataset_path = os.path.join(cfg.synthetic_dataroot,
                                    self.dataset_name + ".h5")

        # Open h5py file
        with h5py.File(dataset_path, "w", libver="latest") as f:
            # Iterate through all types of shape
            primitives = self.parse_drawing_primitives(
                self.config["primitives"])
            split_size = self.config["generation"]["split_sizes"][self.mode]
            for prim in primitives:
                # Create h5 group
                group = f.create_group(prim)
                # Export single primitive
                self.export_single_primitive(prim, split_size, group)

            f.swmr_mode = True

    def export_single_primitive(self, primitive, split_size, group):
        """ Export single primitive. """
        # Check if the primitive is valid or not
        if primitive not in self.available_primitives:
            raise ValueError(
        "[Error]: %s is not a supported primitive" % primitive)
        # Set the random seed
        synthetic_util.set_random_state(np.random.RandomState(
            self.config["generation"]["random_seed"]))

        # Generate shapes
        print("\t Generating %s ..." % primitive)
        for idx in tqdm(range(split_size), ascii=True):
            # Generate background image
            image = synthetic_util.generate_background(
                self.config['generation']['image_size'],
                **self.config['generation']['params']['generate_background'])

            # Generate points
            drawing_func = getattr(synthetic_util, primitive)
            kwarg = self.config["generation"]["params"].get(primitive, {})

            # Get min_len and min_label_len
            min_len = self.config["generation"]["min_len"]
            min_label_len = self.config["generation"]["min_label_len"]

            # Some only take min_label_len, and gaussian noises take nothing
            if primitive in ["draw_lines", "draw_polygon",
                             "draw_multiple_polygons", "draw_star"]:
                data = drawing_func(image, min_len=min_len,
                                    min_label_len=min_label_len, **kwarg)
            elif primitive in ["draw_checkerboard_multiseg",
                               "draw_stripes_multiseg", "draw_cube"]:
                data = drawing_func(image, min_label_len=min_label_len,
                                    **kwarg)
            else:
                data = drawing_func(image, **kwarg)

            # Convert the data
            if data["points"] is not None:
                points = np.flip(data["points"], axis=1).astype(np.float)
                line_map = data["line_map"].astype(np.int32)
            else:
                points = np.zeros([0, 2]).astype(np.float)
                line_map = np.zeros([0, 0]).astype(np.int32)

            # Post-processing
            blur_size = self.config["preprocessing"]["blur_size"]
            image = cv2.GaussianBlur(image, (blur_size, blur_size), 0)

            # Resize the image and the point location.
            points = (points
                      * np.array(self.config['preprocessing']['resize'],
                                 np.float)
                      / np.array(self.config['generation']['image_size'],
                                 np.float))
            image = cv2.resize(
                image, tuple(self.config['preprocessing']['resize'][::-1]),
                interpolation=cv2.INTER_LINEAR)
            image = np.array(image, dtype=np.uint8)

            # Generate the line heatmap after post-processing
            junctions = np.flip(np.round(points).astype(np.int32), axis=1)
            heatmap = (synthetic_util.get_line_heatmap(
                junctions, line_map,
                size=image.shape) * 255.).astype(np.uint8)

            # Record the data in group
            num_pad = math.ceil(math.log10(split_size)) + 1
            file_key_name = self.get_padded_filename(num_pad, idx)
            file_group = group.create_group(file_key_name)

            # Store data
            file_group.create_dataset("points", data=points,
                                      compression="gzip")
            file_group.create_dataset("image", data=image,
                                      compression="gzip")
            file_group.create_dataset("line_map", data=line_map,
                                      compression="gzip")
            file_group.create_dataset("heatmap", data=heatmap,
                                      compression="gzip")

    def get_default_config(self):
        """ Get default configuration of the dataset. """
        # Initialize the default configuration
        self.default_config = {
            "dataset_name": "synthetic_shape",
            "primitives": "all",
            "add_augmentation_to_all_splits": False,
            # Shape generation configuration
            "generation": {
                "split_sizes": {'train': 10000, 'val': 400, 'test': 500},
                "random_seed": 10,
                "image_size": [960, 1280],
                "min_len": 0.09,
                "min_label_len": 0.1,
                'params': {
                    'generate_background': {
                        'min_kernel_size': 150, 'max_kernel_size': 500,
                        'min_rad_ratio': 0.02, 'max_rad_ratio': 0.031},
                    'draw_stripes': {'transform_params': (0.1, 0.1)},
                    'draw_multiple_polygons': {'kernel_boundaries': (50, 100)}
                },
            },
            # Date preprocessing configuration.
            "preprocessing": {
                "resize": [240, 320],
                "blur_size": 11
            },
            'augmentation': {
                'photometric': {
                    'enable': False,
                    'primitives': 'all',
                    'params': {},
                    'random_order': True,
                },
                'homographic': {
                    'enable': False,
                    'params': {},
                    'valid_border_margin': 0,
                },
            }
        }

        return self.default_config

    def parse_drawing_primitives(self, names):
        """ Parse the primitives in config to list of primitive names. """
        if names == "all":
            p = self.available_primitives
        else:
            if isinstance(names, list):
                p = names
            else:
                p = [names]

        assert set(p) <= set(self.available_primitives)

        return p

    @staticmethod
    def get_padded_filename(num_pad, idx):
        """ Get the padded filename using adaptive padding. """
        file_len = len("%d" % (idx))
        filename = "0" * (num_pad - file_len) + "%d" % (idx)

        return filename

    def print_dataset_info(self):
        """ Print dataset info. """
        print("\t ---------Summary------------------")
        print("\t Dataset mode: \t\t %s" % self.mode)
        print("\t Number of primitive: \t %d" % len(self.filename_dataset.keys()))
        print("\t Number of data: \t %d" % len(self.datapoints))
        print("\t ----------------------------------")
    
    #########################
    ## Pytorch related API ##
    #########################
    def get_data_from_datapoint(self, datapoint, reader=None):
        """ Get data given the datapoint
            (keyname of the h5 dataset e.g. "draw_lines/0000.h5"). """
        # Check if the datapoint is valid
        if not datapoint in self.datapoints:
            raise ValueError(
        "[Error] The specified datapoint is not in available datapoints.")

        # Get data from h5 dataset
        if reader is None:
            raise ValueError(
        "[Error] The reader must be provided in __getitem__.")
        else:
            data = reader[datapoint]

        return parse_h5_data(data)

    def get_data_from_signature(self, primitive_name, index):
        """ Get data given the primitive name and index ("draw_lines", 10) """
        # Check the primitive name and index
        self._check_primitive_and_index(primitive_name, index)

        # Get the datapoint from filename dataset
        datapoint = self.filename_dataset[primitive_name][index]

        return self.get_data_from_datapoint(datapoint)

    def parse_transforms(self, names, all_transforms):
        trans = all_transforms if (names == 'all') \
            else (names if isinstance(names, list) else [names])
        assert set(trans) <= set(all_transforms)
        return trans

    def get_photo_transform(self):
        """ Get list of photometric transforms (according to the config). """
        # Get the photometric transform config
        photo_config = self.config["augmentation"]["photometric"]
        if not photo_config["enable"]:
            raise ValueError(
        "[Error] Photometric augmentation is not enabled.")
        
        # Parse photometric transforms
        trans_lst = self.parse_transforms(photo_config["primitives"],
                                          photoaug.available_augmentations)
        trans_config_lst = [photo_config["params"].get(p, {})
                            for p in trans_lst]

        # List of photometric augmentation
        photometric_trans_lst = [
            getattr(photoaug, trans)(**conf) \
            for (trans, conf) in zip(trans_lst, trans_config_lst)
        ]

        return photometric_trans_lst
    
    def get_homo_transform(self):
        """ Get homographic transforms (according to the config). """
        # Get homographic transforms for image
        homo_config = self.config["augmentation"]["homographic"]["params"]
        if not self.config["augmentation"]["homographic"]["enable"]:
            raise ValueError(
        "[Error] Homographic augmentation is not enabled")

        # Parse the homographic transforms
        # ToDo: use the shape from the config
        image_shape = self.config["preprocessing"]["resize"]

        # Compute the min_label_len from config
        try:
            min_label_tmp = self.config["generation"]["min_label_len"]
        except:
            min_label_tmp = None
        
        # float label len => fraction
        if isinstance(min_label_tmp, float): # Skip if not provided
            min_label_len = min_label_tmp * min(image_shape)
        # int label len => length in pixel
        elif isinstance(min_label_tmp, int):
            scale_ratio = (self.config["preprocessing"]["resize"]
                           / self.config["generation"]["image_size"][0])
            min_label_len = (self.config["generation"]["min_label_len"]
                             * scale_ratio)
        # if none => no restriction
        else:
            min_label_len = 0
        
        # Initialize the transform
        homographic_trans = homoaug.homography_transform(
            image_shape, homo_config, 0, min_label_len)

        return homographic_trans

    @staticmethod
    def junc_to_junc_map(junctions, image_size):
        """ Convert junction points to junction maps. """
        junctions = np.round(junctions).astype(np.int)
        # Clip the boundary by image size
        junctions[:, 0] = np.clip(junctions[:, 0], 0., image_size[0]-1)
        junctions[:, 1] = np.clip(junctions[:, 1], 0., image_size[1]-1)

        # Create junction map
        junc_map = np.zeros([image_size[0], image_size[1]])
        junc_map[junctions[:, 0], junctions[:, 1]] = 1

        return junc_map[..., None].astype(np.int)

    def train_preprocessing(self, data, disable_homoaug=False):
        """ Training preprocessing. """
        # Fetch corresponding entries
        image = data["image"]
        junctions = data["points"]
        line_map = data["line_map"]
        heatmap = data["heatmap"]
        image_size = image.shape[:2]

        # Resize the image before the photometric and homographic transforms
        # Check if we need to do the resizing
        if not(list(image.shape) == self.config["preprocessing"]["resize"]):
            # Resize the image and the point location.
            size_old = list(image.shape)
            image = cv2.resize(
                image, tuple(self.config['preprocessing']['resize'][::-1]),
                interpolation=cv2.INTER_LINEAR)
            image = np.array(image, dtype=np.uint8)

            junctions = (
                junctions
                * np.array(self.config['preprocessing']['resize'], np.float)
                / np.array(size_old, np.float))

            # Generate the line heatmap after post-processing
            junctions_xy = np.flip(np.round(junctions).astype(np.int32),
                                   axis=1)
            heatmap = synthetic_util.get_line_heatmap(junctions_xy, line_map,
                                                      size=image.shape)
            heatmap = (heatmap * 255.).astype(np.uint8)

            # Update image size
            image_size = image.shape[:2]
        
        # Declare default valid mask (all ones)
        valid_mask = np.ones(image_size)

        # Check if we need to apply augmentations
        # In training mode => yes.
        # In homography adaptation mode (export mode) => No
        # Check photometric augmentation
        if self.config["augmentation"]["photometric"]["enable"]:
            photo_trans_lst = self.get_photo_transform()
            ### Image transform ###
            np.random.shuffle(photo_trans_lst)
            image_transform = transforms.Compose(
                photo_trans_lst + [photoaug.normalize_image()])
        else:
            image_transform = photoaug.normalize_image()
        image = image_transform(image)

        # Initialize the empty output dict
        outputs = {}
        # Convert to tensor and return the results
        to_tensor = transforms.ToTensor()
        # Check homographic augmentation
        if (self.config["augmentation"]["homographic"]["enable"]
            and disable_homoaug == False):
            homo_trans = self.get_homo_transform()
            # Perform homographic transform
            homo_outputs = homo_trans(image, junctions, line_map)

            # Record the warped results
            junctions = homo_outputs["junctions"]    # Should be HW format
            image = homo_outputs["warped_image"]
            line_map = homo_outputs["line_map"]
            heatmap = homo_outputs["warped_heatmap"]
            valid_mask = homo_outputs["valid_mask"]  # Same for pos and neg
            homography_mat = homo_outputs["homo"]
            
            # Optionally put warpping information first.
            outputs["homography_mat"] = to_tensor(
                homography_mat).to(torch.float32)[0, ...]

        junction_map = self.junc_to_junc_map(junctions, image_size)

        outputs.update({
            "image": to_tensor(image),
            "junctions": to_tensor(np.ascontiguousarray(
                junctions).copy()).to(torch.float32)[0, ...],
            "junction_map": to_tensor(junction_map).to(torch.int),
            "line_map": to_tensor(line_map).to(torch.int32)[0, ...],
            "heatmap": to_tensor(heatmap).to(torch.int32),
            "valid_mask": to_tensor(valid_mask).to(torch.int32),
        })

        return outputs

    def test_preprocessing(self, data):
        """ Test preprocessing. """
        # Fetch corresponding entries
        image = data["image"]
        points = data["points"]
        line_map = data["line_map"]
        heatmap = data["heatmap"]
        image_size = image.shape[:2]

        # Resize the image before the photometric and homographic transforms
        if not (list(image.shape) == self.config["preprocessing"]["resize"]):
            # Resize the image and the point location.
            size_old = list(image.shape)
            image = cv2.resize(
                image, tuple(self.config['preprocessing']['resize'][::-1]),
                interpolation=cv2.INTER_LINEAR)
            image = np.array(image, dtype=np.uint8)

            points = (points
                      * np.array(self.config['preprocessing']['resize'],
                                 np.float)
                      / np.array(size_old, np.float))

            # Generate the line heatmap after post-processing
            junctions = np.flip(np.round(points).astype(np.int32), axis=1)
            heatmap = synthetic_util.get_line_heatmap(junctions, line_map,
                                                      size=image.shape)
            heatmap = (heatmap * 255.).astype(np.uint8)

            # Update image size
            image_size = image.shape[:2]

        ### image transform ###
        image_transform = photoaug.normalize_image()
        image = image_transform(image)

        ### joint transform ###
        junction_map = self.junc_to_junc_map(points, image_size)
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        junctions = to_tensor(points)
        junction_map = to_tensor(junction_map).to(torch.int)
        line_map = to_tensor(line_map)
        heatmap = to_tensor(heatmap)
        valid_mask = to_tensor(np.ones(image_size)).to(torch.int32)

        return {
            "image": image,
            "junctions": junctions,
            "junction_map": junction_map,
            "line_map": line_map,
            "heatmap": heatmap,
            "valid_mask": valid_mask
        }

    def __getitem__(self, index):
        datapoint = self.datapoints[index]

        # Initialize reader and use it
        with h5py.File(self.dataset_path, "r", swmr=True) as reader:
            data = self.get_data_from_datapoint(datapoint, reader)

        # Apply different transforms in different mod.
        if (self.mode == "train"
            or self.config["add_augmentation_to_all_splits"]):
            return_type = self.config.get("return_type", "single")
            data = self.train_preprocessing(data)
        else:
            data = self.test_preprocessing(data)

        return data

    def __len__(self):
        return len(self.datapoints)

    ########################
    ## Some other methods ##
    ########################
    def _check_dataset_cache(self):
        """ Check if dataset cache exists. """
        cache_file_path = os.path.join(self.cache_path, self.cache_name)
        if os.path.exists(cache_file_path):
            return True
        else:
            return False

    def _check_export_dataset(self):
        """ Check if exported dataset exists. """
        dataset_path = os.path.join(cfg.synthetic_dataroot, self.dataset_name)
        if os.path.exists(dataset_path) and len(os.listdir(dataset_path)) > 0:
            return True
        else:
            return False

    def _check_file_existence(self, filename_dataset):
        """ Check if all exported file exists. """
        # Path to the exported dataset
        dataset_path = os.path.join(cfg.synthetic_dataroot, 
                                    self.dataset_name + ".h5")

        flag = True
        # Open the h5 dataset
        with h5py.File(dataset_path, "r") as f:
            # Iterate through all the primitives
            for prim_name in f.keys():
                if (len(filename_dataset[prim_name])
                    != len(f[prim_name].keys())):
                    flag = False

        return flag

    def _check_primitive_and_index(self, primitive, index):
        """ Check if the primitve and index are valid. """
        # Check primitives
        if not primitive in self.available_primitives:
            raise ValueError(
                "[Error] The primitive is not in available primitives.")

        prim_len = len(self.filename_dataset[primitive])
        # Check the index
        if not index < prim_len:
            raise ValueError(
                "[Error] The index exceeds the total file counts %d for %s"
                % (prim_len, primitive))
