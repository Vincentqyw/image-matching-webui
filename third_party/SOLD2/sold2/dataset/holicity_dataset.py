"""
File to process and load the Holicity dataset.
"""
import os
import math
import copy
import PIL
import numpy as np
import h5py
import cv2
import pickle
from skimage.io import imread
from skimage import color
import torch
import torch.utils.data.dataloader as torch_loader
from torch.utils.data import Dataset
from torchvision import transforms

from ..config.project_config import Config as cfg
from .transforms import photometric_transforms as photoaug
from .transforms import homographic_transforms as homoaug
from .transforms.utils import random_scaling
from .synthetic_util import get_line_heatmap
from ..misc.geometry_utils import warp_points, mask_points
from ..misc.train_utils import parse_h5_data


def holicity_collate_fn(batch):
    """ Customized collate_fn. """
    batch_keys = ["image", "junction_map", "valid_mask", "heatmap",
                  "heatmap_pos", "heatmap_neg", "homography",
                  "line_points", "line_indices"]
    list_keys = ["junctions", "line_map", "line_map_pos",
                 "line_map_neg", "file_key"]

    outputs = {}
    for data_key in batch[0].keys():
        batch_match = sum([_ in data_key for _ in batch_keys])
        list_match = sum([_ in data_key for _ in list_keys])
        # print(batch_match, list_match)
        if batch_match > 0 and list_match == 0:
            outputs[data_key] = torch_loader.default_collate(
                [b[data_key] for b in batch])
        elif batch_match == 0 and list_match > 0:
            outputs[data_key] = [b[data_key] for b in batch]
        elif batch_match == 0 and list_match == 0:
            continue
        else:
            raise ValueError(
        "[Error] A key matches batch keys and list keys simultaneously.")

    return outputs


class HolicityDataset(Dataset):
    def __init__(self, mode="train", config=None):
        super(HolicityDataset, self).__init__()
        if not mode in ["train", "test"]:
            raise ValueError(
        "[Error] Unknown mode for Holicity dataset. Only 'train' and 'test'.")
        self.mode = mode

        if config is None:
            self.config = self.get_default_config()
        else:
            self.config = config
        # Also get the default config
        self.default_config = self.get_default_config()

        # Get cache setting
        self.dataset_name = self.get_dataset_name()
        self.cache_name = self.get_cache_name()
        self.cache_path = cfg.holicity_cache_path
        
        # Get the ground truth source if it exists
        self.gt_source = None
        if "gt_source_%s"%(self.mode) in self.config:
            self.gt_source = self.config.get("gt_source_%s"%(self.mode))
            self.gt_source = os.path.join(cfg.export_dataroot, self.gt_source)
            # Check the full path exists
            if not os.path.exists(self.gt_source):
                raise ValueError(
            "[Error] The specified ground truth source does not exist.")
        
        # Get the filename dataset
        print("[Info] Initializing Holicity dataset...")
        self.filename_dataset, self.datapoints = self.construct_dataset()

        # Get dataset length
        self.dataset_length = len(self.datapoints)

        # Print some info
        print("[Info] Successfully initialized dataset")
        print("\t Name: Holicity")
        print("\t Mode: %s" %(self.mode))
        print("\t Gt: %s" %(self.config.get("gt_source_%s"%(self.mode),
                                            "None")))
        print("\t Counts: %d" %(self.dataset_length))
        print("----------------------------------------")

    #######################################
    ## Dataset construction related APIs ##
    #######################################
    def construct_dataset(self):
        """ Construct the dataset (from scratch or from cache). """
        # Check if the filename cache exists
        # If cache exists, load from cache
        if self.check_dataset_cache():
            print("\t Found filename cache %s at %s"%(self.cache_name,
                                                      self.cache_path))
            print("\t Load filename cache...")
            filename_dataset, datapoints = self.get_filename_dataset_from_cache()
        # If not, initialize dataset from scratch
        else:
            print("\t Can't find filename cache ...")
            print("\t Create filename dataset from scratch...")
            filename_dataset, datapoints = self.get_filename_dataset()
            print("\t Create filename dataset cache...")
            self.create_filename_dataset_cache(filename_dataset, datapoints)
        
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
    
    def get_filename_dataset_from_cache(self):
        """ Get filename dataset from cache. """
        # Load from pkl cache
        cache_file_path = os.path.join(self.cache_path, self.cache_name)
        with open(cache_file_path, "rb") as f:
            data = pickle.load(f)
        
        return data["filename_dataset"], data["datapoints"]

    def get_filename_dataset(self):
        """ Get the path to the dataset. """
        if self.mode == "train":
            # Contains 5720 or 11872 images
            dataset_path = [os.path.join(cfg.holicity_dataroot, p)
                            for p in self.config["train_splits"]]
        else:
            # Test mode - Contains 520 images
            dataset_path = [os.path.join(cfg.holicity_dataroot, "2018-03")]
        
        # Get paths to all image files
        image_paths = []
        for folder in dataset_path:
            image_paths += [os.path.join(folder, img)
                            for img in os.listdir(folder)
                            if os.path.splitext(img)[-1] == ".jpg"]
        image_paths = sorted(image_paths)

        # Verify all the images exist
        for idx in range(len(image_paths)):
            image_path = image_paths[idx]
            if not (os.path.exists(image_path)):
                raise ValueError(
            "[Error] The image does not exist. %s"%(image_path))

        # Construct the filename dataset
        num_pad = int(math.ceil(math.log10(len(image_paths))) + 1)
        filename_dataset = {}
        for idx in range(len(image_paths)):
            # Get the file key
            key = self.get_padded_filename(num_pad, idx)

            filename_dataset[key] = {"image": image_paths[idx]}
        
        # Get the datapoints
        datapoints = list(sorted(filename_dataset.keys()))

        return filename_dataset, datapoints
    
    def get_dataset_name(self):
        """ Get dataset name from dataset config / default config. """
        dataset_name = self.config.get("dataset_name",
                                       self.default_config["dataset_name"])
        dataset_name = dataset_name + "_%s" % self.mode
        return dataset_name
    
    def get_cache_name(self):
        """ Get cache name from dataset config / default config. """
        dataset_name = self.config.get("dataset_name",
                                       self.default_config["dataset_name"])
        dataset_name = dataset_name + "_%s" % self.mode
        # Compose cache name
        cache_name = dataset_name + "_cache.pkl"
        return cache_name

    def check_dataset_cache(self):
        """ Check if dataset cache exists. """
        cache_file_path = os.path.join(self.cache_path, self.cache_name)
        if os.path.exists(cache_file_path):
            return True
        else:
            return False
    
    @staticmethod
    def get_padded_filename(num_pad, idx):
        """ Get the padded filename using adaptive padding. """
        file_len = len("%d" % (idx))
        filename = "0" * (num_pad - file_len) + "%d" % (idx)
        return filename

    def get_default_config(self):
        """ Get the default configuration. """
        return {
            "dataset_name": "holicity",
            "train_split": "2018-01",
            "add_augmentation_to_all_splits": False,
            "preprocessing": {
                "resize": [512, 512],
                "blur_size": 11
            },
            "augmentation":{
                "photometric":{
                    "enable": False
                },
                "homographic":{
                    "enable": False
                },
            },
        }
        
    ############################################
    ## Pytorch and preprocessing related APIs ##
    ############################################
    @staticmethod
    def get_data_from_path(data_path):
        """ Get data from the information from filename dataset. """
        output = {}

        # Get image data
        image_path = data_path["image"]
        image = imread(image_path)
        output["image"] = image
        
        return output
    
    @staticmethod
    def convert_line_map(lcnn_line_map, num_junctions):
        """ Convert the line_pos or line_neg
            (represented by two junction indexes) to our line map. """
        # Initialize empty line map
        line_map = np.zeros([num_junctions, num_junctions])

        # Iterate through all the lines
        for idx in range(lcnn_line_map.shape[0]):
            index1 = lcnn_line_map[idx, 0]
            index2 = lcnn_line_map[idx, 1]

            line_map[index1, index2] = 1
            line_map[index2, index1] = 1
        
        return line_map

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
    
    def parse_transforms(self, names, all_transforms):
        """ Parse the transform. """
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

    def get_line_points(self, junctions, line_map, H1=None, H2=None,
                        img_size=None, warp=False):
        """ Sample evenly points along each line segments
            and keep track of line idx. """
        if np.sum(line_map) == 0:
            # No segment detected in the image
            line_indices = np.zeros(self.config["max_pts"], dtype=int)
            line_points = np.zeros((self.config["max_pts"], 2), dtype=float)
            return line_points, line_indices

        # Extract all pairs of connected junctions
        junc_indices = np.array(
            [[i, j] for (i, j) in zip(*np.where(line_map)) if j > i])
        line_segments = np.stack([junctions[junc_indices[:, 0]],
                                  junctions[junc_indices[:, 1]]], axis=1)
        # line_segments is (num_lines, 2, 2)
        line_lengths = np.linalg.norm(
            line_segments[:, 0] - line_segments[:, 1], axis=1)

        # Sample the points separated by at least min_dist_pts along each line
        # The number of samples depends on the length of the line
        num_samples = np.minimum(line_lengths // self.config["min_dist_pts"],
                                 self.config["max_num_samples"])
        line_points = []
        line_indices = []
        cur_line_idx = 1
        for n in np.arange(2, self.config["max_num_samples"] + 1):
            # Consider all lines where we can fit up to n points
            cur_line_seg = line_segments[num_samples == n]
            line_points_x = np.linspace(cur_line_seg[:, 0, 0],
                                        cur_line_seg[:, 1, 0],
                                        n, axis=-1).flatten()
            line_points_y = np.linspace(cur_line_seg[:, 0, 1],
                                        cur_line_seg[:, 1, 1],
                                        n, axis=-1).flatten()
            jitter = self.config.get("jittering", 0)
            if jitter:
                # Add a small random jittering of all points along the line
                angles = np.arctan2(
                    cur_line_seg[:, 1, 0] - cur_line_seg[:, 0, 0],
                    cur_line_seg[:, 1, 1] - cur_line_seg[:, 0, 1]).repeat(n)
                jitter_hyp = (np.random.rand(len(angles)) * 2 - 1) * jitter
                line_points_x += jitter_hyp * np.sin(angles)
                line_points_y += jitter_hyp * np.cos(angles)
            line_points.append(np.stack([line_points_x, line_points_y], axis=-1))
            # Keep track of the line indices for each sampled point
            num_cur_lines = len(cur_line_seg)
            line_idx = np.arange(cur_line_idx, cur_line_idx + num_cur_lines)
            line_indices.append(line_idx.repeat(n))
            cur_line_idx += num_cur_lines
        line_points = np.concatenate(line_points,
                                     axis=0)[:self.config["max_pts"]]
        line_indices = np.concatenate(line_indices,
                                      axis=0)[:self.config["max_pts"]]

        # Warp the points if need be, and filter unvalid ones
        # If the other view is also warped
        if warp and H2 is not None:
            warp_points2 = warp_points(line_points, H2)
            line_points = warp_points(line_points, H1)
            mask = mask_points(line_points, img_size)
            mask2 = mask_points(warp_points2, img_size)
            mask = mask * mask2
        # If the other view is not warped
        elif warp and H2 is None:
            line_points = warp_points(line_points, H1)
            mask = mask_points(line_points, img_size)
        else:
            if H1 is not None:
                raise ValueError("[Error] Wrong combination of homographies.")
            # Remove points that would be outside of img_size if warped by H
            warped_points = warp_points(line_points, H1)
            mask = mask_points(warped_points, img_size)
        line_points = line_points[mask]
        line_indices = line_indices[mask]
        
        # Pad the line points to a fixed length
        # Index of 0 means padded line
        line_indices = np.concatenate([line_indices, np.zeros(
            self.config["max_pts"] - len(line_indices))], axis=0)
        line_points = np.concatenate(
            [line_points,
             np.zeros((self.config["max_pts"] - len(line_points), 2),
                      dtype=float)], axis=0)
        
        return line_points, line_indices

    def export_preprocessing(self, data, numpy=False):
        """ Preprocess the exported data. """
        # Fetch the corresponding entries
        image = data["image"]
        image_size = image.shape[:2]

        # Resize the image before photometric and homographical augmentations
        if not(list(image_size) == self.config["preprocessing"]["resize"]):
            # Resize the image and the point location.
            size_old = list(image.shape)[:2] # Only H and W dimensions

            image = cv2.resize(
                image, tuple(self.config['preprocessing']['resize'][::-1]),
                interpolation=cv2.INTER_LINEAR)
            image = np.array(image, dtype=np.uint8)
        
        # Optionally convert the image to grayscale
        if self.config["gray_scale"]:
            image = (color.rgb2gray(image) * 255.).astype(np.uint8)

        image = photoaug.normalize_image()(image)

        # Convert to tensor and return the results
        to_tensor = transforms.ToTensor()
        if not numpy:
            return {"image": to_tensor(image)}
        else:
            return {"image": image}
    
    def train_preprocessing_exported(
        self, data, numpy=False, disable_homoaug=False, desc_training=False,
        H1=None, H1_scale=None, H2=None, scale=1., h_crop=None, w_crop=None):
        """ Train preprocessing for the exported labels. """
        data = copy.deepcopy(data)
        # Fetch the corresponding entries
        image = data["image"]
        junctions = data["junctions"]
        line_map = data["line_map"]
        image_size = image.shape[:2]

        # Define the random crop for scaling if necessary
        if h_crop is None or w_crop is None:
            h_crop, w_crop = 0, 0
            if scale > 1:
                H, W = self.config["preprocessing"]["resize"]
                H_scale, W_scale = round(H * scale), round(W * scale)
                if H_scale > H:
                    h_crop = np.random.randint(H_scale - H)
                if W_scale > W:
                    w_crop = np.random.randint(W_scale - W)

        # Resize the image before photometric and homographical augmentations
        if not(list(image_size) == self.config["preprocessing"]["resize"]):
            # Resize the image and the point location.
            size_old = list(image.shape)[:2] # Only H and W dimensions

            image = cv2.resize(
                image, tuple(self.config['preprocessing']['resize'][::-1]),
                interpolation=cv2.INTER_LINEAR)
            image = np.array(image, dtype=np.uint8)

            # # In HW format
            # junctions = (junctions * np.array(
            #     self.config['preprocessing']['resize'], np.float)
            #              / np.array(size_old, np.float))

        # Generate the line heatmap after post-processing
        junctions_xy = np.flip(np.round(junctions).astype(np.int32), axis=1)
        image_size = image.shape[:2]
        heatmap = get_line_heatmap(junctions_xy, line_map, image_size)

        # Optionally convert the image to grayscale
        if self.config["gray_scale"]:
            image = (color.rgb2gray(image) * 255.).astype(np.uint8)

        # Check if we need to apply augmentations
        # In training mode => yes.
        # In homography adaptation mode (export mode) => No
        if self.config["augmentation"]["photometric"]["enable"]:
            photo_trans_lst = self.get_photo_transform()
            ### Image transform ###
            np.random.shuffle(photo_trans_lst)
            image_transform = transforms.Compose(
                photo_trans_lst + [photoaug.normalize_image()])
        else:
            image_transform = photoaug.normalize_image()
        image = image_transform(image)

        # Perform the random scaling
        if scale != 1.:
            image, junctions, line_map, valid_mask = random_scaling(
                 image, junctions, line_map, scale,
                 h_crop=h_crop, w_crop=w_crop)
        else:
            # Declare default valid mask (all ones)
            valid_mask = np.ones(image_size)

        # Initialize the empty output dict
        outputs = {}
        # Convert to tensor and return the results
        to_tensor = transforms.ToTensor()

        # Check homographic augmentation
        warp = (self.config["augmentation"]["homographic"]["enable"]
                and disable_homoaug == False)
        if warp:
            homo_trans = self.get_homo_transform()
            # Perform homographic transform
            if H1 is None:
                homo_outputs = homo_trans(image, junctions, line_map,
                                            valid_mask=valid_mask)
            else:
                homo_outputs = homo_trans(
                    image, junctions, line_map, homo=H1, scale=H1_scale,
                    valid_mask=valid_mask)
            homography_mat = homo_outputs["homo"]
            
            # Give the warp of the other view
            if H1 is None:
                H1 = homo_outputs["homo"]

        # Sample points along each line segments for the descriptor
        if desc_training:
            line_points, line_indices = self.get_line_points(
                junctions, line_map, H1=H1, H2=H2,
                img_size=image_size, warp=warp)

        # Record the warped results
        if warp:
            junctions = homo_outputs["junctions"]  # Should be HW format
            image = homo_outputs["warped_image"]
            line_map = homo_outputs["line_map"]
            valid_mask = homo_outputs["valid_mask"]  # Same for pos and neg
            heatmap = homo_outputs["warped_heatmap"]
            
            # Optionally put warping information first.
            if not numpy:
                outputs["homography_mat"] = to_tensor(
                    homography_mat).to(torch.float32)[0, ...]
            else:
                outputs["homography_mat"] = homography_mat.astype(np.float32)

        junction_map = self.junc_to_junc_map(junctions, image_size)
        
        if not numpy:
            outputs.update({
                "image": to_tensor(image),
                "junctions": to_tensor(junctions).to(torch.float32)[0, ...],
                "junction_map": to_tensor(junction_map).to(torch.int),
                "line_map": to_tensor(line_map).to(torch.int32)[0, ...],
                "heatmap": to_tensor(heatmap).to(torch.int32),
                "valid_mask": to_tensor(valid_mask).to(torch.int32)
            })
            if desc_training:
                outputs.update({
                    "line_points": to_tensor(
                        line_points).to(torch.float32)[0],
                    "line_indices": torch.tensor(line_indices,
                                                 dtype=torch.int)
                })
        else:
            outputs.update({
                "image": image,
                "junctions": junctions.astype(np.float32),
                "junction_map": junction_map.astype(np.int32),
                "line_map": line_map.astype(np.int32),
                "heatmap": heatmap.astype(np.int32),
                "valid_mask": valid_mask.astype(np.int32)
            })
            if desc_training:
                outputs.update({
                    "line_points": line_points.astype(np.float32),
                    "line_indices": line_indices.astype(int)
                })
        
        return outputs
    
    def preprocessing_exported_paired_desc(self, data, numpy=False, scale=1.):
        """ Train preprocessing for paired data for the exported labels
            for descriptor training. """
        outputs = {}

        # Define the random crop for scaling if necessary
        h_crop, w_crop = 0, 0
        if scale > 1:
            H, W = self.config["preprocessing"]["resize"]
            H_scale, W_scale = round(H * scale), round(W * scale)
            if H_scale > H:
                h_crop = np.random.randint(H_scale - H)
            if W_scale > W:
                w_crop = np.random.randint(W_scale - W)
        
        # Sample ref homography first
        homo_config = self.config["augmentation"]["homographic"]["params"]
        image_shape = self.config["preprocessing"]["resize"]
        ref_H, ref_scale = homoaug.sample_homography(image_shape,
                                                     **homo_config)

        # Data for target view (All augmentation)
        target_data = self.train_preprocessing_exported(
            data, numpy=numpy, desc_training=True, H1=None, H2=ref_H,
            scale=scale, h_crop=h_crop, w_crop=w_crop)

        # Data for reference view (No homographical augmentation)
        ref_data = self.train_preprocessing_exported(
            data, numpy=numpy, desc_training=True, H1=ref_H,
            H1_scale=ref_scale, H2=target_data['homography_mat'].numpy(),
            scale=scale, h_crop=h_crop, w_crop=w_crop)

        # Spread ref data
        for key, val in ref_data.items():
            outputs["ref_" + key] = val
        
        # Spread target data
        for key, val in target_data.items():
            outputs["target_" + key] = val
        
        return outputs

    def test_preprocessing_exported(self, data, numpy=False):
        """ Test preprocessing for the exported labels. """
        data = copy.deepcopy(data)
        # Fetch the corresponding entries
        image = data["image"]
        junctions = data["junctions"]
        line_map = data["line_map"]      
        image_size = image.shape[:2]

        # Resize the image before photometric and homographical augmentations
        if not(list(image_size) == self.config["preprocessing"]["resize"]):
            # Resize the image and the point location.
            size_old = list(image.shape)[:2] # Only H and W dimensions

            image = cv2.resize(
                image, tuple(self.config['preprocessing']['resize'][::-1]),
                interpolation=cv2.INTER_LINEAR)
            image = np.array(image, dtype=np.uint8)

            # # In HW format
            # junctions = (junctions * np.array(
            #     self.config['preprocessing']['resize'], np.float)
            #              / np.array(size_old, np.float))

        # Optionally convert the image to grayscale
        if self.config["gray_scale"]:
            image = (color.rgb2gray(image) * 255.).astype(np.uint8)

        # Still need to normalize image
        image_transform = photoaug.normalize_image()
        image = image_transform(image)

        # Generate the line heatmap after post-processing
        junctions_xy = np.flip(np.round(junctions).astype(np.int32), axis=1)
        image_size = image.shape[:2]
        heatmap = get_line_heatmap(junctions_xy, line_map, image_size)
        
        # Declare default valid mask (all ones)
        valid_mask = np.ones(image_size)

        junction_map = self.junc_to_junc_map(junctions, image_size)

        # Convert to tensor and return the results
        to_tensor = transforms.ToTensor()
        if not numpy:
            outputs = {
                "image": to_tensor(image),
                "junctions": to_tensor(junctions).to(torch.float32)[0, ...],
                "junction_map": to_tensor(junction_map).to(torch.int),
                "line_map": to_tensor(line_map).to(torch.int32)[0, ...],
                "heatmap": to_tensor(heatmap).to(torch.int32),
                "valid_mask": to_tensor(valid_mask).to(torch.int32)
            }
        else:
            outputs = {
                "image": image,
                "junctions": junctions.astype(np.float32),
                "junction_map": junction_map.astype(np.int32),
                "line_map": line_map.astype(np.int32),
                "heatmap": heatmap.astype(np.int32),
                "valid_mask": valid_mask.astype(np.int32)
            }
        
        return outputs

    def __len__(self):
        return self.dataset_length
    
    def get_data_from_key(self, file_key):
        """ Get data from file_key. """
        # Check key exists
        if not file_key in self.filename_dataset.keys():
            raise ValueError(
        "[Error] the specified key is not in the dataset.")
        
        # Get the data paths
        data_path = self.filename_dataset[file_key]
        # Read in the image and npz labels
        data = self.get_data_from_path(data_path)

        # Perform transform and augmentation
        if (self.mode == "train"
            or self.config["add_augmentation_to_all_splits"]):
            data = self.train_preprocessing(data, numpy=True)
        else:
            data = self.test_preprocessing(data, numpy=True)
        
        # Add file key to the output
        data["file_key"] = file_key
        
        return data
    
    def __getitem__(self, idx):
        """Return data
        file_key: str, keys used to retrieve data from the filename dataset.
        image: torch.float, C*H*W range 0~1,
        junctions: torch.float, N*2,
        junction_map: torch.int32, 1*H*W range 0 or 1,
        line_map: torch.int32, N*N range 0 or 1,
        heatmap: torch.int32, 1*H*W range 0 or 1,
        valid_mask: torch.int32, 1*H*W range 0 or 1
        """
        # Get the corresponding datapoint and contents from filename dataset
        file_key = self.datapoints[idx]
        data_path = self.filename_dataset[file_key]
        # Read in the image and npz labels
        data = self.get_data_from_path(data_path)

        if self.gt_source:
            with h5py.File(self.gt_source, "r") as f:
                exported_label = parse_h5_data(f[file_key])
            
            data["junctions"] = exported_label["junctions"]
            data["line_map"] = exported_label["line_map"]
        
        # Perform transform and augmentation
        return_type = self.config.get("return_type", "single")
        if self.gt_source is None:
            # For export only
            data = self.export_preprocessing(data)
        elif (self.mode == "train"
              or self.config["add_augmentation_to_all_splits"]):
            # Perform random scaling first
            if self.config["augmentation"]["random_scaling"]["enable"]:
                scale_range = self.config["augmentation"]["random_scaling"]["range"]
                # Decide the scaling
                scale = np.random.uniform(min(scale_range), max(scale_range))
            else:
                scale = 1.
            if self.mode == "train" and return_type == "paired_desc":
                data = self.preprocessing_exported_paired_desc(data,
                                                               scale=scale)
            else:
                data = self.train_preprocessing_exported(data, scale=scale)
        else:
            if return_type == "paired_desc":
                data = self.preprocessing_exported_paired_desc(data)
            else:
                data = self.test_preprocessing_exported(data)
        
        # Add file key to the output
        data["file_key"] = file_key
        
        return data

