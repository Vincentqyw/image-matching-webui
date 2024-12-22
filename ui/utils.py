import os
import pickle
import random
import shutil
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import poselib
import psutil
from PIL import Image

sys.path.append(str(Path(__file__).parents[1]))

from hloc import (
    DEVICE,
    extract_features,
    extractors,
    logger,
    match_dense,
    match_features,
    matchers,
)
from hloc.utils.base_model import dynamic_load
from ui.viz import display_keypoints, display_matches, fig2im, plot_images

warnings.simplefilter("ignore")

ROOT = Path(__file__).parent.parent
# some default values
DEFAULT_SETTING_THRESHOLD = 0.1
DEFAULT_SETTING_MAX_FEATURES = 2000
DEFAULT_DEFAULT_KEYPOINT_THRESHOLD = 0.01
DEFAULT_ENABLE_RANSAC = True
DEFAULT_RANSAC_METHOD = "CV2_USAC_MAGSAC"
DEFAULT_RANSAC_REPROJ_THRESHOLD = 8
DEFAULT_RANSAC_CONFIDENCE = 0.999
DEFAULT_RANSAC_MAX_ITER = 10000
DEFAULT_MIN_NUM_MATCHES = 4
DEFAULT_MATCHING_THRESHOLD = 0.2
DEFAULT_SETTING_GEOMETRY = "Homography"
GRADIO_VERSION = gr.__version__.split(".")[0]
MATCHER_ZOO = None


class ModelCache:
    def __init__(self, max_memory_size: int = 8):
        self.max_memory_size = max_memory_size
        self.current_memory_size = 0
        self.model_dict = {}
        self.model_timestamps = []

    def cache_model(self, model_key, model_loader_func, model_conf):
        if model_key in self.model_dict:
            self.model_timestamps.remove(model_key)
            self.model_timestamps.append(model_key)
            logger.info(f"Load cached {model_key}")
            return self.model_dict[model_key]

        model = self._load_model_from_disk(model_loader_func, model_conf)
        while self._calculate_model_memory() > self.max_memory_size:
            if len(self.model_timestamps) == 0:
                logger.warn(
                    "RAM: {}GB, MAX RAM: {}GB".format(
                        self._calculate_model_memory(), self.max_memory_size
                    )
                )
                break
            oldest_model_key = self.model_timestamps.pop(0)
            self.current_memory_size = self._calculate_model_memory()
            logger.info(f"Del cached {oldest_model_key}")
            del self.model_dict[oldest_model_key]

        self.model_dict[model_key] = model
        self.model_timestamps.append(model_key)

        self.print_memory_usage()
        logger.info(f"Total cached {list(self.model_dict.keys())}")

        return model

    def _load_model_from_disk(self, model_loader_func, model_conf):
        return model_loader_func(model_conf)

    def _calculate_model_memory(self, verbose=False):
        host_colocation = int(os.environ.get("HOST_COLOCATION", "1"))
        vm = psutil.virtual_memory()
        du = shutil.disk_usage(".")
        if verbose:
            logger.info(
                f"RAM: {vm.used / 1e9:.1f}/{vm.total / host_colocation / 1e9:.1f}GB"
            )
            logger.info(
                f"DISK: {du.used / 1e9:.1f}/{du.total / host_colocation / 1e9:.1f}GB"
            )
        return vm.used / 1e9

    def print_memory_usage(self):
        self._calculate_model_memory(verbose=True)


model_cache = ModelCache()


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_name: The path to the YAML configuration file.

    Returns:
        The configuration dictionary, with string keys and arbitrary values.
    """
    import yaml

    with open(config_name, "r") as stream:
        try:
            config: Dict[str, Any] = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
    return config


def get_matcher_zoo(
    matcher_zoo: Dict[str, Dict[str, Union[str, bool]]],
) -> Dict[str, Dict[str, Union[Callable, bool]]]:
    """
    Restore matcher configurations from a dictionary.

    Args:
        matcher_zoo: A dictionary with the matcher configurations,
            where the configuration is a dictionary as loaded from a YAML file.

    Returns:
        A dictionary with the matcher configurations, where the configuration is
            a function or a function instead of a string.
    """
    matcher_zoo_restored = {}
    for k, v in matcher_zoo.items():
        matcher_zoo_restored[k] = parse_match_config(v)
    return matcher_zoo_restored


def parse_match_config(conf):
    if conf["dense"]:
        return {
            "matcher": match_dense.confs.get(conf["matcher"]),
            "dense": True,
        }
    else:
        return {
            "feature": extract_features.confs.get(conf["feature"]),
            "matcher": match_features.confs.get(conf["matcher"]),
            "dense": False,
        }


def get_model(match_conf: Dict[str, Any]):
    """
    Load a matcher model from the provided configuration.

    Args:
        match_conf: A dictionary containing the model configuration.

    Returns:
        A matcher model instance.
    """
    Model = dynamic_load(matchers, match_conf["model"]["name"])
    model = Model(match_conf["model"]).eval().to(DEVICE)
    return model


def get_feature_model(conf: Dict[str, Dict[str, Any]]):
    """
    Load a feature extraction model from the provided configuration.

    Args:
        conf: A dictionary containing the model configuration.

    Returns:
        A feature extraction model instance.
    """
    Model = dynamic_load(extractors, conf["model"]["name"])
    model = Model(conf["model"]).eval().to(DEVICE)
    return model


def gen_examples():
    random.seed(1)
    example_matchers = [
        "disk+lightglue",
        "xfeat(sparse)",
        "dedode",
        "loftr",
        "disk",
        "RoMa",
        "d2net",
        "aspanformer",
        "topicfm",
        "superpoint+superglue",
        "superpoint+lightglue",
        "superpoint+mnn",
        "disk",
    ]

    def distribute_elements(A, B):
        new_B = np.array(B, copy=True).flatten()
        np.random.shuffle(new_B)
        new_B = np.resize(new_B, len(A))
        np.random.shuffle(new_B)
        return new_B.tolist()

    # normal examples
    def gen_images_pairs(count: int = 5):
        path = str(ROOT / "datasets/sacre_coeur/mapping")
        imgs_list = [
            os.path.join(path, file)
            for file in os.listdir(path)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        pairs = list(combinations(imgs_list, 2))
        if len(pairs) < count:
            count = len(pairs)
        selected = random.sample(range(len(pairs)), count)
        return [pairs[i] for i in selected]

    # rotated examples
    def gen_rot_image_pairs(count: int = 5):
        path = ROOT / "datasets/sacre_coeur/mapping"
        path_rot = ROOT / "datasets/sacre_coeur/mapping_rot"
        rot_list = [45, 180, 90, 225, 270]
        pairs = []
        for file in os.listdir(path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                for rot in rot_list:
                    file_rot = "{}_rot{}.jpg".format(Path(file).stem, rot)
                    if (path_rot / file_rot).exists():
                        pairs.append(
                            [
                                path / file,
                                path_rot / file_rot,
                            ]
                        )
        if len(pairs) < count:
            count = len(pairs)
        selected = random.sample(range(len(pairs)), count)
        return [pairs[i] for i in selected]

    def gen_scale_image_pairs(count: int = 5):
        path = ROOT / "datasets/sacre_coeur/mapping"
        path_scale = ROOT / "datasets/sacre_coeur/mapping_scale"
        scale_list = [0.3, 0.5]
        pairs = []
        for file in os.listdir(path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                for scale in scale_list:
                    file_scale = "{}_scale{}.jpg".format(Path(file).stem, scale)
                    if (path_scale / file_scale).exists():
                        pairs.append(
                            [
                                path / file,
                                path_scale / file_scale,
                            ]
                        )
        if len(pairs) < count:
            count = len(pairs)
        selected = random.sample(range(len(pairs)), count)
        return [pairs[i] for i in selected]

    # extramely hard examples
    def gen_image_pairs_wxbs(count: int = None):
        prefix = "datasets/wxbs_benchmark/.WxBS/v1.1"
        wxbs_path = ROOT / prefix
        pairs = []
        for catg in os.listdir(wxbs_path):
            catg_path = wxbs_path / catg
            if not catg_path.is_dir():
                continue
            for scene in os.listdir(catg_path):
                scene_path = catg_path / scene
                if not scene_path.is_dir():
                    continue
                img1_path = scene_path / "01.png"
                img2_path = scene_path / "02.png"
                if img1_path.exists() and img2_path.exists():
                    pairs.append([str(img1_path), str(img2_path)])
        return pairs

    # image pair path
    pairs = gen_images_pairs()
    pairs += gen_rot_image_pairs()
    pairs += gen_scale_image_pairs()
    pairs += gen_image_pairs_wxbs()

    match_setting_threshold = DEFAULT_SETTING_THRESHOLD
    match_setting_max_features = DEFAULT_SETTING_MAX_FEATURES
    detect_keypoints_threshold = DEFAULT_DEFAULT_KEYPOINT_THRESHOLD
    ransac_method = DEFAULT_RANSAC_METHOD
    ransac_reproj_threshold = DEFAULT_RANSAC_REPROJ_THRESHOLD
    ransac_confidence = DEFAULT_RANSAC_CONFIDENCE
    ransac_max_iter = DEFAULT_RANSAC_MAX_ITER
    input_lists = []
    dist_examples = distribute_elements(pairs, example_matchers)
    for pair, mt in zip(pairs, dist_examples):
        input_lists.append(
            [
                pair[0],
                pair[1],
                match_setting_threshold,
                match_setting_max_features,
                detect_keypoints_threshold,
                mt,
                # enable_ransac,
                ransac_method,
                ransac_reproj_threshold,
                ransac_confidence,
                ransac_max_iter,
            ]
        )
    return input_lists


def set_null_pred(feature_type: str, pred: dict):
    if feature_type == "KEYPOINT":
        pred["mmkeypoints0_orig"] = np.array([])
        pred["mmkeypoints1_orig"] = np.array([])
        pred["mmconf"] = np.array([])
    elif feature_type == "LINE":
        pred["mline_keypoints0_orig"] = np.array([])
        pred["mline_keypoints1_orig"] = np.array([])
    pred["H"] = None
    pred["geom_info"] = {}
    return pred


def _filter_matches_opencv(
    kp0: np.ndarray,
    kp1: np.ndarray,
    method: int = cv2.RANSAC,
    reproj_threshold: float = 3.0,
    confidence: float = 0.99,
    max_iter: int = 2000,
    geometry_type: str = "Homography",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters matches between two sets of keypoints using OpenCV's findHomography.

    Args:
        kp0 (np.ndarray): Array of keypoints from the first image.
        kp1 (np.ndarray): Array of keypoints from the second image.
        method (int, optional): RANSAC method. Defaults to "cv2.RANSAC".
        reproj_threshold (float, optional): RANSAC reprojection threshold. Defaults to 3.0.
        confidence (float, optional): RANSAC confidence. Defaults to 0.99.
        max_iter (int, optional): RANSAC maximum iterations. Defaults to 2000.
        geometry_type (str, optional): Type of geometry. Defaults to "Homography".

    Returns:
        Tuple[np.ndarray, np.ndarray]: Homography matrix and mask.
    """
    if geometry_type == "Homography":
        M, mask = cv2.findHomography(
            kp0,
            kp1,
            method=method,
            ransacReprojThreshold=reproj_threshold,
            confidence=confidence,
            maxIters=max_iter,
        )
    elif geometry_type == "Fundamental":
        M, mask = cv2.findFundamentalMat(
            kp0,
            kp1,
            method=method,
            ransacReprojThreshold=reproj_threshold,
            confidence=confidence,
            maxIters=max_iter,
        )
    mask = np.array(mask.ravel().astype("bool"), dtype="bool")
    return M, mask


def _filter_matches_poselib(
    kp0: np.ndarray,
    kp1: np.ndarray,
    method: int = None,  # not used
    reproj_threshold: float = 3,
    confidence: float = 0.99,
    max_iter: int = 2000,
    geometry_type: str = "Homography",
) -> dict:
    """
    Filters matches between two sets of keypoints using the poselib library.

    Args:
        kp0 (np.ndarray): Array of keypoints from the first image.
        kp1 (np.ndarray): Array of keypoints from the second image.
        method (str, optional): RANSAC method. Defaults to "RANSAC".
        reproj_threshold (float, optional): RANSAC reprojection threshold. Defaults to 3.
        confidence (float, optional): RANSAC confidence. Defaults to 0.99.
        max_iter (int, optional): RANSAC maximum iterations. Defaults to 2000.
        geometry_type (str, optional): Type of geometry. Defaults to "Homography".

    Returns:
        dict: Information about the homography estimation.
    """
    ransac_options = {
        "max_iterations": max_iter,
        # "min_iterations":  min_iter,
        "success_prob": confidence,
        "max_reproj_error": reproj_threshold,
        # "progressive_sampling": args.sampler.lower() == 'prosac'
    }

    if geometry_type == "Homography":
        M, info = poselib.estimate_homography(kp0, kp1, ransac_options)
    elif geometry_type == "Fundamental":
        M, info = poselib.estimate_fundamental(kp0, kp1, ransac_options)
    else:
        raise NotImplementedError

    return M, np.array(info["inliers"])


def proc_ransac_matches(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    ransac_method: str = DEFAULT_RANSAC_METHOD,
    ransac_reproj_threshold: float = 3.0,
    ransac_confidence: float = 0.99,
    ransac_max_iter: int = 2000,
    geometry_type: str = "Homography",
):
    if ransac_method.startswith("CV2"):
        logger.info(f"ransac_method: {ransac_method}, geometry_type: {geometry_type}")
        return _filter_matches_opencv(
            mkpts0,
            mkpts1,
            ransac_zoo[ransac_method],
            ransac_reproj_threshold,
            ransac_confidence,
            ransac_max_iter,
            geometry_type,
        )
    elif ransac_method.startswith("POSELIB"):
        logger.info(f"ransac_method: {ransac_method}, geometry_type: {geometry_type}")
        return _filter_matches_poselib(
            mkpts0,
            mkpts1,
            None,
            ransac_reproj_threshold,
            ransac_confidence,
            ransac_max_iter,
            geometry_type,
        )
    else:
        raise NotImplementedError


def filter_matches(
    pred: Dict[str, Any],
    ransac_method: str = DEFAULT_RANSAC_METHOD,
    ransac_reproj_threshold: float = DEFAULT_RANSAC_REPROJ_THRESHOLD,
    ransac_confidence: float = DEFAULT_RANSAC_CONFIDENCE,
    ransac_max_iter: int = DEFAULT_RANSAC_MAX_ITER,
    ransac_estimator: str = None,
):
    """
    Filter matches using RANSAC. If keypoints are available, filter by keypoints.
    If lines are available, filter by lines. If both keypoints and lines are
    available, filter by keypoints.

    Args:
        pred (Dict[str, Any]): dict of matches, including original keypoints.
        ransac_method (str, optional): RANSAC method. Defaults to DEFAULT_RANSAC_METHOD.
        ransac_reproj_threshold (float, optional): RANSAC reprojection threshold. Defaults to DEFAULT_RANSAC_REPROJ_THRESHOLD.
        ransac_confidence (float, optional): RANSAC confidence. Defaults to DEFAULT_RANSAC_CONFIDENCE.
        ransac_max_iter (int, optional): RANSAC maximum iterations. Defaults to DEFAULT_RANSAC_MAX_ITER.

    Returns:
        Dict[str, Any]: filtered matches.
    """
    mkpts0: Optional[np.ndarray] = None
    mkpts1: Optional[np.ndarray] = None
    feature_type: Optional[str] = None
    if "mkeypoints0_orig" in pred.keys() and "mkeypoints1_orig" in pred.keys():
        mkpts0 = pred["mkeypoints0_orig"]
        mkpts1 = pred["mkeypoints1_orig"]
        feature_type = "KEYPOINT"
    elif (
        "line_keypoints0_orig" in pred.keys() and "line_keypoints1_orig" in pred.keys()
    ):
        mkpts0 = pred["line_keypoints0_orig"]
        mkpts1 = pred["line_keypoints1_orig"]
        feature_type = "LINE"
    else:
        return set_null_pred(feature_type, pred)
    if mkpts0 is None or mkpts0 is None:
        return set_null_pred(feature_type, pred)
    if ransac_method not in ransac_zoo.keys():
        ransac_method = DEFAULT_RANSAC_METHOD

    if len(mkpts0) < DEFAULT_MIN_NUM_MATCHES:
        return set_null_pred(feature_type, pred)

    geom_info = compute_geometry(
        pred,
        ransac_method=ransac_method,
        ransac_reproj_threshold=ransac_reproj_threshold,
        ransac_confidence=ransac_confidence,
        ransac_max_iter=ransac_max_iter,
    )

    if "Homography" in geom_info.keys():
        mask = geom_info["mask_h"]
        if feature_type == "KEYPOINT":
            pred["mmkeypoints0_orig"] = mkpts0[mask]
            pred["mmkeypoints1_orig"] = mkpts1[mask]
            pred["mmconf"] = pred["mconf"][mask]
        elif feature_type == "LINE":
            pred["mline_keypoints0_orig"] = mkpts0[mask]
            pred["mline_keypoints1_orig"] = mkpts1[mask]
        pred["H"] = np.array(geom_info["Homography"])
    else:
        set_null_pred(feature_type, pred)
    # do not show mask
    geom_info.pop("mask_h", None)
    geom_info.pop("mask_f", None)
    pred["geom_info"] = geom_info
    return pred


def compute_geometry(
    pred: Dict[str, Any],
    ransac_method: str = DEFAULT_RANSAC_METHOD,
    ransac_reproj_threshold: float = DEFAULT_RANSAC_REPROJ_THRESHOLD,
    ransac_confidence: float = DEFAULT_RANSAC_CONFIDENCE,
    ransac_max_iter: int = DEFAULT_RANSAC_MAX_ITER,
) -> Dict[str, List[float]]:
    """
    Compute geometric information of matches, including Fundamental matrix,
    Homography matrix, and rectification matrices (if available).

    Args:
        pred (Dict[str, Any]): dict of matches, including original keypoints.
        ransac_method (str, optional): RANSAC method. Defaults to DEFAULT_RANSAC_METHOD.
        ransac_reproj_threshold (float, optional): RANSAC reprojection threshold. Defaults to DEFAULT_RANSAC_REPROJ_THRESHOLD.
        ransac_confidence (float, optional): RANSAC confidence. Defaults to DEFAULT_RANSAC_CONFIDENCE.
        ransac_max_iter (int, optional): RANSAC maximum iterations. Defaults to DEFAULT_RANSAC_MAX_ITER.

    Returns:
        Dict[str, List[float]]: geometric information in form of a dict.
    """
    mkpts0: Optional[np.ndarray] = None
    mkpts1: Optional[np.ndarray] = None

    if "mkeypoints0_orig" in pred.keys() and "mkeypoints1_orig" in pred.keys():
        mkpts0 = pred["mkeypoints0_orig"]
        mkpts1 = pred["mkeypoints1_orig"]
    elif (
        "line_keypoints0_orig" in pred.keys() and "line_keypoints1_orig" in pred.keys()
    ):
        mkpts0 = pred["line_keypoints0_orig"]
        mkpts1 = pred["line_keypoints1_orig"]

    if mkpts0 is not None and mkpts1 is not None:
        if len(mkpts0) < 2 * DEFAULT_MIN_NUM_MATCHES:
            return {}
        geo_info: Dict[str, List[float]] = {}

        F, mask_f = proc_ransac_matches(
            mkpts0,
            mkpts1,
            ransac_method,
            ransac_reproj_threshold,
            ransac_confidence,
            ransac_max_iter,
            geometry_type="Fundamental",
        )

        if F is not None:
            geo_info["Fundamental"] = F.tolist()
            geo_info["mask_f"] = mask_f
        H, mask_h = proc_ransac_matches(
            mkpts1,
            mkpts0,
            ransac_method,
            ransac_reproj_threshold,
            ransac_confidence,
            ransac_max_iter,
            geometry_type="Homography",
        )

        h0, w0, _ = pred["image0_orig"].shape
        if H is not None:
            geo_info["Homography"] = H.tolist()
            geo_info["mask_h"] = mask_h
            try:
                _, H1, H2 = cv2.stereoRectifyUncalibrated(
                    mkpts0.reshape(-1, 2),
                    mkpts1.reshape(-1, 2),
                    F,
                    imgSize=(w0, h0),
                )
                geo_info["H1"] = H1.tolist()
                geo_info["H2"] = H2.tolist()
            except cv2.error as e:
                logger.error(f"StereoRectifyUncalibrated failed, skip! error: {e}")
        return geo_info
    else:
        return {}


def wrap_images(
    img0: np.ndarray,
    img1: np.ndarray,
    geo_info: Optional[Dict[str, List[float]]],
    geom_type: str,
) -> Tuple[Optional[str], Optional[Dict[str, List[float]]]]:
    """
    Wraps the images based on the geometric transformation used to align them.

    Args:
        img0: numpy array representing the first image.
        img1: numpy array representing the second image.
        geo_info: dictionary containing the geometric transformation information.
        geom_type: type of geometric transformation used to align the images.

    Returns:
        A tuple containing a base64 encoded image string and a dictionary with the transformation matrix.
    """
    h0, w0, _ = img0.shape
    h1, w1, _ = img1.shape
    if geo_info is not None and len(geo_info) != 0:
        rectified_image0 = img0
        rectified_image1 = None
        if "Homography" not in geo_info:
            logger.warning(f"{geom_type} not exist, maybe too less matches")
            return None, None

        H = np.array(geo_info["Homography"])

        title: List[str] = []
        if geom_type == "Homography":
            rectified_image1 = cv2.warpPerspective(img1, H, (w0, h0))
            title = ["Image 0", "Image 1 - warped"]
        elif geom_type == "Fundamental":
            if geom_type not in geo_info:
                logger.warning(f"{geom_type} not exist, maybe too less matches")
                return None, None
            else:
                H1, H2 = np.array(geo_info["H1"]), np.array(geo_info["H2"])
                rectified_image0 = cv2.warpPerspective(img0, H1, (w0, h0))
                rectified_image1 = cv2.warpPerspective(img1, H2, (w1, h1))
                title = ["Image 0 - warped", "Image 1 - warped"]
        else:
            print("Error: Unknown geometry type")
        fig = plot_images(
            [rectified_image0.squeeze(), rectified_image1.squeeze()],
            title,
            dpi=300,
        )
        return fig2im(fig), rectified_image1
    else:
        return None, None


def generate_warp_images(
    input_image0: np.ndarray,
    input_image1: np.ndarray,
    matches_info: Dict[str, Any],
    choice: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Changes the estimate of the geometric transformation used to align the images.

    Args:
        input_image0: First input image.
        input_image1: Second input image.
        matches_info: Dictionary containing information about the matches.
        choice: Type of geometric transformation to use ('Homography' or 'Fundamental') or 'No' to disable.

    Returns:
        A tuple containing the updated images and the warpped images.
    """
    if (
        matches_info is None
        or len(matches_info) < 1
        or "geom_info" not in matches_info.keys()
    ):
        return None, None
    geom_info = matches_info["geom_info"]
    warped_image = None
    if choice != "No":
        wrapped_image_pair, warped_image = wrap_images(
            input_image0, input_image1, geom_info, choice
        )
        return wrapped_image_pair, warped_image
    else:
        return None, None


def send_to_match(state_cache: Dict[str, Any]):
    """
    Send the state cache to the match function.

    Args:
        state_cache (Dict[str, Any]): Current state of the app.

    Returns:
        None
    """
    if state_cache:
        return (
            state_cache["image0_orig"],
            state_cache["wrapped_image"],
        )
    else:
        return None, None


def run_ransac(
    state_cache: Dict[str, Any],
    choice_geometry_type: str,
    ransac_method: str = DEFAULT_RANSAC_METHOD,
    ransac_reproj_threshold: int = DEFAULT_RANSAC_REPROJ_THRESHOLD,
    ransac_confidence: float = DEFAULT_RANSAC_CONFIDENCE,
    ransac_max_iter: int = DEFAULT_RANSAC_MAX_ITER,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, int]]]:
    """
    Run RANSAC matches and return the output images and the number of matches.

    Args:
        state_cache (Dict[str, Any]): Current state of the app, including the matches.
        ransac_method (str, optional): RANSAC method. Defaults to DEFAULT_RANSAC_METHOD.
        ransac_reproj_threshold (int, optional): RANSAC reprojection threshold. Defaults to DEFAULT_RANSAC_REPROJ_THRESHOLD.
        ransac_confidence (float, optional): RANSAC confidence. Defaults to DEFAULT_RANSAC_CONFIDENCE.
        ransac_max_iter (int, optional): RANSAC maximum iterations. Defaults to DEFAULT_RANSAC_MAX_ITER.

    Returns:
        Tuple[Optional[np.ndarray], Optional[Dict[str, int]]]: Tuple containing the output images and the number of matches.
    """
    if not state_cache:
        logger.info("Run Match first before Rerun RANSAC")
        gr.Warning("Run Match first before Rerun RANSAC")
        return None, None
    t1 = time.time()
    logger.info(
        f"Run RANSAC matches using: {ransac_method} with threshold: {ransac_reproj_threshold}"
    )
    logger.info(
        f"Run RANSAC matches using: {ransac_confidence} with iter: {ransac_max_iter}"
    )
    # if enable_ransac:
    filter_matches(
        state_cache,
        ransac_method=ransac_method,
        ransac_reproj_threshold=ransac_reproj_threshold,
        ransac_confidence=ransac_confidence,
        ransac_max_iter=ransac_max_iter,
    )
    logger.info(f"RANSAC matches done using: {time.time()-t1:.3f}s")
    t1 = time.time()

    # plot images with ransac matches
    titles = [
        "Image 0 - Ransac matched keypoints",
        "Image 1 - Ransac matched keypoints",
    ]
    output_matches_ransac, num_matches_ransac = display_matches(
        state_cache, titles=titles, tag="KPTS_RANSAC"
    )
    logger.info(f"Display matches done using: {time.time()-t1:.3f}s")
    t1 = time.time()

    # compute warp images
    output_wrapped, warped_image = generate_warp_images(
        state_cache["image0_orig"],
        state_cache["image1_orig"],
        state_cache,
        choice_geometry_type,
    )
    plt.close("all")

    num_matches_raw = state_cache["num_matches_raw"]
    state_cache["wrapped_image"] = warped_image

    # tmp_state_cache = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    tmp_state_cache = "output.pkl"
    with open(tmp_state_cache, "wb") as f:
        pickle.dump(state_cache, f)

    logger.info("Dump results done!")

    return (
        output_matches_ransac,
        {
            "num_matches_raw": num_matches_raw,
            "num_matches_ransac": num_matches_ransac,
        },
        output_wrapped,
        tmp_state_cache,
    )


def run_matching(
    image0: np.ndarray,
    image1: np.ndarray,
    match_threshold: float,
    extract_max_keypoints: int,
    keypoint_threshold: float,
    key: str,
    ransac_method: str = DEFAULT_RANSAC_METHOD,
    ransac_reproj_threshold: int = DEFAULT_RANSAC_REPROJ_THRESHOLD,
    ransac_confidence: float = DEFAULT_RANSAC_CONFIDENCE,
    ransac_max_iter: int = DEFAULT_RANSAC_MAX_ITER,
    choice_geometry_type: str = DEFAULT_SETTING_GEOMETRY,
    matcher_zoo: Dict[str, Any] = None,
    force_resize: bool = False,
    image_width: int = 640,
    image_height: int = 480,
    use_cached_model: bool = False,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, int],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, float]],
    np.ndarray,
]:
    """Match two images using the given parameters.

    Args:
        image0 (np.ndarray): RGB image 0.
        image1 (np.ndarray): RGB image 1.
        match_threshold (float): match threshold.
        extract_max_keypoints (int): number of keypoints to extract.
        keypoint_threshold (float): keypoint threshold.
        key (str): key of the model to use.
        ransac_method (str, optional): RANSAC method to use.
        ransac_reproj_threshold (int, optional): RANSAC reprojection threshold.
        ransac_confidence (float, optional): RANSAC confidence level.
        ransac_max_iter (int, optional): RANSAC maximum number of iterations.
        choice_geometry_type (str, optional): setting of geometry estimation.
        matcher_zoo (Dict[str, Any], optional): matcher zoo. Defaults to None.
        force_resize (bool, optional): force resize. Defaults to False.
        image_width (int, optional): image width. Defaults to 640.
        image_height (int, optional): image height. Defaults to 480.
        use_cached_model (bool, optional): use cached model. Defaults to False.

    Returns:
        tuple:
            - output_keypoints (np.ndarray): image with keypoints.
            - output_matches_raw (np.ndarray): image with raw matches.
            - output_matches_ransac (np.ndarray): image with RANSAC matches.
            - num_matches (Dict[str, int]): number of raw and RANSAC matches.
            - configs (Dict[str, Dict[str, Any]]): match and feature extraction configs.
            - geom_info (Dict[str, Dict[str, float]]): geometry information.
            - output_wrapped (np.ndarray): wrapped images.
    """
    # image0 and image1 is RGB mode
    if image0 is None or image1 is None:
        logger.error(
            "Error: No images found! Please upload two images or select an example."
        )
        raise gr.Error(
            "Error: No images found! Please upload two images or select an example."
        )
    # init output
    output_keypoints = None
    output_matches_raw = None
    output_matches_ransac = None

    # super slow!
    if "roma" in key.lower() and DEVICE == "cpu":
        gr.Info(
            f"Success! Please be patient and allow for about 2-3 minutes."
            f" Due to CPU inference, {key} is quiet slow."
        )
    t0 = time.time()
    model = matcher_zoo[key]
    match_conf = model["matcher"]
    # update match config
    match_conf["model"]["match_threshold"] = match_threshold
    match_conf["model"]["max_keypoints"] = extract_max_keypoints
    cache_key = "{}_{}".format(key, match_conf["model"]["name"])
    if use_cached_model:
        # because of the model cache, we need to update the config
        matcher = model_cache.cache_model(cache_key, get_model, match_conf)
        matcher.conf["max_keypoints"] = extract_max_keypoints
        matcher.conf["match_threshold"] = match_threshold
        logger.info(f"Loaded cached model {cache_key}")
    else:
        matcher = get_model(match_conf)
    logger.info(f"Loading model using: {time.time()-t0:.3f}s")
    t1 = time.time()

    if model["dense"]:
        if not match_conf["preprocessing"].get("force_resize", False):
            match_conf["preprocessing"]["force_resize"] = force_resize
        else:
            logger.info("preprocessing is already resized")
        if force_resize:
            match_conf["preprocessing"]["height"] = image_height
            match_conf["preprocessing"]["width"] = image_width
            logger.info(f"Force resize to {image_width}x{image_height}")

        pred = match_dense.match_images(
            matcher, image0, image1, match_conf["preprocessing"], device=DEVICE
        )
        del matcher
        extract_conf = None
    else:
        extract_conf = model["feature"]
        # update extract config
        extract_conf["model"]["max_keypoints"] = extract_max_keypoints
        extract_conf["model"]["keypoint_threshold"] = keypoint_threshold
        cache_key = "{}_{}".format(key, extract_conf["model"]["name"])

        if use_cached_model:
            extractor = model_cache.cache_model(
                cache_key, get_feature_model, extract_conf
            )
            # because of the model cache, we need to update the config
            extractor.conf["max_keypoints"] = extract_max_keypoints
            extractor.conf["keypoint_threshold"] = keypoint_threshold
            logger.info(f"Loaded cached model {cache_key}")
        else:
            extractor = get_feature_model(extract_conf)

        if not extract_conf["preprocessing"].get("force_resize", False):
            extract_conf["preprocessing"]["force_resize"] = force_resize
        else:
            logger.info("preprocessing is already resized")
        if force_resize:
            extract_conf["preprocessing"]["height"] = image_height
            extract_conf["preprocessing"]["width"] = image_width
            logger.info(f"Force resize to {image_width}x{image_height}")

        pred0 = extract_features.extract(
            extractor, image0, extract_conf["preprocessing"]
        )
        pred1 = extract_features.extract(
            extractor, image1, extract_conf["preprocessing"]
        )
        pred = match_features.match_images(matcher, pred0, pred1)
        del extractor
    # gr.Info(
    #     f"Matching images done using: {time.time()-t1:.3f}s",
    # )
    logger.info(f"Matching images done using: {time.time()-t1:.3f}s")
    t1 = time.time()

    # plot images with keypoints
    titles = [
        "Image 0 - Keypoints",
        "Image 1 - Keypoints",
    ]
    output_keypoints = display_keypoints(pred, titles=titles)

    # plot images with raw matches
    titles = [
        "Image 0 - Raw matched keypoints",
        "Image 1 - Raw matched keypoints",
    ]
    output_matches_raw, num_matches_raw = display_matches(pred, titles=titles)

    # if enable_ransac:
    filter_matches(
        pred,
        ransac_method=ransac_method,
        ransac_reproj_threshold=ransac_reproj_threshold,
        ransac_confidence=ransac_confidence,
        ransac_max_iter=ransac_max_iter,
    )

    # gr.Info(f"RANSAC matches done using: {time.time()-t1:.3f}s")
    logger.info(f"RANSAC matches done using: {time.time()-t1:.3f}s")
    t1 = time.time()

    # plot images with ransac matches
    titles = [
        "Image 0 - Ransac matched keypoints",
        "Image 1 - Ransac matched keypoints",
    ]
    output_matches_ransac, num_matches_ransac = display_matches(
        pred, titles=titles, tag="KPTS_RANSAC"
    )
    # gr.Info(f"Display matches done using: {time.time()-t1:.3f}s")
    logger.info(f"Display matches done using: {time.time()-t1:.3f}s")

    t1 = time.time()
    # plot wrapped images
    output_wrapped, warped_image = generate_warp_images(
        pred["image0_orig"],
        pred["image1_orig"],
        pred,
        choice_geometry_type,
    )
    plt.close("all")
    # gr.Info(f"In summary, total time: {time.time()-t0:.3f}s")
    logger.info(f"TOTAL time: {time.time()-t0:.3f}s")

    state_cache = pred
    state_cache["num_matches_raw"] = num_matches_raw
    state_cache["num_matches_ransac"] = num_matches_ransac
    state_cache["wrapped_image"] = warped_image

    # tmp_state_cache = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    tmp_state_cache = "output.pkl"
    with open(tmp_state_cache, "wb") as f:
        pickle.dump(state_cache, f)
    logger.info("Dump results done!")
    return (
        output_keypoints,
        output_matches_raw,
        output_matches_ransac,
        {
            "num_raw_matches": num_matches_raw,
            "num_ransac_matches": num_matches_ransac,
        },
        {
            "match_conf": match_conf,
            "extractor_conf": extract_conf,
        },
        {
            "geom_info": pred.get("geom_info", {}),
        },
        output_wrapped,
        state_cache,
        tmp_state_cache,
    )


# @ref: https://docs.opencv.org/4.x/d0/d74/md__build_4_x-contrib_docs-lin64_opencv_doc_tutorials_calib3d_usac.html
# AND: https://opencv.org/blog/2021/06/09/evaluating-opencvs-new-ransacs
ransac_zoo = {
    "POSELIB": "LO-RANSAC",
    "CV2_RANSAC": cv2.RANSAC,
    "CV2_USAC_MAGSAC": cv2.USAC_MAGSAC,
    "CV2_USAC_DEFAULT": cv2.USAC_DEFAULT,
    "CV2_USAC_FM_8PTS": cv2.USAC_FM_8PTS,
    "CV2_USAC_PROSAC": cv2.USAC_PROSAC,
    "CV2_USAC_FAST": cv2.USAC_FAST,
    "CV2_USAC_ACCURATE": cv2.USAC_ACCURATE,
    "CV2_USAC_PARALLEL": cv2.USAC_PARALLEL,
}


def rotate_image(input_path, degrees, output_path):
    img = Image.open(input_path)
    img_rotated = img.rotate(-degrees)
    img_rotated.save(output_path)


def scale_image(input_path, scale_factor, output_path):
    img = Image.open(input_path)
    width, height = img.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    new_img = Image.new("RGB", (width, height), (0, 0, 0))
    img_resized = img.resize((new_width, new_height))
    position = ((width - new_width) // 2, (height - new_height) // 2)
    new_img.paste(img_resized, position)
    new_img.save(output_path)
