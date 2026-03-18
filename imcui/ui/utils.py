import os
import pickle
import random
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datasets import load_dataset

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import poselib
from PIL import Image
from vismatch import get_matcher, available_models

# Simple logger
import logging

logger = logging.getLogger("imcui")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s %(name)s %(levelname)s] %(message)s")
    )
    logger.addHandler(handler)

# Constants
DATASETS_REPO_ID = "Realcat/imcui_datasets"
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

from .viz import display_keypoints, display_matches, fig2im, plot_images
from .modelcache import ARCSizeAwareModelCache as ModelCache

warnings.simplefilter("ignore")

ROOT = Path(__file__).parents[1]
# some default values
DEFAULT_SETTING_THRESHOLD = 0.1
DEFAULT_SETTING_MAX_FEATURES = 2000
DEFAULT_DEFAULT_KEYPOINT_THRESHOLD = 0.01
DEFAULT_ENABLE_RANSAC = True
DEFAULT_RANSAC_METHOD = "CV2_USAC_MAGSAC"
DEFAULT_RANSAC_REPROJ_THRESHOLD = 8
DEFAULT_RANSAC_CONFIDENCE = 0.9999
DEFAULT_RANSAC_MAX_ITER = 10000
DEFAULT_MIN_NUM_MATCHES = 4
DEFAULT_MATCHING_THRESHOLD = 0.2
DEFAULT_SETTING_GEOMETRY = "Homography"
GRADIO_VERSION = gr.__version__.split(".")[0]
MATCHER_ZOO = None


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
    matcher_zoo: Optional[Dict[str, Dict[str, Union[str, bool]]]] = None,
) -> Dict[str, Dict[str, Union[Callable, bool]]]:
    """
    Build matcher zoo from vismatch's available_models.

    If matcher_zoo is provided (from config), use it for backward compatibility.
    Otherwise, dynamically load all available models from vismatch.

    Args:
        matcher_zoo: Optional dict from config. If None, use vismatch.available_models.

    Returns:
        A dictionary mapping model keys to their configurations.
    """
    if matcher_zoo is None:
        # Dynamically load all available models from vismatch
        matcher_zoo_restored = {}
        # available_models is a list, not a function
        for model_name in available_models:
            # Use the model name as the key
            matcher_zoo_restored[model_name] = {
                "model_name": model_name,
                "info": {},
            }
        return matcher_zoo_restored
    else:
        # Backward compatibility: use config-based matcher_zoo
        matcher_zoo_restored = {}
        for k, v in matcher_zoo.items():
            matcher_zoo_restored[k] = parse_match_config(v)
        return matcher_zoo_restored


def parse_match_config(conf):
    """
    Parse match config for vismatch.

    For vismatch, we just store the model name directly.
    """
    return {
        "model_name": conf.get("matcher", "superpoint-lightglue"),
        "info": conf.get("info", {}),
    }


def get_available_model_names() -> List[str]:
    """
    Get list of all available model names from vismatch.

    Returns:
        List of model names (strings).
    """
    return available_models


def get_model(match_conf: Dict[str, Any]):
    """
    Load a matcher model from the provided configuration using vismatch.

    Args:
        match_conf: A dictionary containing the model configuration.
            - model_name: str - name of the matcher
            - max_num_keypoints: int - maximum number of keypoints
            - threshold: float - match threshold (for dense matchers)
            - keypoint_threshold: float - keypoint detection threshold (for some matchers)

    Returns:
        A matcher model instance.
    """
    model_name = match_conf.get("model_name", "superpoint-lightglue")
    max_num_keypoints = match_conf.get("max_num_keypoints", 2048)
    threshold = match_conf.get("threshold", 0.1)  # match threshold
    keypoint_threshold = match_conf.get(
        "keypoint_threshold", None
    )  # keypoint detection threshold

    # Build kwargs for vismatch
    kwargs = {
        "max_num_keypoints": max_num_keypoints,
    }

    # Match threshold (for dense matchers like LoFTR, RoMa)
    if threshold is not None:
        kwargs["threshold"] = threshold

    # Keypoint detection threshold (for some matchers like zippypoint, silk)
    if keypoint_threshold is not None:
        kwargs["keypoint_threshold"] = keypoint_threshold

    return get_matcher(model_name, device=DEVICE, **kwargs)


def get_feature_model(conf: Dict[str, Dict[str, Any]]):
    """
    Load a feature extraction model - not needed for vismatch.
    Kept for compatibility but returns None.
    """
    return None


def download_example_images(repo_id, output_dir):
    logger.info(f"Download example dataset from huggingface: {repo_id}")
    dataset = load_dataset(repo_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for example in dataset["train"]:  # Assuming the dataset is in the "train" split
        file_path = example["path"]
        image = example["image"]  # Access the PIL.Image object directly
        full_path = os.path.join(output_dir, file_path)
        Path(os.path.dirname(full_path)).mkdir(parents=True, exist_ok=True)
        image.save(full_path)
    logger.info(f"Images saved to {output_dir} successfully.")
    return Path(output_dir)


def gen_examples(data_root: Path):
    random.seed(1)
    # Use vismatch available models for examples
    example_algos = [
        "disk-lightglue",
        "xfeat",
        "dedode",
        "loftr",
        "roma",
        "sift-lightglue",
        "d2net",
        "aspanformer",
        "topicfm",
        "superpoint-lightglue",
    ]
    example_algos_rotation_robust = [
        "sift-nn",
        "orb-nn",
        "sift-lightglue",
        # "GIM(dkm)",
    ]
    data_root = Path(data_root)
    if not Path(data_root).exists():
        try:
            download_example_images(DATASETS_REPO_ID, data_root)
        except Exception as e:
            logger.error(f"download_example_images error : {e}")
            data_root = ROOT / "datasets"
    if not Path(data_root / "sacre_coeur/mapping").exists():
        download_example_images(DATASETS_REPO_ID, data_root)

    def distribute_elements(A, B):
        new_B = np.array(B, copy=True).flatten()
        np.random.shuffle(new_B)
        new_B = np.resize(new_B, len(A))
        np.random.shuffle(new_B)
        return new_B.tolist()

    # normal examples
    def gen_images_pairs(count: int = 5):
        path = str(data_root / "sacre_coeur/mapping")
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
        path = data_root / "sacre_coeur/mapping"
        path_rot = data_root / "sacre_coeur/mapping_rot"
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
        path = data_root / "sacre_coeur/mapping"
        path_scale = data_root / "sacre_coeur/mapping_scale"
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
        prefix = "wxbs_benchmark/.WxBS/v1.1"
        wxbs_path = data_root / prefix
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
    # pairs += gen_rot_image_pairs()
    pairs += gen_scale_image_pairs()
    pairs += gen_image_pairs_wxbs()
    pairs_rotation = gen_rot_image_pairs()
    dist_examples = distribute_elements(pairs, example_algos)
    dist_examples_rotation = distribute_elements(
        pairs_rotation, example_algos_rotation_robust
    )
    pairs = pairs_rotation + pairs
    dist_examples = dist_examples_rotation + dist_examples
    match_setting_threshold = DEFAULT_SETTING_THRESHOLD
    match_setting_max_features = DEFAULT_SETTING_MAX_FEATURES
    detect_keypoints_threshold = DEFAULT_DEFAULT_KEYPOINT_THRESHOLD
    ransac_method = DEFAULT_RANSAC_METHOD
    ransac_reproj_threshold = DEFAULT_RANSAC_REPROJ_THRESHOLD
    ransac_confidence = DEFAULT_RANSAC_CONFIDENCE
    ransac_max_iter = DEFAULT_RANSAC_MAX_ITER
    input_lists = []

    for pair, mt in zip(pairs, dist_examples):
        input_lists.append(
            [
                pair[0],
                pair[1],
                match_setting_threshold,
                match_setting_max_features,
                detect_keypoints_threshold,
                mt,
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
        try:
            M, mask = cv2.findHomography(
                kp0,
                kp1,
                method=method,
                ransacReprojThreshold=reproj_threshold,
                confidence=confidence,
                maxIters=max_iter,
            )
        except cv2.error:
            logger.error("compute findHomography error, len(kp0): {}".format(len(kp0)))
            return None, None
    elif geometry_type == "Fundamental":
        try:
            M, mask = cv2.findFundamentalMat(
                kp0,
                kp1,
                method=method,
                ransacReprojThreshold=reproj_threshold,
                confidence=confidence,
                maxIters=max_iter,
            )
        except cv2.error:
            logger.error(
                "compute findFundamentalMat error, len(kp0): {}".format(len(kp0))
            )
            return None, None
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
            mkpts0,
            mkpts1,
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
            H_inv = np.linalg.inv(H)
            rectified_image1 = cv2.warpPerspective(img1, H_inv, (w0, h0))
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


def generate_fake_outputs(
    output_keypoints,
    output_matches_raw,
    output_matches_ransac,
    match_conf,
    extract_conf,
    pred,
):
    return (
        output_keypoints,
        output_matches_raw,
        output_matches_ransac,
        {},
        {
            "match_conf": match_conf,
            "extractor_conf": extract_conf,
        },
        {
            "geom_info": pred.get("geom_info", {}),
        },
        None,
        None,
        None,
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
    use_cached_model: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, int],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, float]],
    np.ndarray,
]:
    """Match two images using vismatch.

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
    # image0 and image1 is RGB mode (H, W, C)
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

    t0 = time.time()
    model = matcher_zoo[key]
    model_name = model["model_name"]

    # Only show efficiency warning if info is available
    info = model.get("info", {})
    if info:
        efficiency = info.get("efficiency", "high")
        if efficiency == "low":
            gr.Warning(
                "Matcher {} is time-consuming, please wait for a while".format(
                    info.get("name", "unknown")
                )
            )

    # Get the vismatch model with parameters
    match_conf = {
        "model_name": model_name,
        "max_num_keypoints": extract_max_keypoints,
        "threshold": match_threshold,
        "keypoint_threshold": keypoint_threshold,
    }
    cache_key = (
        f"{key}_kp{extract_max_keypoints}_th{match_threshold}_kt{keypoint_threshold}"
    )

    if use_cached_model:
        matcher = model_cache.load_model(cache_key, get_model, match_conf)
        logger.info(f"Loaded cached model {cache_key}")
    else:
        matcher = get_model(match_conf)
    logger.info(f"Loading model using: {time.time()-t0:.3f}s")
    t1 = time.time()

    # Convert images from HWC to CHW format for vismatch
    # vismatch expects images in [0, 1] range
    if image0.shape[-1] == 3:
        # Convert from [0, 255] uint8 to [0, 1] float32
        if image0.dtype == np.uint8:
            img0_chw = np.transpose(image0.astype(np.float32) / 255.0, (2, 0, 1))
            img1_chw = np.transpose(image1.astype(np.float32) / 255.0, (2, 0, 1))
        else:
            img0_chw = np.transpose(image0, (2, 0, 1))
            img1_chw = np.transpose(image1, (2, 0, 1))
    else:
        img0_chw = image0
        img1_chw = image1

    yield generate_fake_outputs(
        output_keypoints, output_matches_raw, output_matches_ransac, match_conf, {}, {}
    )

    # Run matching with vismatch
    result = matcher(img0_chw, img1_chw)

    # Convert result to UI format
    pred = convert_vismatch_result(result, image0, image1)

    logger.info(f"Matching images done using: {time.time()-t1:.3f}s")
    t1 = time.time()

    # plot images with keypoints
    titles = [
        "Image 0 - Keypoints",
        "Image 1 - Keypoints",
    ]
    output_keypoints = display_keypoints(pred, titles=titles)
    yield generate_fake_outputs(
        output_keypoints,
        output_matches_raw,
        output_matches_ransac,
        match_conf,
        {},
        pred,
    )

    # plot images with raw matches
    titles = [
        "Image 0 - Raw matched keypoints",
        "Image 1 - Raw matched keypoints",
    ]
    output_matches_raw, num_matches_raw = display_matches(pred, titles=titles)
    yield generate_fake_outputs(
        output_keypoints,
        output_matches_raw,
        output_matches_ransac,
        match_conf,
        {},
        pred,
    )

    # RANSAC is already done by vismatch, but we can also run custom RANSAC if needed
    filter_matches(
        pred,
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
        pred, titles=titles, tag="KPTS_RANSAC"
    )
    yield generate_fake_outputs(
        output_keypoints,
        output_matches_raw,
        output_matches_ransac,
        match_conf,
        {},
        pred,
    )

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
    logger.info(f"TOTAL time: {time.time()-t0:.3f}s")

    state_cache = pred
    state_cache["num_matches_raw"] = num_matches_raw
    state_cache["num_matches_ransac"] = num_matches_ransac
    state_cache["wrapped_image"] = warped_image

    tmp_state_cache = "output.pkl"
    with open(tmp_state_cache, "wb") as f:
        pickle.dump(state_cache, f)
    logger.info("Dump results done!")

    yield (
        output_keypoints,
        output_matches_raw,
        output_matches_ransac,
        {
            "num_raw_matches": num_matches_raw,
            "num_ransac_matches": num_matches_ransac,
        },
        {
            "match_conf": match_conf,
            "extractor_conf": {},
        },
        {
            "geom_info": pred.get("geom_info", {}),
        },
        output_wrapped,
        state_cache,
        tmp_state_cache,
    )


def convert_vismatch_result(
    result: Dict[str, Any], img0: np.ndarray, img1: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Convert vismatch result format to UI expected format.

    vismatch returns:
    - matched_kpts0, matched_kpts1: raw matches
    - inlier_kpts0, inlier_kpts1: RANSAC inliers
    - H: homography matrix

    UI expects:
    - image0_orig, image1_orig: original images
    - keypoints0_orig, keypoints1_orig: all detected keypoints
    - mkeypoints0_orig, mkeypoints1_orig: raw matches
    - mmkeypoints0_orig, mmkeypoints1_orig: RANSAC inliers
    - mconf, mmconf: confidence scores
    """
    # Convert images to uint8 for visualization
    img0_uint8 = (
        (img0 * 255).astype(np.uint8) if img0.max() <= 1.0 else img0.astype(np.uint8)
    )
    img1_uint8 = (
        (img1 * 255).astype(np.uint8) if img1.max() <= 1.0 else img1.astype(np.uint8)
    )

    ret = {
        "image0_orig": img0_uint8,
        "image1_orig": img1_uint8,
    }

    # All detected keypoints (from vismatch, these might be empty for dense matchers)
    all_kpts0 = result.get("all_kpts0", np.array([]))
    all_kpts1 = result.get("all_kpts1", np.array([]))

    if len(all_kpts0) > 0:
        ret["keypoints0_orig"] = all_kpts0
    if len(all_kpts1) > 0:
        ret["keypoints1_orig"] = all_kpts1

    # Raw matches
    matched_kpts0 = result.get("matched_kpts0", np.array([]))
    matched_kpts1 = result.get("matched_kpts1", np.array([]))

    if len(matched_kpts0) > 0 and len(matched_kpts1) > 0:
        ret["mkeypoints0_orig"] = matched_kpts0
        ret["mkeypoints1_orig"] = matched_kpts1
        # Use uniform confidence if not provided
        ret["mconf"] = np.ones(len(matched_kpts0))

    # RANSAC inliers (already computed by vismatch)
    inlier_kpts0 = result.get("inlier_kpts0", np.array([]))
    inlier_kpts1 = result.get("inlier_kpts1", np.array([]))

    if len(inlier_kpts0) > 0 and len(inlier_kpts1) > 0:
        ret["mmkeypoints0_orig"] = inlier_kpts0
        ret["mmkeypoints1_orig"] = inlier_kpts1
        ret["mmconf"] = np.ones(len(inlier_kpts0))

    # Homography matrix
    H = result.get("H")
    if H is not None:
        ret["H"] = H

    ret["num_inliers"] = result.get("num_inliers", 0)

    return ret


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
