import os
import cv2
import torch
import random
import numpy as np
import gradio as gr
from pathlib import Path
from itertools import combinations
from typing import Callable, Dict, Any, Optional, Tuple, List, Union
from hloc import matchers, extractors, logger
from hloc.utils.base_model import dynamic_load
from hloc import match_dense, match_features, extract_features
from hloc.utils.viz import add_text, plot_keypoints
from .viz import (
    fig2im,
    plot_images,
    display_matches,
    plot_color_line_matches,
)
import time
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).parent.parent
# some default values
DEFAULT_SETTING_THRESHOLD = 0.1
DEFAULT_SETTING_MAX_FEATURES = 2000
DEFAULT_DEFAULT_KEYPOINT_THRESHOLD = 0.01
DEFAULT_ENABLE_RANSAC = True
DEFAULT_RANSAC_METHOD = "USAC_MAGSAC"
DEFAULT_RANSAC_REPROJ_THRESHOLD = 8
DEFAULT_RANSAC_CONFIDENCE = 0.999
DEFAULT_RANSAC_MAX_ITER = 10000
DEFAULT_MIN_NUM_MATCHES = 4
DEFAULT_MATCHING_THRESHOLD = 0.2
DEFAULT_SETTING_GEOMETRY = "Homography"
GRADIO_VERSION = gr.__version__.split(".")[0]
MATCHER_ZOO = None
models_already_loaded = {}


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
    matcher_zoo: Dict[str, Dict[str, Union[str, bool]]]
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
        dense = v["dense"]
        if dense:
            matcher_zoo_restored[k] = {
                "matcher": match_dense.confs.get(v["matcher"]),
                "dense": dense,
            }
        else:
            matcher_zoo_restored[k] = {
                "feature": extract_features.confs.get(v["feature"]),
                "matcher": match_features.confs.get(v["matcher"]),
                "dense": dense,
            }
    return matcher_zoo_restored


def get_model(match_conf: Dict[str, Any]):
    """
    Load a matcher model from the provided configuration.

    Args:
        match_conf: A dictionary containing the model configuration.

    Returns:
        A matcher model instance.
    """
    Model = dynamic_load(matchers, match_conf["model"]["name"])
    model = Model(match_conf["model"]).eval().to(device)
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
    model = Model(conf["model"]).eval().to(device)
    return model


def gen_examples():
    random.seed(1)
    example_matchers = [
        "disk+lightglue",
        "xfeat(sparse)",
        "dedode",
        "loftr",
        "disk",
        "roma",
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

    def gen_images_pairs(count: int = 5):
        path = str(ROOT / "datasets/sacre_coeur/mapping")
        imgs_list = [
            os.path.join(path, file)
            for file in os.listdir(path)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        pairs = list(combinations(imgs_list, 2))
        selected = random.sample(range(len(pairs)), count)
        return [pairs[i] for i in selected]

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


def filter_matches(
    pred: Dict[str, Any],
    ransac_method: str = DEFAULT_RANSAC_METHOD,
    ransac_reproj_threshold: float = DEFAULT_RANSAC_REPROJ_THRESHOLD,
    ransac_confidence: float = DEFAULT_RANSAC_CONFIDENCE,
    ransac_max_iter: int = DEFAULT_RANSAC_MAX_ITER,
) -> Dict[str, Any]:
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
    if "keypoints0_orig" in pred.keys() and "keypoints1_orig" in pred.keys():
        mkpts0 = pred["keypoints0_orig"]
        mkpts1 = pred["keypoints1_orig"]
        feature_type = "KEYPOINT"
    elif (
        "line_keypoints0_orig" in pred.keys()
        and "line_keypoints1_orig" in pred.keys()
    ):
        mkpts0 = pred["line_keypoints0_orig"]
        mkpts1 = pred["line_keypoints1_orig"]
        feature_type = "LINE"
    else:
        return pred
    if mkpts0 is None or mkpts0 is None:
        return pred
    if ransac_method not in ransac_zoo.keys():
        ransac_method = DEFAULT_RANSAC_METHOD

    if len(mkpts0) < DEFAULT_MIN_NUM_MATCHES:
        return pred
    H, mask = cv2.findHomography(
        mkpts0,
        mkpts1,
        method=ransac_zoo[ransac_method],
        ransacReprojThreshold=ransac_reproj_threshold,
        confidence=ransac_confidence,
        maxIters=ransac_max_iter,
    )
    mask = np.array(mask.ravel().astype("bool"), dtype="bool")
    if H is not None:
        if feature_type == "KEYPOINT":
            pred["keypoints0_orig"] = mkpts0[mask]
            pred["keypoints1_orig"] = mkpts1[mask]
            pred["mconf"] = pred["mconf"][mask]
        elif feature_type == "LINE":
            pred["line_keypoints0_orig"] = mkpts0[mask]
            pred["line_keypoints1_orig"] = mkpts1[mask]
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

    if "keypoints0_orig" in pred.keys() and "keypoints1_orig" in pred.keys():
        mkpts0 = pred["keypoints0_orig"]
        mkpts1 = pred["keypoints1_orig"]
    elif (
        "line_keypoints0_orig" in pred.keys()
        and "line_keypoints1_orig" in pred.keys()
    ):
        mkpts0 = pred["line_keypoints0_orig"]
        mkpts1 = pred["line_keypoints1_orig"]

    if mkpts0 is not None and mkpts1 is not None:
        if len(mkpts0) < 2 * DEFAULT_MIN_NUM_MATCHES:
            return {}
        h1, w1, _ = pred["image0_orig"].shape
        geo_info: Dict[str, List[float]] = {}
        F, inliers = cv2.findFundamentalMat(
            mkpts0,
            mkpts1,
            method=ransac_zoo[ransac_method],
            ransacReprojThreshold=ransac_reproj_threshold,
            confidence=ransac_confidence,
            maxIters=ransac_max_iter,
        )
        if F is not None:
            geo_info["Fundamental"] = F.tolist()
        H, _ = cv2.findHomography(
            mkpts1,
            mkpts0,
            method=ransac_zoo[ransac_method],
            ransacReprojThreshold=ransac_reproj_threshold,
            confidence=ransac_confidence,
            maxIters=ransac_max_iter,
        )
        if H is not None:
            geo_info["Homography"] = H.tolist()
            try:
                _, H1, H2 = cv2.stereoRectifyUncalibrated(
                    mkpts0.reshape(-1, 2),
                    mkpts1.reshape(-1, 2),
                    F,
                    imgSize=(w1, h1),
                )
                geo_info["H1"] = H1.tolist()
                geo_info["H2"] = H2.tolist()
            except cv2.error as e:
                logger.error(f"{e}, skip")
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
    h1, w1, _ = img0.shape
    h2, w2, _ = img1.shape
    result_matrix: Optional[np.ndarray] = None
    if geo_info is not None and len(geo_info) != 0:
        rectified_image0 = img0
        rectified_image1 = None
        H = np.array(geo_info["Homography"])
        F = np.array(geo_info["Fundamental"])
        title: List[str] = []
        if geom_type == "Homography":
            rectified_image1 = cv2.warpPerspective(
                img1, H, (img0.shape[1], img0.shape[0])
            )
            result_matrix = H
            title = ["Image 0", "Image 1 - warped"]
        elif geom_type == "Fundamental":
            H1, H2 = np.array(geo_info["H1"]), np.array(geo_info["H2"])
            rectified_image0 = cv2.warpPerspective(img0, H1, (w1, h1))
            rectified_image1 = cv2.warpPerspective(img1, H2, (w2, h2))
            result_matrix = F
            title = ["Image 0 - warped", "Image 1 - warped"]
        else:
            print("Error: Unknown geometry type")
        fig = plot_images(
            [rectified_image0.squeeze(), rectified_image1.squeeze()],
            title,
            dpi=300,
        )
        dictionary = {
            "row1": result_matrix[0].tolist(),
            "row2": result_matrix[1].tolist(),
            "row3": result_matrix[2].tolist(),
        }
        return fig2im(fig), dictionary
    else:
        return None, None


def generate_warp_images(
    input_image0: np.ndarray,
    input_image1: np.ndarray,
    matches_info: Dict[str, Any],
    choice: str,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Changes the estimate of the geometric transformation used to align the images.

    Args:
        input_image0: First input image.
        input_image1: Second input image.
        matches_info: Dictionary containing information about the matches.
        choice: Type of geometric transformation to use ('Homography' or 'Fundamental') or 'No' to disable.

    Returns:
        A tuple containing the updated images and the updated matches info.
    """
    if (
        matches_info is None
        or len(matches_info) < 1
        or "geom_info" not in matches_info.keys()
    ):
        return None, None
    geom_info: Dict[str, Any] = matches_info["geom_info"]
    wrapped_images: Optional[np.ndarray] = None
    if choice != "No":
        wrapped_images, _ = wrap_images(
            input_image0, input_image1, geom_info, choice
        )
        return wrapped_images, matches_info
    else:
        return None, None


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
        raise gr.Error("Error: No images found! Please upload two images.")
    # init output
    output_keypoints = None
    output_matches_raw = None
    output_matches_ransac = None

    # super slow!
    if "roma" in key.lower():
        gr.Info(
            f"Success! Please be patient and allow for about 2-3 minutes."
            f" Due to CPU inference, {key} is quiet slow."
        )
    model = matcher_zoo[key]
    match_conf = model["matcher"]
    # update match config
    match_conf["model"]["match_threshold"] = match_threshold
    match_conf["model"]["max_keypoints"] = extract_max_keypoints
    t0 = time.time()
    cache_key = "{}_{}".format(key, match_conf["model"]["name"])
    if cache_key in models_already_loaded:
        matcher = models_already_loaded[cache_key]
        matcher.conf["max_keypoints"] = extract_max_keypoints
        matcher.conf["match_threshold"] = match_threshold
        logger.info(f"Loaded cached model {cache_key}")
    else:
        matcher = get_model(match_conf)
        models_already_loaded[cache_key] = matcher
    gr.Info(f"Loading model using: {time.time()-t0:.3f}s")
    logger.info(f"Loading model using: {time.time()-t0:.3f}s")
    t1 = time.time()

    if model["dense"]:
        pred = match_dense.match_images(
            matcher, image0, image1, match_conf["preprocessing"], device=device
        )
        del matcher
        extract_conf = None
    else:
        extract_conf = model["feature"]
        # update extract config
        extract_conf["model"]["max_keypoints"] = extract_max_keypoints
        extract_conf["model"]["keypoint_threshold"] = keypoint_threshold
        cache_key = "{}_{}".format(key, extract_conf["model"]["name"])
        if cache_key in models_already_loaded:
            extractor = models_already_loaded[cache_key]
            extractor.conf["max_keypoints"] = extract_max_keypoints
            extractor.conf["keypoint_threshold"] = keypoint_threshold
            logger.info(f"Loaded cached model {cache_key}")
        else:
            extractor = get_feature_model(extract_conf)
            models_already_loaded[cache_key] = extractor
        pred0 = extract_features.extract(
            extractor, image0, extract_conf["preprocessing"]
        )
        pred1 = extract_features.extract(
            extractor, image1, extract_conf["preprocessing"]
        )
        pred = match_features.match_images(matcher, pred0, pred1)
        del extractor
    gr.Info(f"Matching images done using: {time.time()-t1:.3f}s")
    logger.info(f"Matching images done using: {time.time()-t1:.3f}s")
    t1 = time.time()
    # plot images with keypoints\
    titles = [
        "Image 0 - Keypoints",
        "Image 1 - Keypoints",
    ]
    output_keypoints = plot_images([image0, image1], titles=titles, dpi=300)
    if "keypoints0" in pred.keys() and "keypoints1" in pred.keys():
        plot_keypoints([pred["keypoints0"], pred["keypoints1"]])
        text = (
            f"# keypoints0: {len(pred['keypoints0'])} \n"
            + f"# keypoints1: {len(pred['keypoints1'])}"
        )
        add_text(0, text, fs=15)
    output_keypoints = fig2im(output_keypoints)

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
    gr.Info(f"RANSAC matches done using: {time.time()-t1:.3f}s")
    logger.info(f"RANSAC matches done using: {time.time()-t1:.3f}s")
    t1 = time.time()

    # plot images with ransac matches
    titles = [
        "Image 0 - Ransac matched keypoints",
        "Image 1 - Ransac matched keypoints",
    ]
    output_matches_ransac, num_matches_ransac = display_matches(
        pred, titles=titles
    )
    gr.Info(f"Display matches done using: {time.time()-t1:.3f}s")
    logger.info(f"Display matches done using: {time.time()-t1:.3f}s")

    t1 = time.time()
    # plot wrapped images
    geom_info = compute_geometry(pred)
    output_wrapped, _ = generate_warp_images(
        pred["image0_orig"],
        pred["image1_orig"],
        {"geom_info": geom_info},
        choice_geometry_type,
    )
    plt.close("all")
    del pred
    logger.info(f"TOTAL time: {time.time()-t0:.3f}s")
    gr.Info(f"In summary, total time: {time.time()-t0:.3f}s")

    return (
        output_keypoints,
        output_matches_raw,
        output_matches_ransac,
        {
            "number raw matches": num_matches_raw,
            "number ransac matches": num_matches_ransac,
        },
        {
            "match_conf": match_conf,
            "extractor_conf": extract_conf,
        },
        {
            "geom_info": geom_info,
        },
        output_wrapped,
    )


# @ref: https://docs.opencv.org/4.x/d0/d74/md__build_4_x-contrib_docs-lin64_opencv_doc_tutorials_calib3d_usac.html
# AND: https://opencv.org/blog/2021/06/09/evaluating-opencvs-new-ransacs
ransac_zoo = {
    "RANSAC": cv2.RANSAC,
    "USAC_MAGSAC": cv2.USAC_MAGSAC,
    "USAC_DEFAULT": cv2.USAC_DEFAULT,
    "USAC_FM_8PTS": cv2.USAC_FM_8PTS,
    "USAC_PROSAC": cv2.USAC_PROSAC,
    "USAC_FAST": cv2.USAC_FAST,
    "USAC_ACCURATE": cv2.USAC_ACCURATE,
    "USAC_PARALLEL": cv2.USAC_PARALLEL,
}
