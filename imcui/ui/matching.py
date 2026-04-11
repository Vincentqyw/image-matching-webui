"""Matching utilities for the UI module."""

import pickle
import time
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from .config import (
    DEFAULT_RANSAC_CONFIDENCE,
    DEFAULT_RANSAC_MAX_ITER,
    DEFAULT_RANSAC_METHOD,
    DEFAULT_RANSAC_REPROJ_THRESHOLD,
    DEFAULT_SETTING_GEOMETRY,
    get_model,
)
from .model_cache import ARCSizeAwareModelCache as ModelCache
from .geometry import filter_matches
from .image_utils import generate_warp_images
from .visualization import display_keypoints, display_matches

# Module-level model cache (lazy initialization)
_model_cache = None


def get_model_cache():
    """Get the global model cache instance, creating it on first use."""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache()
    return _model_cache


# For backward compatibility, provide a property-like access
class ModelCacheWrapper:
    """Wrapper for lazy model cache initialization."""

    def __getattr__(self, name):
        cache = get_model_cache()
        return getattr(cache, name)

    def __call__(self, *args, **kwargs):
        cache = get_model_cache()
        return cache(*args, **kwargs)


model_cache = ModelCacheWrapper()


def send_to_match(
    state_cache: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Send the state cache to the match function.

    Args:
        state_cache (Dict[str, Any]): Current state of the app.

    Returns:
        Tuple of (image0, wrapped_image) or (None, None) if state_cache is empty.
    """
    if state_cache:
        return (
            state_cache["image0_orig"],
            state_cache["wrapped_image"],
        )
    else:
        return None, None


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


def generate_fake_outputs(
    output_keypoints,
    output_matches_raw,
    output_matches_ransac,
    match_conf,
    extract_conf,
    pred,
):
    """Generate placeholder outputs for progressive display during matching."""
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
        use_cached_model (bool, optional): use cached model. Defaults to True.

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
        matcher = get_model_cache().load_model(cache_key, get_model, match_conf)
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
