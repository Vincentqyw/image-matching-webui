"""Geometry and RANSAC utilities for the UI module."""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import poselib
from loguru import logger

# Import defaults from config
from .config import (
    DEFAULT_MIN_NUM_MATCHES,
    DEFAULT_RANSAC_CONFIDENCE,
    DEFAULT_RANSAC_MAX_ITER,
    DEFAULT_RANSAC_METHOD,
    DEFAULT_RANSAC_REPROJ_THRESHOLD,
)


def filter_matches_opencv(
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


def filter_matches_poselib(
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


def process_ransac_matches(
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
        return filter_matches_opencv(
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
        return filter_matches_poselib(
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


def set_null_pred(feature_type: str, pred: dict):
    """Set prediction to null values for the given feature type."""
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

        F, mask_f = process_ransac_matches(
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
        H, mask_h = process_ransac_matches(
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
