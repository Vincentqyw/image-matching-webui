# api.py - Using vismatch
import warnings
from loguru import logger
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from vismatch import get_matcher

from ..ui.utils import filter_matches
from ..ui.viz import display_matches, fig2im, plot_images, add_text, plot_keypoints

warnings.simplefilter("ignore")


class ImageMatchingAPI(torch.nn.Module):
    default_conf = {
        "ransac": {
            "enable": True,
            "estimator": "poselib",
            "geometry": "homography",
            "method": "RANSAC",
            "reproj_threshold": 3,
            "confidence": 0.9999,
            "max_iter": 10000,
        },
    }

    def __init__(
        self,
        conf: dict = {},
        device: str = "cpu",
        detect_threshold: float = 0.015,
        max_keypoints: int = 1024,
        match_threshold: float = 0.2,
    ) -> None:
        """
        Initializes an instance of the ImageMatchingAPI class using vismatch.

        Args:
            conf (dict): A dictionary containing the configuration parameters.
            device (str, optional): The device to use for computation. Defaults to "cpu".
            detect_threshold (float, optional): The threshold for detecting keypoints. Defaults to 0.015.
            max_keypoints (int, optional): The maximum number of keypoints to extract. Defaults to 1024.
            match_threshold (float, optional): The threshold for matching keypoints. Defaults to 0.2.

        Returns:
            None
        """
        super().__init__()
        self.device = device
        self.conf = {**self.default_conf, **conf}
        self._update_config(detect_threshold, max_keypoints, match_threshold)
        self._init_models()
        if device == "cuda":
            memory_allocated = torch.cuda.memory_allocated(device)
            memory_reserved = torch.cuda.memory_reserved(device)
            logger.info(f"GPU memory allocated: {memory_allocated / 1024**2:.3f} MB")
            logger.info(f"GPU memory reserved: {memory_reserved / 1024**2:.3f} MB")
        self.pred = None

    def _update_config(
        self,
        detect_threshold: float = 0.015,
        max_keypoints: int = 1024,
        match_threshold: float = 0.2,
    ):
        """Update configuration parameters."""
        self.max_keypoints = max_keypoints
        self.detect_threshold = detect_threshold
        self.match_threshold = match_threshold

    def _init_models(self):
        """Initialize the vismatch model with parameters."""
        model_name = self.conf.get("model_name", "superpoint-lightglue")
        # Build kwargs for vismatch
        kwargs = {
            "max_num_keypoints": getattr(self, "max_keypoints", 2048),
        }
        # Some matchers support threshold parameter (especially dense matchers)
        threshold = getattr(self, "match_threshold", None)
        if threshold is not None:
            kwargs["threshold"] = threshold
        self.matcher = get_matcher(model_name, device=self.device, **kwargs)

    def set_model(
        self, model_name: str, max_keypoints: int = None, threshold: float = None
    ):
        """Switch to a different vismatch model."""
        kwargs = {
            "max_num_keypoints": max_keypoints or getattr(self, "max_keypoints", 2048),
        }
        if threshold is not None:
            kwargs["threshold"] = threshold
        self.matcher = get_matcher(model_name, device=self.device, **kwargs)

    def _convert_result_format(
        self, result: Dict[str, Any], img0: np.ndarray, img1: np.ndarray
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
            (img0 * 255).astype(np.uint8)
            if img0.max() <= 1.0
            else img0.astype(np.uint8)
        )
        img1_uint8 = (
            (img1 * 255).astype(np.uint8)
            if img1.max() <= 1.0
            else img1.astype(np.uint8)
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

    @torch.inference_mode()
    def forward(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Forward pass of the image matching API using vismatch.

        Args:
            img0: A 3D NumPy array of shape (H, W, C) representing the first image.
                  Values are in the range [0, 1] and are in RGB mode.
            img1: A 3D NumPy array of shape (H, W, C) representing the second image.
                  Values are in the range [0, 1] and are in RGB mode.

        Returns:
            A dictionary containing the following keys:
            - image0_orig: The original image 0.
            - image1_orig: The original image 1.
            - keypoints0_orig: The keypoints detected in image 0.
            - keypoints1_orig: The keypoints detected in image 1.
            - mkeypoints0_orig: The raw matches between image 0 and image 1.
            - mkeypoints1_orig: The raw matches between image 1 and image 0.
            - mmkeypoints0_orig: The RANSAC inliers in image 0.
            - mmkeypoints1_orig: The RANSAC inliers in image 1.
            - mconf: The confidence scores for the raw matches.
            - mmconf: The confidence scores for the RANSAC inliers.
        """
        # Take as input a pair of images (not a batch)
        assert isinstance(img0, np.ndarray)
        assert isinstance(img1, np.ndarray)

        # Convert HWC to CHW format for vismatch
        if img0.shape[-1] == 3:
            img0_chw = np.transpose(img0, (2, 0, 1))
            img1_chw = np.transpose(img1, (2, 0, 1))
        else:
            img0_chw = img0
            img1_chw = img1

        # Run vismatch
        result = self.matcher(img0_chw, img1_chw)

        # Convert to UI format
        self.pred = self._convert_result_format(result, img0, img1)

        return self.pred

    def _geometry_check(
        self,
        pred: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Filter matches using RANSAC. Note: vismatch already does RANSAC internally,
        but we keep this for consistency with the UI.

        Args:
            pred (Dict[str, Any]): dict of matches, including original keypoints.

        Returns:
            Dict[str, Any]: filtered matches
        """
        # vismatch already provides inliers, but we can re-run RANSAC if needed
        pred = filter_matches(
            pred,
            ransac_method=self.conf["ransac"]["method"],
            ransac_reproj_threshold=self.conf["ransac"]["reproj_threshold"],
            ransac_confidence=self.conf["ransac"]["confidence"],
            ransac_max_iter=self.conf["ransac"]["max_iter"],
        )
        return pred

    def visualize(
        self,
        log_path: Optional[Path] = None,
    ) -> None:
        """
        Visualize the matches.

        Args:
            log_path (Path, optional): The directory to save the images. Defaults to None.

        Returns:
            None
        """
        postfix = self.conf.get("model_name", "vismatch")
        titles = [
            "Image 0 - Keypoints",
            "Image 1 - Keypoints",
        ]
        pred: Dict[str, Any] = self.pred
        image0: np.ndarray = pred["image0_orig"]
        image1: np.ndarray = pred["image1_orig"]
        output_keypoints: np.ndarray = plot_images(
            [image0, image1], titles=titles, dpi=300
        )
        if "keypoints0_orig" in pred.keys() and "keypoints1_orig" in pred.keys():
            plot_keypoints([pred["keypoints0_orig"], pred["keypoints1_orig"]])
            text: str = (
                f"# keypoints0: {len(pred['keypoints0_orig'])} \n"
                + f"# keypoints1: {len(pred['keypoints1_orig'])}"
            )
            add_text(0, text, fs=15)
        output_keypoints = fig2im(output_keypoints)
        # plot images with raw matches
        titles = [
            "Image 0 - Raw matched keypoints",
            "Image 1 - Raw matched keypoints",
        ]
        output_matches_raw, num_matches_raw = display_matches(
            pred, titles=titles, tag="KPTS_RAW"
        )
        # plot images with ransac matches
        titles = [
            "Image 0 - Ransac matched keypoints",
            "Image 1 - Ransac matched keypoints",
        ]
        output_matches_ransac, num_matches_ransac = display_matches(
            pred, titles=titles, tag="KPTS_RANSAC"
        )
        if log_path is not None:
            img_keypoints_path: Path = log_path / f"img_keypoints_{postfix}.png"
            img_matches_raw_path: Path = log_path / f"img_matches_raw_{postfix}.png"
            img_matches_ransac_path: Path = (
                log_path / f"img_matches_ransac_{postfix}.png"
            )
            cv2.imwrite(
                str(img_keypoints_path),
                output_keypoints[:, :, ::-1].copy(),  # RGB -> BGR
            )
            cv2.imwrite(
                str(img_matches_raw_path),
                output_matches_raw[:, :, ::-1].copy(),  # RGB -> BGR
            )
            cv2.imwrite(
                str(img_matches_ransac_path),
                output_matches_ransac[:, :, ::-1].copy(),  # RGB -> BGR
            )
            plt.close("all")
