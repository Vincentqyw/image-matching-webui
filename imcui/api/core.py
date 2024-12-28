# api.py
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from imcui.hloc import extract_features, logger, match_dense, match_features
from imcui.hloc.utils.viz import add_text, plot_keypoints
from imcui.ui.utils import filter_matches, get_feature_model, get_model
from imcui.ui.viz import display_matches, fig2im, plot_images

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
        Initializes an instance of the ImageMatchingAPI class.

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
        self._updata_config(detect_threshold, max_keypoints, match_threshold)
        self._init_models()
        if device == "cuda":
            memory_allocated = torch.cuda.memory_allocated(device)
            memory_reserved = torch.cuda.memory_reserved(device)
            logger.info(f"GPU memory allocated: {memory_allocated / 1024**2:.3f} MB")
            logger.info(f"GPU memory reserved: {memory_reserved / 1024**2:.3f} MB")
        self.pred = None

    def parse_match_config(self, conf):
        if conf["dense"]:
            return {
                **conf,
                "matcher": match_dense.confs.get(conf["matcher"]["model"]["name"]),
                "dense": True,
            }
        else:
            return {
                **conf,
                "feature": extract_features.confs.get(conf["feature"]["model"]["name"]),
                "matcher": match_features.confs.get(conf["matcher"]["model"]["name"]),
                "dense": False,
            }

    def _updata_config(
        self,
        detect_threshold: float = 0.015,
        max_keypoints: int = 1024,
        match_threshold: float = 0.2,
    ):
        self.dense = self.conf["dense"]
        if self.conf["dense"]:
            try:
                self.conf["matcher"]["model"]["match_threshold"] = match_threshold
            except TypeError as e:
                logger.error(e)
        else:
            self.conf["feature"]["model"]["max_keypoints"] = max_keypoints
            self.conf["feature"]["model"]["keypoint_threshold"] = detect_threshold
            self.extract_conf = self.conf["feature"]

        self.match_conf = self.conf["matcher"]

    def _init_models(self):
        # initialize matcher
        self.matcher = get_model(self.match_conf)
        # initialize extractor
        if self.dense:
            self.extractor = None
        else:
            self.extractor = get_feature_model(self.conf["feature"])

    def _forward(self, img0, img1):
        if self.dense:
            pred = match_dense.match_images(
                self.matcher,
                img0,
                img1,
                self.match_conf["preprocessing"],
                device=self.device,
            )
            last_fixed = "{}".format(  # noqa: F841
                self.match_conf["model"]["name"]
            )
        else:
            pred0 = extract_features.extract(
                self.extractor, img0, self.extract_conf["preprocessing"]
            )
            pred1 = extract_features.extract(
                self.extractor, img1, self.extract_conf["preprocessing"]
            )
            pred = match_features.match_images(self.matcher, pred0, pred1)
        return pred

    def _convert_pred(self, pred):
        ret = {
            k: v.cpu().detach()[0].numpy() if isinstance(v, torch.Tensor) else v
            for k, v in pred.items()
        }
        ret = {
            k: v[0].cpu().detach().numpy() if isinstance(v, list) else v
            for k, v in ret.items()
        }
        return ret

    @torch.inference_mode()
    def extract(self, img0: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Extract features from a single image.

        Args:
            img0 (np.ndarray): image

        Returns:
            Dict[str, np.ndarray]: feature dict
        """

        # setting prams
        self.extractor.conf["max_keypoints"] = kwargs.get("max_keypoints", 512)
        self.extractor.conf["keypoint_threshold"] = kwargs.get(
            "keypoint_threshold", 0.0
        )

        pred = extract_features.extract(
            self.extractor, img0, self.extract_conf["preprocessing"]
        )
        pred = self._convert_pred(pred)
        # back to origin scale
        s0 = pred["original_size"] / pred["size"]
        pred["keypoints_orig"] = (
            match_features.scale_keypoints(pred["keypoints"] + 0.5, s0) - 0.5
        )
        # TODO: rotate back
        binarize = kwargs.get("binarize", False)
        if binarize:
            assert "descriptors" in pred
            pred["descriptors"] = (pred["descriptors"] > 0).astype(np.uint8)
            pred["descriptors"] = pred["descriptors"].T  # N x DIM
        return pred

    @torch.inference_mode()
    def forward(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Forward pass of the image matching API.

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
        self.pred = self._forward(img0, img1)
        if self.conf["ransac"]["enable"]:
            self.pred = self._geometry_check(self.pred)
        return self.pred

    def _geometry_check(
        self,
        pred: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Filter matches using RANSAC. If keypoints are available, filter by keypoints.
        If lines are available, filter by lines. If both keypoints and lines are
        available, filter by keypoints.

        Args:
            pred (Dict[str, Any]): dict of matches, including original keypoints.
                                  See :func:`filter_matches` for the expected keys.

        Returns:
            Dict[str, Any]: filtered matches
        """
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
        if self.conf["dense"]:
            postfix = str(self.conf["matcher"]["model"]["name"])
        else:
            postfix = "{}_{}".format(
                str(self.conf["feature"]["model"]["name"]),
                str(self.conf["matcher"]["model"]["name"]),
            )
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
