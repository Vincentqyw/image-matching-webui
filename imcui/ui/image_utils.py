"""Image manipulation utilities for the UI."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from loguru import logger

# Import visualization functions
from .visualization import fig2im, plot_images


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
            logger.error("Unknown geometry type")
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


def rotate_image(
    input_path: Union[str, Path], degrees: float, output_path: Union[str, Path]
):
    """Rotate an image by the specified degrees.

    Args:
        input_path: Path to the input image.
        degrees: Degrees to rotate (counter-clockwise).
        output_path: Path to save the rotated image.
    """
    img = Image.open(input_path)
    img_rotated = img.rotate(-degrees)
    img_rotated.save(output_path)


def scale_image(
    input_path: Union[str, Path],
    scale_factor: float,
    output_path: Union[str, Path],
):
    """Scale an image by the specified factor (with padding to maintain aspect ratio).

    Args:
        input_path: Path to the input image.
        scale_factor: Scale factor (e.g., 0.5 for half size).
        output_path: Path to save the scaled image.
    """
    img = Image.open(input_path)
    width, height = img.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    new_img = Image.new("RGB", (width, height), (0, 0, 0))
    img_resized = img.resize((new_width, new_height))
    position = ((width - new_width) // 2, (height - new_height) // 2)
    new_img.paste(img_resized, position)
    new_img.save(output_path)
