"""Configuration utilities for the UI module."""

import gradio as gr
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import kornia as K
from loguru import logger
from vismatch import available_models, get_matcher

from .visualization import figure_to_numpy_array, plot_images  # noqa: F401

# Constants
DEVICE = K.utils.get_cuda_or_mps_device_if_available()
GRADIO_VERSION = gr.__version__.split(".")[0]

# Default values
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

ROOT = Path(__file__).parents[1]


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
    return list(available_models)


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
    """
    Deprecated: Use imcui.ui.config_utils.get_example_data_path() instead.

    Download example dataset from HuggingFace.
    Kept for backward compatibility.
    """
    from .config_utils import _download_example_data

    logger.warning(
        "download_example_images is deprecated. Use imcui.ui.config_utils.get_example_data_path() instead."
    )
    _download_example_data(Path(output_dir))
    return Path(output_dir)
