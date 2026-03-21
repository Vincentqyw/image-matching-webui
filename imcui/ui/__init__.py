"""Gradio UI module for Image Matching WebUI."""

# Import version from top-level package for backward compatibility
from .. import __version__  # noqa: F401

# Import main application class
from .image_matching_app import ImageMatchingApp  # noqa: F401

# Import common utilities for easy access
from .config import (  # noqa: F401
    DEVICE,
    GRADIO_VERSION,
    get_matcher_zoo,
    get_available_model_names,
    load_config,
    DEFAULT_RANSAC_METHOD,
    DEFAULT_RANSAC_REPROJ_THRESHOLD,
    DEFAULT_RANSAC_CONFIDENCE,
    DEFAULT_RANSAC_MAX_ITER,
    DEFAULT_SETTING_GEOMETRY,
    DEFAULT_SETTING_THRESHOLD,
    DEFAULT_SETTING_MAX_FEATURES,
    DEFAULT_DEFAULT_KEYPOINT_THRESHOLD,
    DEFAULT_MATCHING_THRESHOLD,
    DEFAULT_MIN_NUM_MATCHES,
)

# Import geometry utilities
from .geometry import (  # noqa: F401
    ransac_zoo,
    filter_matches,
    compute_geometry,
    process_ransac_matches,
    set_null_pred,
)

# Import matching utilities
from .matching import (  # noqa: F401
    run_matching,
    run_ransac,
    send_to_match,
    convert_vismatch_result,
    model_cache,
)

# Import image utilities
from .image_utils import (  # noqa: F401
    wrap_images,
    generate_warp_images,
    rotate_image,
    scale_image,
)

# Import config utilities (path and version functions)
from .config_utils import (  # noqa: F401
    get_default_config_path,
    get_example_data_path,
    get_version,
)

# Import examples
from .examples import gen_examples  # noqa: F401

# Import visualization (for advanced use)
from .visualization import (  # noqa: F401
    display_keypoints,
    display_matches,
    plot_images,
    fig2im,
)

# Import model cache
from .model_cache import ARCSizeAwareModelCache, LRUModelCache  # noqa: F401

__all__ = [
    "__version__",
    "ImageMatchingApp",
    # Config utility functions
    "get_default_config_path",
    "get_example_data_path",
    "get_version",
    # Constants
    "DEVICE",
    "GRADIO_VERSION",
    "DEFAULT_RANSAC_METHOD",
    "DEFAULT_RANSAC_REPROJ_THRESHOLD",
    "DEFAULT_RANSAC_CONFIDENCE",
    "DEFAULT_RANSAC_MAX_ITER",
    "DEFAULT_SETTING_GEOMETRY",
    "DEFAULT_SETTING_THRESHOLD",
    "DEFAULT_SETTING_MAX_FEATURES",
    "DEFAULT_DEFAULT_KEYPOINT_THRESHOLD",
    "DEFAULT_MATCHING_THRESHOLD",
    "DEFAULT_MIN_NUM_MATCHES",
    # Config functions
    "get_matcher_zoo",
    "get_available_model_names",
    "load_config",
    # Geometry functions
    "ransac_zoo",
    "filter_matches",
    "compute_geometry",
    "process_ransac_matches",
    "set_null_pred",
    # Matching functions
    "run_matching",
    "run_ransac",
    "send_to_match",
    "convert_vismatch_result",
    "model_cache",
    # Image utilities
    "wrap_images",
    "generate_warp_images",
    "rotate_image",
    "scale_image",
    # Examples
    "gen_examples",
    # Visualization
    "display_keypoints",
    "display_matches",
    "plot_images",
    "fig2im",
    # Model cache
    "ARCSizeAwareModelCache",
    "LRUModelCache",
]
