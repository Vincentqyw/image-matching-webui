"""Gradio UI module for Image Matching WebUI."""

# Import version from top-level package for backward compatibility
from .. import __version__  # noqa: F401

# Import main application class (will be updated in Phase 3)
from .app_class import ImageMatchingApp  # noqa: F401

# Import common utilities for easy access
from .utils import (  # noqa: F401
    DEVICE,
    GRADIO_VERSION,
    get_matcher_zoo,
    get_available_model_names,
)

__all__ = [
    "__version__",
    "ImageMatchingApp",
    "DEVICE",
    "GRADIO_VERSION",
    "get_matcher_zoo",
    "get_available_model_names",
]
