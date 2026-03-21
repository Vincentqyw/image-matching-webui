"""Image Matching WebUI package."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("imcui")
except PackageNotFoundError:
    # Development mode - read from pyproject.toml
    try:
        from pathlib import Path
        import tomllib

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject, "rb") as f:
            __version__ = tomllib.load(f)["project"]["version"]
    except Exception:
        __version__ = "dev"

from .utils.config import get_default_config_path, get_example_data_path, get_version
from .ui.app_class import ImageMatchingApp

__all__ = [
    "__version__",
    "ImageMatchingApp",
    "get_default_config_path",
    "get_example_data_path",
    "get_version",
]
