"""Configuration and path utility functions.

This module provides centralized utilities for configuration management
and path resolution, eliminating code duplication across entry points.
"""

from pathlib import Path
from loguru import logger


def get_default_config_path() -> Path:
    """
    Get the default configuration file path.

    Search order:
        1. cwd/app.yaml
        2. cwd/config/app.yaml
        3. Package default: imcui/config/app.yaml

    Returns:
        Path: Path to the configuration file to use.
    """
    # Check current directory
    for local_path in [Path.cwd() / "app.yaml", Path.cwd() / "config" / "app.yaml"]:
        if local_path.exists():
            logger.info(f"Using config file from current directory: {local_path}")
            return local_path

    # Fall back to package default
    default = Path(__file__).parent.parent / "config" / "app.yaml"
    logger.info(f"Using package default config: {default}")
    return default


def get_example_data_path() -> Path:
    """
    Get the example data directory path.

    Returns:
        Path: Path to the example datasets directory.
    """
    path = Path(__file__).parent.parent / "datasets"
    logger.info(f"Using example data root: {path}")
    return path


def get_version() -> str:
    """
    Get the package version (single source of truth).

    Attempts to read from installed package metadata first,
    falls back to reading pyproject.toml for development mode.

    Returns:
        str: Version string (e.g., "1.0.0" or "dev").
    """
    try:
        from importlib.metadata import version, PackageNotFoundError

        return version("imcui")
    except PackageNotFoundError:
        # Development mode - read from pyproject.toml
        try:
            import tomllib

            pyproject = Path(__file__).parent.parent.parent / "pyproject.toml"
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
                return data["project"]["version"]
        except Exception:
            return "dev"
