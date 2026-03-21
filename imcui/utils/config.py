"""Configuration and path utility functions.

This module provides centralized utilities for configuration management
and path resolution, eliminating code duplication across entry points.
"""

import os
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
    Get the example data directory path with auto-download support.

    Search order:
        1. IMCUI_DATA_DIR environment variable
        2. Package datasets (development mode with git clone)
        3. User cache directory with auto-download from HuggingFace

    Returns:
        Path: Path to the example datasets directory.
    """
    # 1. Check environment variable
    if env_path := os.environ.get("IMCUI_DATA_DIR"):
        path = Path(env_path)
        if path.exists():
            logger.info(f"Using IMCUI_DATA_DIR: {path}")
            return path

    # 2. Check package datasets (development mode)
    package_datasets = Path(__file__).parent.parent / "datasets"
    if (package_datasets / "sacre_coeur" / "mapping").exists():
        logger.info(f"Using package datasets (dev mode): {package_datasets}")
        return package_datasets

    # 3. Download to user cache directory
    cache_dir = _get_cache_dir()
    if not (cache_dir / "sacre_coeur" / "mapping").exists():
        logger.info(
            f"Example datasets not found. Will download from HuggingFace to: {cache_dir}"
        )
        _download_example_data(cache_dir)

    logger.info(f"Using example data from: {cache_dir}")
    return cache_dir


def _get_cache_dir() -> Path:
    """
    Get the cache directory for example datasets.

    Returns:
        Path: Platform-specific cache directory.
    """
    # Use platform-specific cache directory
    if os.name == "nt":  # Windows
        cache_root = Path(
            os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        )
    elif os.name == "posix":
        # macOS and Linux
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    else:
        cache_root = Path.home() / ".cache"

    cache_dir = cache_root / "imcui" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _download_example_data(output_dir: Path) -> None:
    """
    Download example data from HuggingFace Hub.

    Args:
        output_dir: Directory to save the downloaded data.
    """
    try:
        from datasets import load_dataset

        repo_id = "Realcat/imcui_datasets"
        logger.info(f"Downloading example datasets from HuggingFace: {repo_id}")
        logger.info(f"Target directory: {output_dir}")

        dataset = load_dataset(repo_id)
        for example in dataset["train"]:
            file_path = example["path"]
            image = example["image"]  # PIL.Image object
            full_path = output_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(full_path)

        logger.info(
            f"Successfully downloaded {len(dataset['train'])} example images to {output_dir}"
        )
    except Exception as e:
        logger.error(f"Failed to download example data: {e}")
        logger.warning(
            "Please either:\n"
            "  1. Install datasets: pip install datasets\n"
            "  2. Set IMCUI_DATA_DIR environment variable\n"
            "  3. Clone the repo to get example data: git clone https://github.com/Vincentqyw/image-matching-webui.git"
        )
        raise


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
