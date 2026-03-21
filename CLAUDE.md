# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Image Matching WebUI (IMCUI) is a Gradio-based web interface for matching image pairs using various state-of-the-art computer vision algorithms. It supports both sparse (keypoint-based) and dense (learned) matching methods.

## Common Commands

### Installation
```bash
# From PyPI (recommended)
pip install imcui

# Or from source
git clone https://github.com/Vincentqyw/image-matching-webui.git
cd image-matching-webui
pip install -e .

# ⚠️ Deprecated: conda environment.yaml
# Use pip instead for all installations
```

### Running the Application
```bash
# Using CLI (recommended)
imcui

# Or directly
python app.py

# With custom config
imcui --config /path/to/config.yaml

# On custom port
imcui -p 8080

# Custom example data directory
imcui -d /path/to/datasets

# Or use environment variable
export IMCUI_DATA_DIR=/path/to/datasets
imcui

# Docker
docker-compose up webui
```

**Note for PyPI Users**: Example datasets (82MB) are NOT included in the PyPI package. On first run, they will be automatically downloaded from HuggingFace to your user cache directory (`~/.cache/imcui/datasets/` on Linux/macOS, `%LOCALAPPDATA%\imcui\datasets\` on Windows). To use local datasets, either:
- Clone the repository: `git clone https://github.com/Vincentqyw/image-matching-webui.git`
- Set `IMCUI_DATA_DIR` environment variable
- Use `-d` flag to specify custom path

### CLI Options
| Flag | Default | Description |
|------|---------|-------------|
| `-s, --server-name` | `0.0.0.0` | Host to bind |
| `-p, --server-port` | `7860` | Port to run on |
| `-c, --config` | Auto-detected | Custom config YAML path |
| `-d, --example-data-root` | `imcui/datasets` | Example datasets directory |
| `-v, --verbose` | `False` | Enable verbose output |

### Development
```bash
# Run tests
pytest tests/ -v

# Run a single test
pytest tests/test_basic.py::test_one -v

# Run pre-commit checks (uses ruff, ruff-format, mypy, clang-format)
pre-commit run -a
```

### API Server
```bash
# Start API server
python -m imcui.api.server

# Docker with GPU
docker-compose up api
```

## Architecture

### Directory Structure
- `imcui/` - Main package (Python)
  - `cli/` - CLI entry point (main.py)
  - `ui/` - Gradio-based web interface (app_class.py, utils.py)
  - `api/` - Core matching API (ImageMatchingAPI class, server.py, client.py)
  - `utils/` - Shared utilities (config.py - version, paths, configuration)
  - `config/` - Configuration files (app.yaml, api.yaml)
  - `datasets/` - Example datasets for testing
- `cpp/` - C++ code (independent build system)
  - `test/` - C++ API test client
  - `README.md` - Build instructions
- `tests/` - Python test suite
- `docker/` - Docker deployment configuration
- `app.py` - HuggingFace Spaces entry point (backward compatibility)
- `experiments/` - Debug output (can be safely deleted)
- `assets/` - Image assets for documentation

### Key Concepts

**Sparse vs Dense Matching**: Sparse methods extract discrete keypoints and match them. Dense methods work on full image correlations.

**Matcher Zoo**: All matchers are dynamically loaded from [vismatch](https://github.com/gmberton/vismatch) package (by [@gmberton](https://github.com/gmberton)). This WebUI no longer maintains matching algorithms.

**Entry Points**:
- `imcui` CLI (recommended) - uses shared utilities from `imcui.utils.config`
- `python app.py` - HuggingFace Spaces entry point, also uses shared utilities
- Both entry points share identical configuration loading logic

**Version Management**: Single source of truth in `pyproject.toml` (currently 1.0.0). The `imcui.__version__` reads from installed package metadata or falls back to pyproject.toml for development.

**API Usage**:
```python
from imcui.api import ImageMatchingAPI
from imcui.ui.utils import DEVICE, get_matcher_zoo

matcher_zoo = get_matcher_zoo()  # Dynamically loaded from vismatch
api = ImageMatchingAPI(conf=matcher_zoo["superpoint-lightglue"], device=DEVICE)
pred = api(image0, image1)  # RGB numpy arrays
```

**Utility Functions** (available from top-level imports):
```python
from imcui import get_default_config_path, get_example_data_path, get_version

config_path = get_default_config_path()  # Auto-detects local or package default
data_path = get_example_data_path()      # Returns imcui/datasets path
version = get_version()                  # Returns current version string
```

### Adding New Algorithms

> **Note:** This WebUI no longer maintains matching algorithms. All matchers are maintained in the [vismatch](https://github.com/gmberton/vismatch) repository by [@gmberton](https://github.com/gmberton). To add new matchers, please contribute to the vismatch repository.

## Configuration Precedence

Config files are loaded in this order (first found):
1. Custom path via `-c` flag
2. `cwd/app.yaml` (current working directory)
3. `cwd/config/app.yaml` (current working directory config subdirectory)
4. Package default: `imcui/config/app.yaml`

This logic is shared across all entry points (CLI and app.py) via `imcui.utils.config.get_default_config_path()`.

## Code Quality Notes

**Avoid Code Duplication**: The codebase was refactored to eliminate duplicate logic. All configuration and path management is centralized in `imcui/utils/config.py`.

**Version Management**: Never define version in multiple places. The single source of truth is `pyproject.toml`, and `imcui.__version__` reads it dynamically.

**Entry Points**: Both `app.py` and `imcui/cli/main.py` should import shared utilities from `imcui` rather than implementing their own versions.

## Cleanup

The following files/directories can be safely deleted to save space:
- `experiments/` - Debug output directory
- `log.txt` - Runtime logs
- `output.pkl` - Cached output

The following should NOT be deleted:
- `imcui/datasets/` - Example datasets (required for testing)
- `cpp/test/` - C++ test code for API
- `assets/gui.jpg` - Screenshot for README

## Important Files

**Configuration Management**:
- `imcui/utils/config.py` - Centralized utility functions for paths, version, configuration
- `imcui/config/app.yaml` - Default configuration for the application
- `imcui/config/api.yaml` - Configuration for API server

**Entry Points**:
- `imcui/cli/main.py` - Primary CLI entry point (uses `imcui.utils.config`)
- `app.py` - Secondary entry point for HuggingFace Spaces (uses `imcui.utils.config`)
- Both share identical configuration loading logic

**Version Management**:
- `pyproject.toml` - Single source of truth for version (1.0.0)
- `imcui/__init__.py` - Reads version from installed package or pyproject.toml
- Avoid defining version in `imcui/ui/__init__.py` or elsewhere

**Deprecated Files**:
- `environment.yaml` - Deprecated, includes deprecation notice. Recommend pip install instead.
