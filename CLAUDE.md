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

**Note**: Example datasets (82MB) are **automatically downloaded** from HuggingFace on first run to your user cache directory (`~/.cache/imcui/datasets/` on Linux/macOS, `%LOCALAPPDATA%\imcui\datasets\` on Windows). To use local datasets, set `IMCUI_DATA_DIR` environment variable or use `-d` flag.

### CLI Options
| Flag | Default | Description |
|------|---------|-------------|
| `-s, --server-name` | `0.0.0.0` | Host to bind |
| `-p, --server-port` | `7860` | Port to run on |
| `-c, --config` | Auto-detected | Custom config YAML path |
| `-d, --example-data-root` | Auto-resolved | Example datasets directory (auto-download if not found) |
| `-v, --verbose` | `False` | Enable verbose output |

> **Note:** Both `imcui` CLI and `python app.py` support the same command-line options.

### Development
```bash
# Run tests
pytest tests/ -v

# Run a single test
pytest tests/test_image_matching.py::test_one -v

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
  - `ui/` - Gradio-based web interface (image_matching_app.py, config_utils.py, etc.)
  - `api/` - Core matching API (ImageMatchingAPI class, server.py, client.py)
  - `config/` - Configuration files (app.yaml, api.yaml)
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
- `imcui` CLI (recommended) - uses shared utilities from `imcui.ui`
- `python app.py` - HuggingFace Spaces entry point, also uses shared utilities
- Both entry points share identical configuration loading logic

**Version Management**: Single source of truth in `pyproject.toml` (currently 1.0.0). The `imcui.__version__` reads from installed package metadata or falls back to pyproject.toml for development.

**API Usage**:
```python
from imcui.api import ImageMatchingAPI
from imcui.ui import DEVICE, get_matcher_zoo

matcher_zoo = get_matcher_zoo()  # Dynamically loaded from vismatch
api = ImageMatchingAPI(conf=matcher_zoo["superpoint-lightglue"], device=DEVICE)
pred = api(image0, image1)  # RGB numpy arrays
```

**UI Module Structure** (`imcui/ui/`):

After refactoring, the monolithic `utils.py` was split into focused modules:

| File | Purpose |
|------|---------|
| `image_matching_app.py` | Main Gradio application class (`ImageMatchingApp`) |
| `config.py` | Configuration constants and matching functions |
| `config_utils.py` | Path resolution, version management, auto-download |
| `matching.py` | Matching logic, RANSAC filtering, model cache |
| `geometry.py` | Geometry estimation (Homography, Fundamental matrix) |
| `image_utils.py` | Image processing utilities (warp, rotate, scale) |
| `examples.py` | Example dataset generation |
| `visualization.py` | Visualization utilities (keypoints, matches) |
| `model_cache.py` | Model caching (ARCSizeAwareModelCache, LRUModelCache) |
| `sfm_engine.py` | SfM engine (temporarily disabled) |

**Utility Functions** (available from top-level imports):
```python
from imcui import get_default_config_path, get_example_data_path, get_version

config_path = get_default_config_path()  # Auto-detects local or package default
data_path = get_example_data_path()      # Auto-resolves datasets path with download support
version = get_version()                  # Returns current version string
```

**Example Datasets Management**:
- **PyPI Package**: Datasets (82MB) are **excluded** from PyPI to keep package size small
- **Auto-download**: On first run, datasets are automatically downloaded from HuggingFace to user cache:
  - Linux/macOS: `~/.cache/imcui/datasets/`
  - Windows: `%LOCALAPPDATA%\imcui\datasets\`
- **Development Mode**: Git clone includes datasets locally at `imcui/datasets/`
- **Custom Path**: Use `IMCUI_DATA_DIR` env var or `-d` CLI flag
- **Resolution Order**: See `imcui/ui/config_utils.py:get_example_data_path()` for details

### Adding New Algorithms

> **Note:** This WebUI no longer maintains matching algorithms. All matchers are maintained in the [vismatch](https://github.com/gmberton/vismatch) repository by [@gmberton](https://github.com/gmberton). To add new matchers, please contribute to the vismatch repository.

## Configuration Precedence

Config files are loaded in this order (first found):
1. Custom path via `-c` flag
2. `cwd/app.yaml` (current working directory)
3. `cwd/config/app.yaml` (current working directory config subdirectory)
4. Package default: `imcui/config/app.yaml`

This logic is shared across all entry points (CLI and app.py) via `imcui.ui.config_utils.get_default_config_path()`.

## Example Datasets Resolution

Dataset path is resolved via `imcui.ui.config_utils.get_example_data_path()` in this order:

1. **Environment Variable**: `IMCUI_DATA_DIR` if set
2. **User Cache**: Auto-download from HuggingFace on first run
   - Linux/macOS: `~/.cache/imcui/datasets/`
   - Windows: `%LOCALAPPDATA%\imcui\datasets\`

**Notes**:
- **All users** (including developers) use auto-download by default
- Download requires `datasets` package: `pip install datasets` (or `pip install imcui[datasets]`)
- Download progress and target directory are logged clearly
- Gradio `allowed_paths` is automatically configured to include cache directory
- For offline use, set `IMCUI_DATA_DIR` to a custom directory with pre-downloaded data

## Code Quality Notes

**Avoid Code Duplication**: The codebase was refactored to eliminate duplicate logic. All configuration and path management is centralized in `imcui/ui/config_utils.py`. The monolithic `utils.py` was split into modular components.

**Version Management**: Never define version in multiple places. The single source of truth is `pyproject.toml`, and `imcui.__version__` reads it dynamically.

**Entry Points**: Both `app.py` and `imcui/cli/main.py` should import shared utilities from `imcui` rather than implementing their own versions.

**Exports via `imcui.ui`**: After the refactoring, `imcui/ui/__init__.py` exports all commonly used utilities for easy access:
- `DEVICE`, `GRADIO_VERSION` - Device and version constants
- `get_matcher_zoo`, `get_available_model_names` - Matcher management
- `filter_matches`, `compute_geometry` - Geometry utilities
- `run_matching`, `run_ransac`, `send_to_match` - Matching functions
- `wrap_images`, `generate_warp_images` - Image processing
- `display_keypoints`, `display_matches`, `plot_images` - Visualization
- `ARCSizeAwareModelCache`, `LRUModelCache` - Model caching
- Path functions from `config_utils`: `get_default_config_path`, `get_example_data_path`, `get_version`

## Cleanup

The following files/directories can be safely deleted to save space:
- `experiments/` - Debug output directory
- `log.txt` - Runtime logs
- `output.pkl` - Cached output

The following should NOT be deleted:
- `cpp/test/` - C++ test code for API
- `assets/gui.jpg` - Screenshot for README
- User cache directory (auto-created on first run):
  - Linux/macOS: `~/.cache/imcui/datasets/`
  - Windows: `%LOCALAPPDATA%\imcui\datasets\`

## Important Files

**UI Modules** (`imcui/ui/`):
- `image_matching_app.py` - Main Gradio application class (`ImageMatchingApp`)
- `config.py` - Configuration constants and matching utilities
- `config_utils.py` - Path resolution, version management, auto-download
- `matching.py` - Matching logic, RANSAC, model cache
- `geometry.py` - Geometry estimation (Homography, Fundamental matrix)
- `image_utils.py` - Image processing utilities
- `examples.py` - Example dataset generation
- `visualization.py` - Visualization utilities
- `model_cache.py` - Model caching classes

**Configuration**:
- `imcui/config/app.yaml` - Default configuration for the application
- `imcui/config/api.yaml` - Configuration for API server

**Entry Points**:
- `imcui/cli/main.py` - Primary CLI entry point
- `app.py` - Secondary entry point for HuggingFace Spaces
- Both use shared utilities from `imcui.ui`

**Version Management**:
- `pyproject.toml` - Single source of truth for version (1.0.0)
- `imcui/__init__.py` - Reads version from installed package or pyproject.toml
- Avoid defining version in `imcui/ui/__init__.py` or elsewhere

**Deprecated Files**:
- `environment.yaml` - Deprecated, includes deprecation notice. Recommend pip install instead.
- `imcui/ui/utils.py` - Removed, split into modular components above.

**Gradio Security Configuration**:
- `imcui/ui/image_matching_app.py:run()` - Dynamically adds external dataset paths to `allowed_paths`
- Ensures Gradio can serve images from cache directory (`~/.cache/imcui/datasets/`)
- Only adds cache directory if it's outside the package root
