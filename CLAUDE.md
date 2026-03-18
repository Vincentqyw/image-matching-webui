# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Image Matching WebUI (IMCUI) is a Gradio-based web interface for matching image pairs using various state-of-the-art computer vision algorithms. It supports both sparse (keypoint-based) and dense (learned) matching methods.

## Common Commands

### Installation
```bash
# From source
git clone https://github.com/Vincentqyw/image-matching-webui.git
cd image-matching-webui
conda env create -f environment.yaml
conda activate imcui
pip install -e .

# Or from pip
pip install imcui
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

# Custom config and options
imcui -c config.yaml -s 127.0.0.1 -d ./datasets -v

# Docker
docker-compose up webui
```

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
- `imcui/ui/` - Gradio-based web interface (app_class.py, utils.py)
- `imcui/api/` - Core matching API (ImageMatchingAPI class)
- `imcui/config/` - Configuration files (app.yaml, api.yaml)
- `imcui/assets/` - Static assets (logo, etc.)
- `imcui/datasets/` - Example datasets for testing

### Key Concepts

**Sparse vs Dense Matching**: Sparse methods extract discrete keypoints and match them. Dense methods work on full image correlations.

**Matcher Zoo**: All matchers are dynamically loaded from [vismatch](https://github.com/gmberton/vismatch) package (by [@gmberton](https://github.com/gmberton)). This WebUI no longer maintains matching algorithms.

**API Usage**:
```python
from imcui.api import ImageMatchingAPI
from imcui.ui.utils import DEVICE, get_matcher_zoo

matcher_zoo = get_matcher_zoo()  # Dynamically loaded from vismatch
api = ImageMatchingAPI(conf=matcher_zoo["superpoint-lightglue"], device=DEVICE)
pred = api(image0, image1)  # RGB numpy arrays
```

### Adding New Algorithms

> **Note:** This WebUI no longer maintains matching algorithms. All matchers are maintained in the [vismatch](https://github.com/gmberton/vismatch) repository by [@gmberton](https://github.com/gmberton). To add new matchers, please contribute to the vismatch repository.

## Configuration Precedence

Config files are loaded in this order (first found):
1. Custom path via `-c` flag
2. Default: `imcui/config/app.yaml`
