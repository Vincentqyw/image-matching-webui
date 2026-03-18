# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Image Matching WebUI (IMCUI) is a Gradio-based web interface for matching image pairs using various state-of-the-art computer vision algorithms. It supports both sparse (keypoint-based) and dense (learned) matching methods.

## Common Commands

### Installation
```bash
# From source (requires recursive clone for submodules)
git clone --recursive https://github.com/Vincentqyw/image-matching-webui.git
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

# Docker
docker-compose up webui
```

### Development
```bash
# Run tests
pytest tests/ -v

# Run a single test
pytest tests/test_basic.py::test_one -v

# Run pre-commit checks
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
- `imcui/hloc/` - Feature extraction and matching (adapted from Hierarchical-Localization)
- `imcui/hloc/extractors/` - Sparse feature extractors (SuperPoint, DISK, ALIKED, etc.)
- `imcui/hloc/matchers/` - Sparse/dense matchers (LightGlue, LoFTR, RoMa, etc.)
- `imcui/third_party/` - Git submodules for external algorithms
- `config/app.yaml` - Matcher configuration and defaults

### Key Concepts

**Sparse vs Dense Matching**: Sparse methods extract discrete keypoints and match them (SuperPoint+LightGlue). Dense methods work on full image correlations (LoFTR, RoMa).

**Matcher Zoo**: All algorithms are configured in `config/app.yaml` under `matcher_zoo`. Each entry specifies:
- `matcher`: The matching method
- `feature`: For sparse matchers, the feature extractor
- `dense`: Boolean indicating sparse (false) or dense (true) approach
- `enable`: Whether to show in UI

**API Usage**:
```python
from imcui.api import ImageMatchingAPI
from imcui.ui.utils import DEVICE, load_config

config = load_config("config/app.yaml")
api = ImageMatchingAPI(conf=config["matcher_zoo"]["disk+lightglue"], device=DEVICE)
pred = api(image0, image1)  # RGB numpy arrays
```

### Adding New Algorithms

1. Add feature extractor to `imcui/hloc/extractors/` following existing patterns
2. Add matcher to `imcui/hloc/matchers/` (dense) or modify match_features.py (sparse)
3. Register in `config/app.yaml` under `matcher_zoo`

## Configuration Precedence

Config files are loaded in this order (first found):
1. Custom path via `--config` flag
2. `config.yaml` in current directory
3. `config/config.yaml` in current directory
4. Default: `imcui/config/app.yaml`
