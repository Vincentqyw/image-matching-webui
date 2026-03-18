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
- `imcui/hloc/` - Feature extraction and matching (adapted from Hierarchical-Localization)
- `imcui/hloc/extractors/` - Sparse feature extractors (SuperPoint, DISK, ALIKED, etc.)
- `imcui/hloc/matchers/` - Sparse/dense matchers (LightGlue, LoFTR, RoMa, etc.)
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

The project now uses [vismatch](https://github.com/Vincentqyw/vismatch) pip package for algorithms. To add new algorithms:

1. Install the vismatch package with your custom algorithm
2. Register in config file under `matcher_zoo`:
3. Or add custom feature extractor to `imcui/hloc/extractors/` following existing patterns

```yaml
matcher_zoo:
  my_algorithm:
    matcher: my-matcher-name
    feature: my-feature-name  # only for sparse matchers
    dense: false  # true for dense methods like LoFTR
    info:
      name: My Algorithm
      source: "CVPR 2024"
      github: https://github.com/example
      display: true
      efficiency: high
```

## Configuration Precedence

Config files are loaded in this order (first found):
1. Custom path via `--config` flag
2. `config.yaml` in current directory
3. `config/config.yaml` in current directory
4. Default: `imcui/config/app.yaml`
