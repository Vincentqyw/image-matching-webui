# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Install from source
conda env create -f environment.yaml && conda activate imcui
pip install -e .

# Run the WebUI (default: http://localhost:7860)
imcui                          # package CLI
python app.py                  # direct script
imcui -s 127.0.0.1 -p 8080    # custom host/port

# Lint and format
pre-commit run -a             # Run all hooks (ruff, mypy, yaml-check, etc.)

# Run tests
python -m pytest tests/ -s
python -m pytest tests/test_basic.py -s  # single test file

# Docker
docker-compose up webui       # Web UI service
docker-compose up -d webui    # Background
docker-compose build webui    # Rebuild

# Git submodules
git submodule update --init --recursive          # clone all
git submodule update --remote <path>             # update one
```

## Architecture

This is a Gradio WebUI for image matching built on a modified **Hierarchical-Localization (hloc)** framework. The core pattern: **extractors** detect keypoints+descriptors from images, **matchers** pair them. Dense matchers bypass extraction and operate directly on images.

### Directory layout

```
imcui/
  hloc/                  # Core engine (forked from cvg/Hierarchical-Localization)
    extractors/          # Keypoint detector + descriptor models (ALIKED, DISK, SuperPoint, RaCo…)
    matchers/            # Matching models (LightGlue, LoFTR, RoMa, SuperGlue, LoMa…)
    configs/
      extractors.py      # confs dict: maps extractor config names → model+preprocessing
      matchers.py        # confs dict: maps matcher config names → model+preprocessing
    pipelines/           # SfM reconstruction pipelines (unused by WebUI)
    utils/base_model.py  # BaseModel ABC: all models inherit _init / _forward pattern
  ui/
    app_class.py         # Gradio UI definition (ImageMatchingApp, AppSfmUI)
    utils.py             # run_matching pipeline, RANSAC, visualization, model loading
    modelcache.py        # ARCSizeAwareModelCache: GPU/CPU memory-aware model LRU cache
  cli/main.py            # Click CLI entry point (the `imcui` command)
  config/app.yaml        # Package default config — declarative matcher zoo registry
  third_party/           # Git submodules (30+ upstream repos)
config/app.yaml          # User-facing config (mirrors the package default)
```

### Two matching pipelines

**Sparse** — feature extraction then matching:
1. `extract_features.py` → instantiate extractor (e.g., SuperPoint, DISK, ALIKED)
2. Extract keypoints+descriptors from each image
3. `match_features.py` → instantiate matcher (e.g., LightGlue, SuperGlue, NN-mutual)
4. Return matched keypoints

**Dense** — single model processes both images end-to-end:
1. `match_dense.py` → instantiate dense matcher (e.g., LoFTR, RoMa, LoMa, DKM)
2. Model takes `image0` + `image1`, returns keypoints + matches in one shot

### Config system

`app.yaml` → `parse_match_config()` → resolves string names to actual config dicts from `hloc/configs/extractors.py` or `hloc/configs/matchers.py`.

Each matcher zoo entry:
```yaml
matcher_zoo:
  disk+lightglue:
    matcher: disk-lightglue      # key into matchers.confs
    feature: disk                # key into extractors.confs (sparse only)
    standalone: false           # needs separate feature extractor
    skip_ci: false               # skip in CI tests
```

### BaseModel pattern (extractors and matchers)

All models inherit from `BaseModel` (in `hloc/utils/base_model.py`):
- `default_conf` — default parameters merged with user config
- `required_inputs` — validates keys in forward data dict
- `_init(conf)` — load weights, set up model
- `_forward(data)` — run inference

Model weights are downloaded from HuggingFace Hub (`Realcat/imcui_checkpoints`) via `hf_hub_download`.

### Adding a new matcher

1. Add the upstream repo as a submodule: `git submodule add <url> imcui/third_party/<Name>`
2. Create extractor module (`imcui/hloc/extractors/<name>.py`) and/or matcher module (`imcui/hloc/matchers/<name>.py`) following the `example.py` template
3. Register configs in `configs/extractors.py` and/or `configs/matchers.py`
4. Add entry to `config/app.yaml` and `imcui/config/app.yaml`
5. Upload model weights to `Realcat/imcui_checkpoints` on HuggingFace

See `.claude/skills/integrate-matcher/SKILL.md` for the detailed procedure.

### Third-party code rule

**Never modify files inside `imcui/third_party/`.** Those are pinned git submodules. If a dependency needs a compatibility fix:
1. Fork the repo to `agipro` (or push to existing Vincentqyw fork)
2. Apply the fix in the fork
3. Replace the submodule URL: `deinit -f` → `rm` → delete `.git/modules/` entry → `submodule add` new URL
4. Update `.gitmodules`

### RANSAC

Two backends: OpenCV (`CV2_USAC_MAGSAC` default) and poselib. Both `Homography` and `Fundamental` matrix estimation are supported.

### Model caching

`modelcache.py` implements an LRU cache that monitors GPU/CPU memory and evicts models when near capacity. Models are cached by `cache_key = "{matcher_name}_{model_name}"`.

### CI

GitHub Actions run `pip install .` then `pytest tests/ -s --timeout=1200` on Python 3.10/3.11/3.12 (Ubuntu CPU-only). Models with `skip_ci: true` in `app.yaml` are excluded from CI testing.

### Python version constraint

`requires-python = ">=3.10,<3.13"` — Python 3.13 is not yet supported.
