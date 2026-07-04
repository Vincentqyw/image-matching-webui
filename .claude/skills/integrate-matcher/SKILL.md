---
name: "integrate-matcher"
description: "Integrate a new image matching model into the image-matching-webui project. Invoke when user provides a GitHub repo URL of a feature matching method and asks to add/integrate it."
---

# Integrate Matcher into image-matching-webui

This skill automates the integration of a new local feature matching method into the `image-matching-webui` project. Given a GitHub repository URL, it follows the project's established patterns to add the matcher as a fully functional option in the WebUI.

## Prerequisites

- The target repository must be a local feature matching method (sparse or standalone)
- You must be working inside the `image-matching-webui` project root

## Step-by-Step Integration Guide

### Step 1: Analyze the Target Repository

1. **Clone the repo** to `/tmp/<repo-name>` for analysis (do NOT add as submodule yet)
2. **Identify the matcher type**:
   - **Dense/Standalone matcher**: Takes raw images as input, performs detect+describe+match internally (e.g., LoMa, RoMa, LoFTR). `required_inputs = ["image0", "image1"]`
   - **Sparse matcher**: Takes keypoints+descriptors as input, performs matching only (e.g., LightGlue, SuperGlue). `required_inputs` includes `keypoints0`, `descriptors0`, etc.
3. **Find the core model class** and its API:
   - How to initialize the model (constructor args, config options)
   - How to run inference (forward method signature)
   - What the model outputs (keypoints, matches, scores, etc.)
   - Model weight download URLs
4. **Check dependencies** in `pyproject.toml` or `requirements.txt`
5. **Identify model variants** (e.g., different sizes: B/L/G/R)
6. **Check for device/amp issues**:
   - Does the model use `torch.autocast` or mixed precision (`mp`, `amp`)?
   - Does it manage its own device placement (like LoMa's `loma.device`)?
   - On MPS/CPU, does it produce dtype mismatches?

### Step 2: Add Git Submodule

```bash
git submodule add <repo-url> imcui/third_party/<RepoName>
```

- Use the original repo name (PascalCase) as the submodule directory name
- If the repo has a fork in the project's org (e.g., `Vincentqyw/xxx` or `agipro/xxx`), prefer the fork

#### ⚠️ CRITICAL: Never modify third_party code directly

`imcui/third_party/` contains **pinned third-party submodules** — you are NOT the owner of this code. If a dependency needs a compatibility fix (e.g., API changes in PyTorch/kornia, import path changes):

1. **Fork** the original repo to the `agipro` GitHub account
2. **Apply the fix** in the fork and push
3. **Replace** the submodule URL in the main repo to point to the fork:
   ```bash
   git submodule deinit -f imcui/third_party/<RepoName>
   git rm -f imcui/third_party/<RepoName>
   rm -rf .git/modules/imcui/third_party/<RepoName>
   git submodule add https://github.com/agipro/<RepoName>.git imcui/third_party/<RepoName>
   ```
4. **Update `.gitmodules`** — the URL must point to the fork

Example: `EfficientLoFTR` was forked to `agipro/EfficientLoFTR` to fix a `kornia.utils.grid` → `kornia.utils` import change required by kornia 0.8+.

### Step 3: Create Matcher Implementation

Create `imcui/hloc/matchers/<matcher-name>.py` following these rules:

#### 3.1 Import Pattern

```python
import sys
from pathlib import Path

from .. import logger
from ..utils.base_model import BaseModel

# Add third_party to sys.path for import
<matcher>_path = Path(__file__).parent / "../../third_party/<RepoName>/src"
sys.path.append(str(<matcher>_path))

# Import the model class
from <module> import <ModelClass>
```

- If the third_party repo has `src/` as the package root, append `<path>/src`
- Some repos put the package directly at root level — adjust accordingly

#### 3.2 Class Definition

The class MUST:
- Inherit from `BaseModel`
- Be the **only** `BaseModel` subclass in the file (the `dynamic_load` function asserts this)
- Define `default_conf` dict with all configurable parameters
- Define `required_inputs` list

```python
class MatcherName(BaseModel):
    default_conf = {
        "name": "two_view_pipeline",
        "model_name": "<default_variant>",
        "max_keypoints": 2048,
        # ... other config
    }
    required_inputs = [
        "image0",
        "image1",
        # For sparse matchers, also include:
        # "keypoints0", "scores0", "descriptors0",
        # "keypoints1", "scores1", "descriptors1",
    ]
```

#### 3.3 `_init` Method

- Use `self.conf` (not the `conf` parameter) to access merged config
- Log model loading with `logger.info()`
- Handle model weight downloading:
  - If weights are on HuggingFace: use `self._download_model(repo_id=MODEL_REPO_ID, filename=...)`
  - If weights are auto-downloaded by the model (e.g., `torch.hub`), let it handle
  - If weights need manual download, document the URL
- **Platform compatibility**: On non-CUDA devices, disable mixed precision:
  ```python
  if not torch.cuda.is_available():
      # Disable mp/amp for CPU/MPS compatibility
  ```

#### 3.4 `_forward` Method

**For standalone matchers** (input: raw images):

The `data` dict contains preprocessed tensors:
- `data["image0"]`: shape `(1, C, H, W)`, float32, range [0, 1]
- `data["image1"]: shape `(1, C, H, W)`, float32, range [0, 1]

If the model expects PIL images or file paths, convert:
```python
img0 = data["image0"].cpu().numpy().squeeze() * 255
img0 = img0.transpose(1, 2, 0)  # CHW -> HWC
img0 = Image.fromarray(img0.astype("uint8"))
```

**For sparse matchers** (input: keypoints + descriptors):
```python
# Repackage data for the model's expected format
input = {
    "image0": {"image": data["image0"], "keypoints": data["keypoints0"], ...},
    "image1": {"image": data["image1"], "keypoints": data["keypoints1"], ...},
}
return self.net(input)
```

#### 3.5 Output Format

The `_forward` method MUST return a dict with these keys:

**Dense matchers that output matched keypoints only:**
```python
pred = {
    "keypoints0": kpts0,      # torch.Tensor, shape (N, 2), pixel coords in resized image
    "keypoints1": kpts1,      # torch.Tensor, shape (N, 2)
    "mconf": confidence,      # torch.Tensor, shape (N,), match confidence scores
}
```

**Dense matchers that can separate detected vs matched keypoints (PREFERRED):**
```python
pred = {
    "keypoints0": all_kpts0,     # All detected keypoints (for UI "Keypoints" display)
    "keypoints1": all_kpts1,     # All detected keypoints
    "mkeypoints0": matched_kpts0, # Matched keypoints (for UI match lines)
    "mkeypoints1": matched_kpts1, # Matched keypoints
    "mconf": confidence,          # Match confidence scores
}
```

**Sparse matchers** (LightGlue-style):
```python
# Return the model's raw output — match_dense.py handles the rest
return self.net(input)
```

Key output rules:
- `keypoints0/1`: ALL detected keypoints (shown in UI "Open for More: Keypoints")
- `mkeypoints0/1`: Only MATCHED keypoints (shown as match lines in UI)
- If `mkeypoints0/1` is missing, `match_dense.py` falls back to `keypoints0/1`
- Coordinates must be in **pixel space** of the resized image (not normalized [-1,1])
- `mconf` should be real confidence scores (not all-ones)

### Step 4: Add Matcher Configuration

#### 4.1 `imcui/hloc/configs/matchers.py`

Add a configuration entry for each model variant:

```python
"<matcher-name>": {
    "output": "matches-<matcher-name>",
    "model": {
        "name": "<matcher-module-name>",  # Must match the .py filename in matchers/
        "model_name": "<variant>",         # Passed as conf["model_name"]
        "max_keypoints": 2048,
        # ... other model-specific config
    },
    "preprocessing": {
        "grayscale": False,     # True for LoFTR-style; False for most modern matchers
        "force_resize": True,
        "resize_max": 1024,
        "width": 640,
        "height": 480,
        "dfactor": 8,          # Image dimensions must be divisible by this
    },
},
```

Key rules:
- `"name"` in `model` must match the Python filename (e.g., `"loma"` → `loma.py`)
- Each model variant gets its own top-level entry (e.g., `loma-b`, `loma-l`, `loma-g`)
- `preprocessing.grayscale`: Set `True` only for models that expect 1-channel input (e.g., LoFTR)
- `preprocessing.dfactor`: Ensure resized dimensions are divisible by this value

#### 4.2 `config/app.yaml` (local dev) AND `imcui/config/app.yaml` (package default)

BOTH files must be updated identically. Add an entry under `matcher_zoo`:

```yaml
<DisplayName>:
  matcher: <matcher-config-name>    # Must match key in matchers.py
  standalone: true                   # true = takes two images directly (no separate extractor needed)
  info:
    name: <DisplayName>              # Display name in WebUI dropdown
    source: "Venue Year"             # e.g., "ECCV 2026", "ICCV 2023"
    paper: <paper-url>               # arXiv or published paper URL
    github: <github-url>             # Official GitHub repo
    display: true                    # Whether to show in WebUI
    efficiency: medium               # low (heavy), medium, high (fast)
```

Key rules:
- `standalone: true` means the matcher takes two raw images directly (no separate feature extractor needed)
- `standalone: false` means the matcher requires a feature extractor (e.g., SuperPoint+LightGlue)
- For feature+matcher combos, use `<extractor>+<matcher>` format (e.g., `superpoint+lightglue`)
- Set `enable: false` for very heavy models that most users won't use by default
- **CRITICAL**: Both `config/app.yaml` and `imcui/config/app.yaml` must be kept in sync

### Step 5: Handle Platform Compatibility

Common issues and fixes:

#### 5.1 Mixed Precision (MP/AMP) Issues

On MPS/CPU, `torch.autocast` with float16/bfloat16 causes:
- `RuntimeError: Input type (c10::Half) and bias type (float) should be the same`
- `RuntimeError: Input type (MPSFloatType) and weight type (torch.FloatTensor) should be the same`

Fix pattern (before importing third-party code):
```python
# Patch module-level amp_dtype before import
import <module>.device as _device
if not torch.cuda.is_available():
    _device.amp_dtype = torch.float32

# Also patch submodule local bindings
import <module>.submod as _submod
if not torch.cuda.is_available():
    _submod.amp_dtype = torch.float32
```

After model construction:
```python
if not torch.cuda.is_available():
    cfg = dataclasses.replace(cfg, mp=False)
    for module in self.net.modules():
        if hasattr(module, "amp"):
            module.amp = False
```

#### 5.2 Device Mismatch Issues

If the model manages its own device (like LoMa's `loma.device`):
```python
# Override .to() to keep model on its expected device
def to(self, device=None, **kwargs):
    return super().to(<model_expected_device>, **kwargs)
```

#### 5.3 Inference Mode vs No Grad

If a model uses `@torch.inference_mode()` but you need to pass its outputs to another module that requires grad tracking:
```python
with torch.no_grad():
    output = model.detect_and_describe(...)
    output = output.clone()  # Detach from inference mode graph
```

### Step 6: Verification Checklist

After integration, verify:

1. **Import test**: `python -c "from imcui.hloc.matchers import <name>"`
2. **Model loading**: The model loads without errors on CPU/MPS/CUDA
3. **Inference test**: Run matching on a test image pair
4. **WebUI test**: `python app.py` → select the matcher → run matching
5. **Keypoints display**: UI "Open for More: Keypoints" shows detected keypoints
6. **Match lines display**: UI shows correct match lines between images
7. **Both config files**: `config/app.yaml` and `imcui/config/app.yaml` are identical for the new matcher

## File Change Summary

For each integration, these files are typically modified:

| File | Action | Description |
|------|--------|-------------|
| `.gitmodules` | Modify | Add submodule entry |
| `imcui/third_party/<RepoName>` | Add | Git submodule |
| `imcui/hloc/matchers/<name>.py` | Create | Matcher implementation |
| `imcui/hloc/configs/matchers.py` | Modify | Add matcher config entries |
| `config/app.yaml` | Modify | Add WebUI display config |
| `imcui/config/app.yaml` | Modify | Add WebUI display config (must match config/app.yaml) |

## Reference Implementations

| Pattern | File | Description |
|---------|------|-------------|
| Dense standalone | `imcui/hloc/matchers/roma.py` | RoMa — raw images → matched keypoints |
| Dense standalone + separate detected/matched kpts | `imcui/hloc/matchers/loma.py` | LoMa — separates all detected from matched keypoints |
| Sparse matcher | `imcui/hloc/matchers/lightglue.py` | LightGlue — keypoints+descriptors → matches |
| Detector-only + external descriptor | `imcui/hloc/extractors/raco.py` | RaCo — detects keypoints, delegates descriptor to ALIKED |

### RaCo pattern: detector that needs a descriptor extractor

Some models are **keypoint detectors only** — they output keypoints/scores but no descriptors. The RaCo integration demonstrates chaining RaCo detection with ALIKED description:

1. Create an **extractor** (not matcher) in `imcui/hloc/extractors/raco.py`
2. In `_forward`, first run RaCo detection → keypoints, then run ALIKED descriptor on those keypoints
3. Register in `configs/extractors.py` with `max_num_keypoints` / `nms_radius` parameters
4. Register a matcher config in `configs/matchers.py` with `features: "raco-aliked"` — a custom LightGlue+ checkpoint trained for RaCo+ALIKED features
5. In `app.yaml`, the entry uses `feature: raco` (the extractor) + `matcher: raco-lightglue` (the LightGlue variant)

### Fork priority for third-party fixes

When a submodule needs a compatibility fix:

| Repo owner | Action |
|------------|--------|
| `Vincentqyw/*` | Push fix directly to the Vincentqyw fork |
| `agipro/*` | Push fix directly to the agipro fork |
| Anyone else | Fork to `agipro`, apply fix, switch submodule URL |

Always verify push succeeded with `git log --oneline -1` in the submodule directory.

### Kornia compatibility (kornia >= 0.8)

Kornia 0.8.0 removed the `kornia.utils.grid` submodule. `create_meshgrid` moved to `kornia.utils` (0.8.0-0.8.2), then to `kornia.geometry` (0.8.3+). When fixing submodule imports, use this future-proof pattern:

```python
try:
    from kornia.geometry import create_meshgrid  # kornia >= 0.8.3
except ImportError:
    from kornia.utils import create_meshgrid  # kornia < 0.8.3
```
