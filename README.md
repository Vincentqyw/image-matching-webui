---
title: Image Matching WebUI
emoji: 👀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.19.0"
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
---

# Image Matching WebUI

A Gradio WebUI for image matching using state-of-the-art algorithms.

## Quick Start

```bash
pip install imcui
imcui
```

This Space runs image matching with 30+ algorithms including: LightGlue, LoFTR, RoMa, DKM, OmniGlue, GlueStick, Mast3R, DUSt3R, GIM, and many more.

## Supported Matchers

| Category | Algorithms |
|----------|------------|
| Sparse | SuperPoint+LightGlue, DISK+LightGlue, ALIKED+LightGlue, SuperGlue, SIFT, XFeat |
| Dense | LoFTR, SE2-LoFTR, Efficient LoFTR, RoMa, RoMaV2, DKM, LoMa |
| End-to-End | OmniGlue, GlueStick, Mast3R, DUSt3R, GIM |
| Efficient | MINIMA variants, XFeat, ALIKED |

## Links

- 📦 [PyPI](https://pypi.org/project/imcui/)
- 💻 [GitHub](https://github.com/Vincentqyw/image-matching-webui)
- 📄 [Paper (RoMa)](https://arxiv.org/abs/2305.15404)
