---
name: 🔗 Add New Matcher
about: Request integration of a new image matching model
title: "add matcher: <NAME>"
labels: ["new-matcher", "enhancement"]
assignees: ""
---

## 📌 Repository

<!-- Required: paste the GitHub repository URL of the matcher -->
Repository URL:

<!-- Required: link to the paper -->
Paper (arXiv / conference):

## 🔧 Matcher Info

<!-- Required -->
Type:
- [ ] **Sparse** (extractor + matcher, e.g., SuperPoint+LightGlue)
- [ ] **Dense / Standalone** (end-to-end, e.g., LoFTR, RoMa)

<!-- Optional but helpful -->
Efficiency:
- [ ] High (real-time)
- [ ] Medium
- [ ] Low (slow, research-grade)

## ✅ Checklist

<!-- Mark with `[x]` what applies -->

- [ ] The repository has an open-source license (MIT, Apache, BSD, etc.)
- [ ] The model has pretrained weights available
- [ ] The code is compatible with PyTorch
- [ ] I understand the matcher will be added as a git submodule under `imcui/third_party/`

## 📝 Notes

<!-- Any special dependencies, known issues, or additional context? -->
