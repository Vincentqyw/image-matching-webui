---
name: "deploy-hf"
description: "Release a new imcui version and deploy to HuggingFace Spaces (test → prod). Invoke when user asks to release, deploy, or ship a new version."
---

# Deploy to HuggingFace Spaces

This skill handles the full release pipeline: version bump → tag → release → deploy to HF test → deploy to HF prod.

## Prerequisites

- Working inside `image-matching-webui` project root
- Authenticated with GitHub (`gh`) and HuggingFace (`hf`)
- HF git credential configured (see branch setup below)

## Branch Setup (one-time)

```bash
# Add HF remotes (only needed once)
git remote add hf-test https://huggingface.co/spaces/Realcat/imcui
git remote add hf-prod https://huggingface.co/spaces/Realcat/image-matching-webui

# Configure git credential for HF
git config --global credential.https://huggingface.co.username Realcat
```

## Release & Deploy Workflow

### Step 1: Release new version on GitHub

```bash
# On main branch
# 1. Bump version in pyproject.toml
# 2. Add any new deps to requirements.txt
# 3. Commit
git add pyproject.toml requirements.txt
git commit -m "chore: bump version to X.Y.Z"
git push origin main

# 4. Tag and release
git tag -a vX.Y.Z -m "vX.Y.Z: <description>"
git push origin vX.Y.Z
gh release create vX.Y.Z --title "vX.Y.Z" --notes "..."
```

This triggers two CI workflows:
- `docker-publish.yml` (on tag `v*`) → Docker Hub image
- `release.yml` (on release published) → PyPI whl

### Step 2: Update huggingface branch

```bash
git checkout huggingface
# Update requirements.txt: imcui==X.Y.Z
git add requirements.txt
git commit -m "chore: bump imcui to X.Y.Z"
git push origin huggingface
```

### Step 3: Deploy to test Space

Wait for PyPI release to complete (~10 min), then:

```bash
git push hf-test huggingface:main --force
```

Verify at https://huggingface.co/spaces/Realcat/imcui

### Step 4: Deploy to production Space

After confirming test Space works:

```bash
git push hf-prod huggingface:main --force
```

## HuggingFace Space Configuration

The `huggingface` branch is a lightweight orphan branch containing only deployment files:

| File | Purpose |
|------|---------|
| `app.py` | Launcher importing from `imcui` package |
| `requirements.txt` | `imcui==X.Y.Z` (pulls all deps from PyPI) |
| `packages.txt` | System deps: `build-essential`, `ffmpeg`, `libsm6`, `libxext6` |
| `config/app.yaml` | Full matcher zoo config |
| `README.md` | HF metadata: `sdk: gradio`, `python_version: "3.12"`, `pinned: true` |

ZeroGPU requires Gradio SDK (not Docker). Python pinned to 3.12 because imcui does not support 3.13 yet.

## Sync config from main

When main's `imcui/config/app.yaml` changes (new matchers added), sync to huggingface branch:

```bash
git checkout huggingface
cp imcui/config/app.yaml config/app.yaml
git add config/app.yaml
git commit -m "chore: sync config from main"
git push origin huggingface
```
