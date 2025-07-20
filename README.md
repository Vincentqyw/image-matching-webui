<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url] -->

<p align="center">
  <h1 align="center"><br><ins>Image Matching WebUI</ins>
  <br>Matching Keypoints between two images</h1>
</p>
<div align="center">
  <a target="_blank" href="https://github.com/Vincentqyw/image-matching-webui/actions/workflows/release.yml"><img src="https://github.com/Vincentqyw/image-matching-webui/actions/workflows/release.yml/badge.svg" alt="PyPI Release"></a>
  <a target="_blank" href='https://huggingface.co/spaces/Realcat/image-matching-webui'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
  <a target="_blank" href="https://pypi.org/project/imcui"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/imcui?style=flat&logo=pypi&label=imcui&link=https%3A%2F%2Fpypi.org%2Fproject%2Fimcui"></a>
  <a target="_blank" href="https://hub.docker.com/r/vincentqin/image-matching-webui"><img alt="Docker Image Version" src="https://img.shields.io/docker/v/vincentqin/image-matching-webui?sort=date&arch=amd64&logo=docker&label=imcui&link=https%3A%2F%2Fhub.docker.com%2Fr%2Fvincentqin%2Fimage-matching-webui"></a>
  <a target="_blank" href="https://pepy.tech/projects/imcui"><img src="https://static.pepy.tech/badge/imcui" alt="PyPI Downloads"></a>
  <a target="_blank" href="https://deepwiki.com/Vincentqyw/image-matching-webui"><img src="https://img.shields.io/badge/DeepWiki-imcui-blue.svg" alt="DeepWiki"></a>
</div>

## Description

`Image Matching WebUI (IMCUI)` efficiently matches image pairs using multiple famous image matching algorithms. The tool features a Graphical User Interface (GUI) designed using [gradio](https://gradio.app/). You can effortlessly select two images and a matching algorithm and obtain a precise matching result.
**Note**: the images source can be either local images or webcam images.

Try it on
<a href='https://huggingface.co/spaces/Realcat/image-matching-webui'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
<a target="_blank" href="https://lightning.ai/realcat/studios/image-matching-webui"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a>

Here is a demo of the tool:

https://github.com/Vincentqyw/image-matching-webui/assets/18531182/263534692-c3484d1b-cc00-4fdc-9b31-e5b7af07ecd9

The tool currently supports various popular image matching algorithms, namely:

| Algorithm        | Supported | Conference/Journal | Year | GitHub Link |
|------------------|-----------|--------------------|------|-------------|
| RIPE           | ✅ | ICCV    | 2025 | [Link](https://github.com/fraunhoferhhi/RIPE)  |
| RDD            | ✅ | CVPR    | 2025 | [Link](https://github.com/xtcpete/rdd)  |
| LiftFeat       | ✅ | ICRA    | 2025 | [Link](https://github.com/lyp-deeplearning/LiftFeat) |
| DaD            | ✅ | ARXIV   | 2025 | [Link](https://github.com/Parskatt/dad) |
| MINIMA         | ✅ | ARXIV   | 2024 | [Link](https://github.com/LSXI7/MINIMA) |
| XoFTR          | ✅ | CVPR    | 2024 | [Link](https://github.com/OnderT/XoFTR) |
| EfficientLoFTR | ✅ | CVPR    | 2024 | [Link](https://github.com/zju3dv/EfficientLoFTR) |
| MASt3R         | ✅ | CVPR    | 2024 | [Link](https://github.com/naver/mast3r) |
| DUSt3R         | ✅ | CVPR    | 2024 | [Link](https://github.com/naver/dust3r) |
| OmniGlue       | ✅ | CVPR    | 2024 | [Link](https://github.com/Vincentqyw/omniglue-onnx) |
| XFeat          | ✅ | CVPR    | 2024 | [Link](https://github.com/verlab/accelerated_features) |
| RoMa           | ✅ | CVPR    | 2024 | [Link](https://github.com/Vincentqyw/RoMa) |
| DeDoDe         | ✅ | 3DV     | 2024 | [Link](https://github.com/Parskatt/DeDoDe) |
| Mickey         | ❌ | CVPR    | 2024 | [Link](https://github.com/nianticlabs/mickey) |
| GIM            | ✅ | ICLR    | 2024 | [Link](https://github.com/xuelunshen/gim) |
| ALIKED         | ✅ | ICCV    | 2023 | [Link](https://github.com/Shiaoming/ALIKED) |
| LightGlue      | ✅ | ICCV    | 2023 | [Link](https://github.com/cvg/LightGlue) |
| DarkFeat       | ✅ | AAAI    | 2023 | [Link](https://github.com/THU-LYJ-Lab/DarkFeat) |
| SFD2           | ✅ | CVPR    | 2023 | [Link](https://github.com/feixue94/sfd2) |
| IMP            | ✅ | CVPR    | 2023 | [Link](https://github.com/feixue94/imp-release) |
| ASTR           | ❌ | CVPR    | 2023 | [Link](https://github.com/ASTR2023/ASTR) |
| SEM            | ❌ | CVPR    | 2023 | [Link](https://github.com/SEM2023/SEM) |
| DeepLSD        | ❌ | CVPR    | 2023 | [Link](https://github.com/cvg/DeepLSD) |
| GlueStick      | ✅ | ICCV    | 2023 | [Link](https://github.com/cvg/GlueStick) |
| ConvMatch      | ❌ | AAAI    | 2023 | [Link](https://github.com/SuhZhang/ConvMatch) |
| LoFTR          | ✅ | CVPR    | 2021 | [Link](https://github.com/zju3dv/LoFTR) |
| SOLD2          | ✅ | CVPR    | 2021 | [Link](https://github.com/cvg/SOLD2) |
| LineTR         | ❌ | RA-L    | 2021 | [Link](https://github.com/yosungho/LineTR) |
| DKM            | ✅ | CVPR    | 2023 | [Link](https://github.com/Parskatt/DKM) |
| NCMNet         | ❌ | CVPR    | 2023 | [Link](https://github.com/xinliu29/NCMNet) |
| TopicFM        | ✅ | AAAI    | 2023 | [Link](https://github.com/Vincentqyw/TopicFM) |
| AspanFormer    | ✅ | ECCV    | 2022 | [Link](https://github.com/Vincentqyw/ml-aspanformer) |
| LANet          | ✅ | ACCV    | 2022 | [Link](https://github.com/wangch-g/lanet) |
| LISRD          | ❌ | ECCV    | 2022 | [Link](https://github.com/rpautrat/LISRD) |
| REKD           | ❌ | CVPR    | 2022 | [Link](https://github.com/bluedream1121/REKD) |
| CoTR           | ✅ | ICCV    | 2021 | [Link](https://github.com/ubc-vision/COTR) |
| ALIKE          | ✅ | TMM     | 2022 | [Link](https://github.com/Shiaoming/ALIKE) |
| RoRD           | ✅ | IROS    | 2021 | [Link](https://github.com/UditSinghParihar/RoRD) |
| SGMNet         | ✅ | ICCV    | 2021 | [Link](https://github.com/vdvchen/SGMNet) |
| SuperPoint     | ✅ | CVPRW   | 2018 | [Link](https://github.com/magicleap/SuperPointPretrainedNetwork) |
| SuperGlue      | ✅ | CVPR    | 2020 | [Link](https://github.com/magicleap/SuperGluePretrainedNetwork) |
| D2Net          | ✅ | CVPR    | 2019 | [Link](https://github.com/Vincentqyw/d2-net) |
| R2D2           | ✅ | NeurIPS | 2019 | [Link](https://github.com/naver/r2d2) |
| DISK           | ✅ | NeurIPS | 2020 | [Link](https://github.com/cvlab-epfl/disk) |
| Key.Net        | ❌ | ICCV    | 2019 | [Link](https://github.com/axelBarroso/Key.Net) |
| OANet          | ❌ | ICCV    | 2019 | [Link](https://github.com/zjhthu/OANet) |
| SOSNet         | ✅ | CVPR    | 2019 | [Link](https://github.com/scape-research/SOSNet) |
| HardNet        | ✅ | NeurIPS | 2017 | [Link](https://github.com/DagnyT/hardnet) |
| SIFT           | ✅ | IJCV    | 2004 | [Link](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html) |


## How to use

### HuggingFace / Lightning AI

Just try it on <a href='https://huggingface.co/spaces/Realcat/image-matching-webui'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
<a target="_blank" href="https://lightning.ai/realcat/studios/image-matching-webui">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a>

or deploy it locally following the instructions below.

### Requirements

- [Python 3.9+](https://www.python.org/downloads/)

#### Install from pip [NEW]

Update: now support install from [pip](https://pypi.org/project/imcui), just run:

```bash
pip install imcui
```

#### Install from source

``` bash
git clone --recursive https://github.com/Vincentqyw/image-matching-webui.git
cd image-matching-webui
conda env create -f environment.yaml
conda activate imcui
pip install -e .
```

or using [docker](https://hub.docker.com/r/vincentqin/image-matching-webui):

```bash
docker pull vincentqin/image-matching-webui:latest

# Start the WebUI service
docker-compose up webui

# Or run in the background
docker-compose up -d webui
```

<details>
<summary><strong>More Docker Compose Commands</strong> (click to expand)</summary>

```bash
# Build and start the WebUI service
docker-compose up --build webui

# Check the status of the WebUI service
docker-compose ps webui

# View logs for the WebUI service
docker-compose logs webui
docker-compose logs -f webui  # Follow logs in real time

# Stop the WebUI service
docker-compose stop webui

# Restart the WebUI service
docker-compose restart webui

# Remove the WebUI service container
docker-compose rm webui

# Remove all containers
docker-compose down

```
</details>

### Deploy to Railway

Deploy to [Railway](https://railway.app/), setting up a `Custom Start Command` in `Deploy` section:

``` bash
python -m imcui.api.server
```

### Run demo
``` bash
python app.py --config ./config/config.yaml
```
then open http://localhost:7860 in your browser.

![](assets/gui.jpg)

### Add your own feature / matcher

I provide an example to add local feature in [imcui/hloc/extractors/example.py](imcui/hloc/extractors/example.py). Then add feature settings in `confs` in file [imcui/hloc/extract_features.py](imcui/hloc/extract_features.py). Last step is adding some settings to `matcher_zoo` in file [imcui/ui/config.yaml](imcui/ui/config.yaml).

### Upload models

IMCUI hosts all models on [Huggingface](https://huggingface.co/Realcat/imcui_checkpoints).  You can upload your model to Huggingface and add it to the [Realcat/imcui_checkpoints](https://huggingface.co/Realcat/imcui_checkpoints) repository.


## Contributions welcome!

External contributions are very much welcome. Please follow the [PEP8 style guidelines](https://www.python.org/dev/peps/pep-0008/) using a linter like flake8. This is a non-exhaustive list of features that might be valuable additions:

- [x] support pip install command
- [x] add [CPU CI](.github/workflows/ci.yml)
- [x] add webcam support
- [x] add [line feature matching](https://github.com/Vincentqyw/LineSegmentsDetection) algorithms
- [x] example to add a new feature extractor / matcher
- [x] ransac to filter outliers
- [ ] add [rotation images](https://github.com/pidahbus/deep-image-orientation-angle-detection) options before matching
- [ ] support export matches to colmap ([#issue 6](https://github.com/Vincentqyw/image-matching-webui/issues/6))
- [x] add config file to set default parameters
- [x] dynamically load models and reduce GPU overload

Adding local features / matchers as submodules is very easy. For example, to add the [GlueStick](https://github.com/cvg/GlueStick):

``` bash
git submodule add https://github.com/cvg/GlueStick.git imcui/third_party/GlueStick
```

If remote submodule repositories are updated, don't forget to pull submodules with:

``` bash
git submodule update --init --recursive  # init and download
git submodule update --remote  # update
```

if you only want to update one submodule, use `git submodule update --remote imcui/third_party/GlueStick`.

To format code before committing, run:

```bash
pre-commit run -a  # Auto-checks and fixes
```

## Contributors

<a href="https://github.com/Vincentqyw/image-matching-webui/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Vincentqyw/image-matching-webui" />
</a>

## Resources
- [Image Matching: Local Features & Beyond](https://image-matching-workshop.github.io)
- [Long-term Visual Localization](https://www.visuallocalization.net)

## Acknowledgement

This code is built based on [Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization). We express our gratitude to the authors for their valuable source code.

[contributors-shield]: https://img.shields.io/github/contributors/Vincentqyw/image-matching-webui.svg?style=for-the-badge
[contributors-url]: https://github.com/Vincentqyw/image-matching-webui/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Vincentqyw/image-matching-webui.svg?style=for-the-badge
[forks-url]: https://github.com/Vincentqyw/image-matching-webui/network/members
[stars-shield]: https://img.shields.io/github/stars/Vincentqyw/image-matching-webui.svg?style=for-the-badge
[stars-url]: https://github.com/Vincentqyw/image-matching-webui/stargazers
[issues-shield]: https://img.shields.io/github/issues/Vincentqyw/image-matching-webui.svg?style=for-the-badge
[issues-url]: https://github.com/Vincentqyw/image-matching-webui/issues
