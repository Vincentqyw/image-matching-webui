[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

<p align="center">
  <h1 align="center"><br><ins>Image Matching WebUI</ins><br>find matches between 2 images</h1> 
</p>

## Description

This simple tool efficiently matches image pairs using multiple famous image matching algorithms. The tool features a Graphical User Interface (GUI) designed using [gradio](https://gradio.app/). You can effortlessly select two images and a matching algorithm and obtain a precise matching result.
**Note**: the images source can be either local images or webcam images.

Here is a demo of the tool:

![demo](assets/demo.gif)

The tool currently supports various popular image matching algorithms, namely:
- [x] [LightGlue](https://github.com/cvg/LightGlue), ICCV 2023
- [x] [DeDoDe](https://github.com/Parskatt/DeDoDe), TBD
- [x] [DarkFeat](https://github.com/THU-LYJ-Lab/DarkFeat), AAAI 2023
- [ ] [ASTR](https://github.com/ASTR2023/ASTR), CVPR 2023
- [ ] [SEM](https://github.com/SEM2023/SEM), CVPR 2023
- [ ] [DeepLSD](https://github.com/cvg/DeepLSD), CVPR 2023
- [x] [GlueStick](https://github.com/cvg/GlueStick), ArXiv 2023
- [ ] [ConvMatch](https://github.com/SuhZhang/ConvMatch), AAAI 2023
- [x] [SOLD2](https://github.com/cvg/SOLD2), CVPR 2021
- [ ] [LineTR](https://github.com/yosungho/LineTR), RA-L 2021
- [x] [DKM](https://github.com/Parskatt/DKM), CVPR 2023
- [x] [RoMa](https://github.com/Vincentqyw/RoMa), Arxiv 2023
- [ ] [NCMNet](https://github.com/xinliu29/NCMNet), CVPR 2023
- [x] [TopicFM](https://github.com/Vincentqyw/TopicFM), AAAI 2023
- [x] [AspanFormer](https://github.com/Vincentqyw/ml-aspanformer), ECCV 2022
- [x] [LANet](https://github.com/wangch-g/lanet), ACCV 2022
- [ ] [LISRD](https://github.com/rpautrat/LISRD), ECCV 2022
- [ ] [REKD](https://github.com/bluedream1121/REKD), CVPR 2022
- [x] [ALIKE](https://github.com/Shiaoming/ALIKE), ArXiv 2022
- [x] [SGMNet](https://github.com/vdvchen/SGMNet), ICCV 2021
- [x] [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork), CVPRW 2018
- [x] [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork), CVPR 2020
- [x] [D2Net](https://github.com/Vincentqyw/d2-net), CVPR 2019
- [x] [R2D2](https://github.com/naver/r2d2), NeurIPS 2019
- [x] [DISK](https://github.com/cvlab-epfl/disk), NeurIPS 2020
- [ ] [Key.Net](https://github.com/axelBarroso/Key.Net), ICCV 2019
- [ ] [OANet](https://github.com/zjhthu/OANet), ICCV 2019
- [ ] [SOSNet](https://github.com/scape-research/SOSNet), CVPR 2019
- [x] [SIFT](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html), IJCV 2004

## How to use

### HuggingFace

Just try it on HuggingFace: [Realcat/image-matching-webui](https://huggingface.co/spaces/Realcat/image-matching-webui).

or deploy it locally following the instructions below.

### Requirements
``` bash
git clone --recursive https://github.com/Vincentqyw/image-matching-webui.git
cd image-matching-webui
conda env create -f environment.yaml
conda activate imw
```
 
### Run demo
``` bash
python3 ./app.py
```
then open http://localhost:7860 in your browser.

![](assets/gui.jpg)

### Add your own feature / matcher

I provide an example to add local feature in [hloc/extractors/example.py](hloc/extractors/example.py). Then add feature settings in `confs` in file [hloc/extract_features.py](hloc/extract_features.py). Last step is adding some settings to `model_zoo` in file [extra_utils/utils.py](extra_utils/utils.py).

## Contributions welcome!

External contributions are very much welcome. Please follow the [PEP8 style guidelines](https://www.python.org/dev/peps/pep-0008/) using a linter like flake8 (reformat using command `python -m black .`). This is a non-exhaustive list of features that might be valuable additions:

- [x] add webcam support
- [x] add [line feature matching](https://github.com/Vincentqyw/LineSegmentsDetection) algorithms
- [x] example to add a new feature extractor / matcher
- [ ] ransac to filter outliers
- [ ] support export matches to colmap ([#issue 6](https://github.com/Vincentqyw/image-matching-webui/issues/6))
- [ ] add config file to set default parameters
- [ ] dynamically load models and reduce GPU overload

Adding local features / matchers as submodules is very easy. For example, to add the [GlueStick](https://github.com/cvg/GlueStick): 

``` bash
git submodule add https://github.com/cvg/GlueStick.git third_party/GlueStick
```

If remote submodule repositories are updated, don't forget to pull submodules with `git submodule update --remote`, if you only want to update one submodule, use `git submodule update --remote third_party/GlueStick`.

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
