<p align="center">
  <h1 align="center"><br><ins>Image Matching WebGUI</ins><br>matching images with given images</h1> 
</p>

## Description

This simple tool efficiently matches image pairs using multiple famous image matching algorithms. The tool features a Graphical User Interface (GUI) designed using [gradio](https://gradio.app/). You can effortlessly select two images and a matching algorithm and obtain a precise matching result.

![](assets/gui.png)

The tool currently supports various popular image matching algorithms, namely:

- [ ] [DKM](https://github.com/Parskatt/DKM), CVPR 2023
- [x] [TopicFM](https://github.com/TruongKhang/TopicFM), AAAI 2023
- [x] [AspanFormer](https://github.com/apple/ml-aspanformer), ECCV 2022
- [x] [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork), CVPRW 2018
- [x] [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork), CVPR 2020
- [x] [D2Net](https://github.com/mihaidusmanu/d2-net), CVPR 2019
- [x] [R2D2](https://github.com/naver/r2d2), NeurIPS 2019
- [x] [DISK](https://github.com/cvlab-epfl/disk), NeurIPS 2020
- [x] [SIFT](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html), IJCV 2004

## How to use

### requirements
``` bash
pip install -r requirements.txt
```
### run demo
``` bash
python3 ./main.py
```
## Acknowledgement

This code is built based on [Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization). We express our gratitude to the authors for their valuable source code.