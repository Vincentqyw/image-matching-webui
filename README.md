<p align="center">
  <h1 align="center"><br><ins>Image Matching WebGUI</ins><br>matching images with given images</h1> 
</p>

## Description

A simple python script to match image pairs. It has a GUI made using [gradio](https://gradio.app/).

Now this tool support many famous image matching algorithms, including:
- [x] [TopicFM](https://github.com/TruongKhang/TopicFM)
- [x] [AspanFormer](https://github.com/apple/ml-aspanformer)
- [x] [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)
- [x] [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
- [x] [D2Net](https://github.com/mihaidusmanu/d2-net)
- [x] [R2D2](https://github.com/naver/r2d2)
- [x] [DISK](https://github.com/cvlab-epfl/disk)
- [x] [SIFT](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html)
- [ ] [DKM](https://github.com/Parskatt/DKM), CVPR 2023

## How to use

### requirements
``` bash
pip install -r requirements.txt
```
### run demo
``` bash
python3 src/main.py
```
![](assets/gui.png)