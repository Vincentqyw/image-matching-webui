# GlueStick
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cvg/GlueStick/blob/main/gluestick_matching_demo.ipynb) [![arXiv](https://img.shields.io/badge/arXiv-2304.02008-b31b1b.svg?style=flat)](https://arxiv.org/abs/2304.02008) [![Project Page](https://badgen.net/badge/color/project/green?icon=awesome&label)](https://iago-suarez.com/gluestick)

Joint deep matcher for points and lines 🖼️💥🖼️

![Visualization of point and line matches](resources/demo_seq1.gif)

This repository contains the official implementation of 
[GlueStick: Robust Image Matching by Sticking Points and Lines Together](https://arxiv.org/abs/2304.02008).

## Install 🛠️

To install the software in Ubuntu 22.04 follow these instructions:
```bash
sudo apt-get install build-essential cmake libopencv-dev libopencv-contrib-dev
git clone --recursive https://github.com/cvg/GlueStick.git
cd GlueStick
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Running GlueStick 🏃
Download the weights of the model:
```
wget https://github.com/cvg/GlueStick/releases/download/v0.1_arxiv/checkpoint_GlueStick_MD.tar -P resources/weights
```

You can execute the inference with it with:
```
python -m gluestick.run -img1 resources/img1.jpg -img2 resources/img2.jpg
```

## Training 🏋️
We want to provide you with high-quality and flexible code for training. Stay tuned, we will release it soon!

## Citation 📝
If you use this code in your project, please consider citing the following paper:
```bibtex
@article{pautrat_suarez_2023_gluestick,
    title={{GlueStick}: Robust Image Matching by Sticking Points and Lines Together},
    author={Pautrat, R{\'e}mi* and Su{\'a}rez, Iago* and Yu, Yifan and Pollefeys, Marc and Larsson, Viktor},
    journal={ArXiv},
    year={2023}
}
```
