# DKM: Dense Kernelized Feature Matching for Geometry Estimation
### [Project Page](https://parskatt.github.io/DKM) | [Paper](https://arxiv.org/abs/2202.00667)
<br/>

> DKM: Dense Kernelized Feature Matching for Geometry Estimation  
> [Johan Edstedt](https://scholar.google.com/citations?user=Ul-vMR0AAAAJ), [Ioannis Athanasiadis](https://scholar.google.com/citations?user=RCAtJgUAAAAJ), [Mårten Wadenbäck](https://scholar.google.com/citations?user=6WRQpCQAAAAJ), [Michael Felsberg](https://scholar.google.com/citations?&user=lkWfR08AAAAJ)  
> CVPR 2023

## How to Use?
<details>
Our model produces a dense (for all pixels) warp and certainty.

Warp: [B,H,W,4] for all images in batch of size B, for each pixel HxW, we ouput the input and matching coordinate in the normalized grids [-1,1]x[-1,1].

Certainty: [B,H,W] a number in each pixel indicating the matchability of the pixel.

See [demo](dkm/demo/) for two demos of DKM.

See [api.md](docs/api.md) for API.
</details>

## Qualitative Results
<details>

https://user-images.githubusercontent.com/22053118/223748279-0f0c21b4-376a-440a-81f5-7f9a5d87483f.mp4


https://user-images.githubusercontent.com/22053118/223748512-1bca4a17-cffa-491d-a448-96aac1353ce9.mp4



https://user-images.githubusercontent.com/22053118/223748518-4d475d9f-a933-4581-97ed-6e9413c4caca.mp4



https://user-images.githubusercontent.com/22053118/223748522-39c20631-aa16-4954-9c27-95763b38f2ce.mp4


</details>



## Benchmark Results

<details>

### Megadepth1500

|       | @5    | @10  | @20  |
|-------|-------|------|------|
| DKMv1 | 54.5  | 70.7 | 82.3 |
| DKMv2 | *56.8*  | *72.3* | *83.2* |
| DKMv3 (paper) | **60.5**  | **74.9** | **85.1** |
| DKMv3 (this repo) | **60.0**  | **74.6** | **84.9** |

### Megadepth 8 Scenes
|       | @5    | @10  | @20  |
|-------|-------|------|------|
| DKMv3 (paper) | **60.5**  | **74.5** | **84.2** |
| DKMv3 (this repo) | **60.4**  | **74.6** | **84.3** |


### ScanNet1500
|       | @5    | @10  | @20  |
|-------|-------|------|------|
| DKMv1 | 24.8  | 44.4 | 61.9 |
| DKMv2 | *28.2*  | *49.2* | *66.6* |
| DKMv3 (paper) | **29.4**  | **50.7** | **68.3** |
| DKMv3 (this repo) | **29.8**  | **50.8** | **68.3** |

</details>

## Navigating the Code
* Code for models can be found in [dkm/models](dkm/models/)
* Code for benchmarks can be found in [dkm/benchmarks](dkm/benchmarks/)
* Code for reproducing experiments from our paper can be found in [experiments/](experiments/)

## Install
Run ``pip install -e .``

## Demo

A demonstration of our method can be run by:
``` bash
python demo_match.py
```
This runs our model trained on mega on two images taken from Sacre Coeur.

## Benchmarks
See [Benchmarks](docs/benchmarks.md) for details.
## Training
See [Training](docs/training.md) for details.
## Reproducing Results
Given that the required benchmark or training dataset has been downloaded and unpacked, results can be reproduced by running the experiments in the experiments folder.

## Using DKM matches for estimation
We recommend using the excellent Graph-Cut RANSAC algorithm: https://github.com/danini/graph-cut-ransac

|       | @5    | @10  | @20  |
|-------|-------|------|------|
| DKMv3 (RANSAC) | *60.5*  | *74.9* | *85.1* |
| DKMv3 (GC-RANSAC) | **65.5**  | **78.0** | **86.7** |


## Acknowledgements
We have used code and been inspired by https://github.com/PruneTruong/DenseMatching, https://github.com/zju3dv/LoFTR, and https://github.com/GrumpyZhou/patch2pix. We additionally thank the authors of ECO-TR for providing their benchmark.

## BibTeX
If you find our models useful, please consider citing our paper!
```
@inproceedings{edstedt2023dkm,
title={{DKM}: Dense Kernelized Feature Matching for Geometry Estimation},
author={Edstedt, Johan and Athanasiadis, Ioannis and Wadenbäck, Mårten and Felsberg, Michael},
booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
year={2023}
}
```
