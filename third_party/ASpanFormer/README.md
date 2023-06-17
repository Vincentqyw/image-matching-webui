# ASpanFormer Implementation

![Framework](assets/teaser.png)

This is a PyTorch implementation of ASpanFormer for ECCV'22 [paper](https://arxiv.org/abs/2208.14201), “ASpanFormer: Detector-Free Image Matching with Adaptive Span Transformer”, and can be used to reproduce the results in the paper.

This work focuses on detector-free image matching. We propose a hierarchical attention framework for cross-view feature update, which adaptively adjusts attention span based on region-wise matchability.

This repo contains training, evaluation and basic demo scripts used in our paper.

A large part of the code base is borrowed from the [LoFTR Repository](https://github.com/zju3dv/LoFTR) under its own separate license, terms and conditions.  The authors of this software are not responsible for the contents of third-party websites.

## Installation 
```bash
conda env create -f environment.yaml
conda activate ASpanFormer
```

## Get started
Download model weights from [here](https://drive.google.com/file/d/1eavM9dTkw9nbc-JqlVVfGPU5UvTTfc6k/view?usp=share_link)  

Extract weights by
```bash
tar -xvf weights_aspanformer.tar
```

A demo to match one image pair is provided. To get a quick start, 

```bash
cd demo
python demo.py
```


## Data Preparation
Please follow the [training doc](docs/TRAINING.md) for data organization



## Evaluation


### 1. ScanNet Evaluation 
```bash
cd scripts/reproduce_test
bash indoor.sh
```
Similar results as below should be obtained,
```bash
'auc@10': 0.46640095171012563,
'auc@20': 0.6407042320049785,
'auc@5': 0.26241231577189295,
'prec@5e-04': 0.8827665604024288,
'prec_flow@2e-03': 0.810938751342228
```

### 2. MegaDepth Evaluation
 ```bash
cd scripts/reproduce_test
bash outdoor.sh
```
Similar results as below should be obtained,
```bash
'auc@10': 0.7184113573584142,
'auc@20': 0.8333835724453831,
'auc@5': 0.5567622479156181,
'prec@5e-04': 0.9901741341790503,
'prec_flow@2e-03': 0.7188964321862907
```


## Training

### 1. ScanNet Training
```bash
cd scripts/reproduce_train
bash indoor.sh
```

### 2. MegaDepth Training
```bash
cd scripts/reproduce_train
bash outdoor.sh
```
      

If you find this project useful, please cite:

```
@article{chen2022aspanformer,
  title={ASpanFormer: Detector-Free Image Matching with Adaptive Span Transformer},
  author={Chen, Hongkai and Luo, Zixin and Zhou, Lei and Tian, Yurun and Zhen, Mingmin and Fang, Tian and McKinnon, David and Tsin, Yanghai and Quan, Long},
  journal={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
