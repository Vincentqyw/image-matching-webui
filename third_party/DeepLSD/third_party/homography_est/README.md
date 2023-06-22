# homography_est
[![Build](https://github.com/rpautrat/homography_est/actions/workflows/cmake.yml/badge.svg)](https://github.com/rpautrat/homography_est/actions/workflows/cmake.yml)

Light-weight Python bindings to perform homography estimation between two images with RANSAC from point, line or point-line correspondences.
Based on [RansacLib](https://github.com/tsattler/RansacLib) and developed by [Iago Suarez](https://github.com/iago-suarez), [Viktor Larsson](https://github.com/vlarsson), and [Rémi Pautrat](https://github.com/rpautrat).

## Installation
This work relies on the following dependencies:
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
- [Ceres Solver](http://ceres-solver.org/)

After installing these two dependencies, you can clone and install the homography estimation code:
```
git clone --recurse-submodules git@github.com:rpautrat/homography_est.git
cd homography_est
pip install -e .
```

## Usage
Check the examples in `run_hest_test.cc` and `run_hest_test.py` to understand how to use the library. We provide the following Python functions:
- `ransac_point_homography`: homography estimation from a minimal set of 4 points.
- `ransac_line_homography`: homography estimation from a minimal set of 4 lines.
- `ransac_point_line_homography`: homography estimation with hybrid RANSAC from minimal sets of 4 points or 4 lines.

We also give the option to estimate the homography in case of a pure rotation between the images. In this case, only 2 features are minimal.
