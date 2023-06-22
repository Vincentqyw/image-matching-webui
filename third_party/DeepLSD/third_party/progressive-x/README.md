# Progressive-X

The Progressive-X algorithm proposed in paper: Daniel Barath and Jiri Matas; Progressive-X: Efficient, Anytime, Multi-Model Fitting Algorithm, International Conference on Computer Vision, 2019. 
It is available at https://arxiv.org/pdf/1906.02290

# Installation C++

To build and install C++ only `Progressive-X`, clone or download this repository and then build the project by CMAKE. 
```shell
$ git clone --recursive https://github.com/danini/progressive-x.git
$ cd build
$ cmake ..
$ make
```

# Install Python package and compile C++

```bash
python3 ./setup.py install
```

or

```bash
pip3 install -e .
```

# Example project

To build the sample project showing examples of fundamental matrix, homography and essential matrix fitting, set variable `CREATE_SAMPLE_PROJECT = ON` when creating the project in CMAKE. 
Then 
```shell
$ cd build
$ ./SampleProject
```

# Jupyter Notebook code for re-producing the results in the paper

The code for multiple homography fitting is available at: [notebook](dataset_comparison/adelaideH.ipynb).

The code for multiple two-view motion fitting is available at: [notebook](dataset_comparison/adelaideF.ipynb).

# Jupyter Notebook example

The example for multiple homography fitting is available at: [notebook](examples/example_multi_homography.ipynb).

The example for multiple two-view motion fitting is available at: [notebook](examples/example_multi_two_view_motion.ipynb).
 
The example for multiple 6D pose fitting is available at: [notebook](examples/example_multi_pose_6d.ipynb).
 
The example for multiple vanishing point detection is available at: [notebook](examples/example_multi_vanishing_point.ipynb).

# Requirements

- Eigen 3.0 or higher
- CMake 2.8.12 or higher
- OpenCV 3.0 or higher
- GFlags
- GLog
- A modern compiler with C++17 support


# Acknowledgements

When using the algorithm, please cite `Barath, Daniel, and Matas, Jiří. "Progressive-X: Efficient, Anytime, Multi-Model Fitting Algorithm". Proceedings of the IEEE International Conference on Computer Vision. 2019`.

If you use Progressive-X with Graph-Cut RANSAC as a proposal engine, please cite `Barath, Daniel, and Matas, Jiří. "Graph-cut RANSAC." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018`.

If you use Progressive-X with Progressive NAPSAC sampler, please cite `Barath, Daniel and Noskova, Jana and Ivashechkin, Maksym and Matas, Jiří. "MAGSAC++, a fast, reliable and accurate robust estimator" Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2020`.
