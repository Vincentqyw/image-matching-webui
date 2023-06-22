# DeepLSD
Implementation of the paper [DeepLSD: Line Segment Detection and Refinement with Deep Image Gradients](https://arxiv.org/abs/2212.07766), accepted at CVPR 2023. **DeepLSD is a generic line detector that combines the robustness of deep learning with the accuracy of handcrafted detectors**. It can be used to extract **generic line segments from images in-the-wild**, and is **suitable for any task requiring high precision**, such as homography estimation, visual localization, and 3D reconstruction. By predicting a line distance and angle fields, **it can furthermore refine any existing line segments** through an optimization.

Demo of the lines detected by DeepLSD, its line distance field, and line angle field:

<p align="center">
<img width=40% src="assets/videos/demo_deeplsd.gif" style="margin:-300px 0px -300px 0px">
</p>

## Installation
First clone the repository and its submodules:
```
git clone --recurse-submodules https://github.com/cvg/DeepLSD.git
cd DeepLSD
```

### Quickstart install (for inference only)

To test the pre-trained model on your images, without the final line refinement, the following installation is sufficient:
```
bash quickstart_install.sh
```
You can then test it with the notebook `notebooks/quickstart_demo.ipynb`.

### Full install

Follow these instructions if you wish to re-train DeepLSD, evaluate it, or use the final step of line refinement.

Dependencies that need to be installed on your system:
- [OpenCV](https://opencv.org/)
- [GFlags](https://github.com/gflags/gflags)
- [GLog](https://github.com/google/glog)
- [Ceres 2.0.0](http://ceres-solver.org/)
- DeepLSD was successfully tested with GCC 9, Python 3.7, and CUDA 11. Other combinations may work as well.

Once these libraries are installed, you can proceed with the installation of the necessary requirements and third party libraries:
```
bash install.sh
```

This repo uses a base experiment folder (EXPER_PATH) containing the output of all trainings, and a base dataset path (DATA_PATH) containing all the evaluation and training datasets. You can set the path to these two folders in the file `deeplsd/settings.py`.

## Usage
We provide two pre-trained models for DeepLSD: [deeplsd_wireframe.tar](https://www.polybox.ethz.ch/index.php/s/FQWGkH57UNTqlJZ) and [deeplsd_md.tar](https://www.polybox.ethz.ch/index.php/s/XVb30sUyuJttFys), trained respectively on the Wireframe and MegaDepth datasets. The former can be used for easy indoor datasets, while the latter is more generic and works outdoors and on more challenging scenes.
The example notebook `notebooks/demo_line_detection.ipynb` showcases how to use DeepLSD in practice. Please refer to the comments on the config of this notebook to understand the usage of each hyperparameter.

## Ground truth (GT) generation
DeepLSD requires generating a ground truth for line distance and angle field, through homography adaptation. We provide a Python script to do it for any list of images, leveraging CUDA:
```
python -m deeplsd.scripts.homography_adaptation_df <path to a txt file containing the image paths> <output folder> --num_H <number of homographies (e.g. 100)> --n_jobs <number of parallel jobs>
```
Note that the GT generation can take a long time, from hours to days depending on the number of images, number of homographies and computation power of your machine.
The output folder can then be specified in the training config of the corresponding dataset. For example, after generating the GT for the wireframe dataset in the folder `DATA_PATH/export_datasets/wireframe_ha5`, the field 'gt_dir' of the config file `deeplsd/configs/train_wireframe.yaml` can be updated with the value `export_datasets/wireframe_ha5` (paths are given relative to DATA_PATH).

## Training
To train the network, simply run the following command:
```
python -m deeplsd.scripts.train <model name> --conf <path to config file>
```

We provide data loaders for the [Wireframe dataset](https://github.com/huangkuns/wireframe) and [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/), using the config files `deeplsd/configs/train_wireframe.yaml` and  `deeplsd/configs/train_minidepth.yaml`, or `deeplsd/configs/train_merged_datasets.yaml` to train on both at the same time. Note that due to the sheer size of the MegaDepth dataset, we only sampled 50 images per scene (hence the name "Minidepth") and used the train/val split available in `deeplsd/datasets/utils/`. To train on the wireframe dataset, the command would typically look like:
```
python -m deeplsd.scripts.train deeplsd_wireframe --conf deeplsd/configs/train_wireframe.yaml
```

A model can be restored or fine-tuned by adding the '--restore' option.

## Line refinement
The backbone extractor of DeepLSD can also be used to generate a distance and angle field from an image, and to refine existing line segments (from any existsing line detector). This can be done given a folder of images and pre-extracted lines as follows:
```
python -m deeplsd.scripts.line_refinement <path to the image folder> <path to the line detections> <path to the checkpoint of DeepLSD>
```
Please refer to the help tool of this function for more details on the format of the line detections.

## Evaluation
Similarly as in the paper, we provide code for the evaluation of low-level line detection metrics, as well as vanishing point (VP) estimation. In both cases, the lines and VPs need to be extracted with the script:
```
python -m deeplsd.scripts.export_features <config file> <path to the DeepLSD checkpoint> <output_folder>
```
Add the option '--pred_vps' to predict the vanishing points in addition to the line segments.

### Low-level line detection metrics
We provide dataloaders for the following datasets:
- [Wireframe](https://github.com/huangkuns/wireframe)
- [HPatches](https://github.com/hpatches/hpatches-dataset) (with the full image sequences)
- [RDNIM](https://www.polybox.ethz.ch/index.php/s/P89YkZyOfdhmdPN)
- [York Urban DB](https://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/)

The corresponding config files (to export the features) are located in the `deeplsd/configs` folder. For example, exporting line detections on HPatches would look like:
```
python -m deeplsd.scripts.export_features deeplsd/configs/export_hpatches.yaml weights/deeplsd_md.tar hpatches_outputs
```

The evaluation can then be run with:
```
python -m deeplsd.scripts.evaluate_line_detection <dataset name ('wireframe', 'hpatches', 'rdnim' or 'york_urban')> <folder containing the pre-extracted line segments> <output folder> <method name>
```
On HPatches, this could look like:
```
python -m deeplsd.scripts.evaluate_line_detection hpatches hpatches_outputs hpatches_evaluation deeplsd
```

### VP estimation metrics
We provide dataloaders for the following datasets:
- [York Urban DB](https://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/)
- [NYU-VP](https://github.com/fkluger/nyu_vp)

First, the VP needs to be exported. For example, exporting line detections on York Urban would look like:
```
python -m deeplsd.scripts.export_features deeplsd/configs/export_york_urban.yaml weights/deeplsd_md.tar yud_outputs --pred_vps
```

The evaluation can then be run with:
```
python -m deeplsd.scripts.evaluate_vp_estimation <dataset name ('york_urban' or 'nyu')> <folder containing the pre-extracted VPs> <output folder> <method name>
```
On York Urban, this could look like:
```
python -m deeplsd.scripts.evaluate_vp_estimation york_urban yud_outputs yud_evaluation deeplsd
```

**Note:** the 3D line reconstruction and visual localization applications of the paper will be released in a separate repository.

## Bibtex
If you use this code in your project, please consider citing the following paper:
```bibtex
@InProceedings{Pautrat_2023_DeepLSD,
    author = {Pautrat, Rémi and Barath, Daniel and Larsson, Viktor and Oswald, Martin R. and Pollefeys, Marc},
    title = {DeepLSD: Line Segment Detection and Refinement with Deep Image Gradients},
    booktitle = {Computer Vision and Pattern Recognition (CVPR)},
    year = {2023},
}
```
