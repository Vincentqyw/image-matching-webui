Benchmarking datasets for geometry estimation can be somewhat cumbersome to download. We provide instructions for the benchmarks we use below, and are happy to answer any questions.

### HPatches
First, make sure that the "data/hpatches" path exists, e.g. by

`` ln -s place/where/your/datasets/are/stored/hpatches data/hpatches `` 

Then run (if you don't already have hpatches downloaded)

`` bash scripts/download_hpatches.sh``

### Megadepth-1500 (LoFTR Split)
1. We use the split made by LoFTR, which can be downloaded here https://drive.google.com/drive/folders/1nTkK1485FuwqA0DbZrK2Cl0WnXadUZdc. (You can also use the preprocessed megadepth dataset if you have it available)
2. The images should be located in data/megadepth/Undistorted_SfM/0015 and 0022.
3. The pair infos are provided here https://github.com/zju3dv/LoFTR/tree/master/assets/megadepth_test_1500_scene_info 
3. Put those files in data/megadepth/xxx

### Megadepth-8-Scenes (DKM Split)
1. The pair infos are provided in [assets](../assets/)
2. Put those files in data/megadepth/xxx


### Scannet-1500 (SuperGlue Split)
We use the same split of scannet as superglue.
1. LoFTR provides the split here: https://drive.google.com/drive/folders/1nTkK1485FuwqA0DbZrK2Cl0WnXadUZdc
2. Note that ScanNet requires you to sign a License agreement, which can be found http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf
3. This benchmark should be put in the data/scannet folder
