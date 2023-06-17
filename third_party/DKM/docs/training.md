Here we provide instructions for how to train our models, including download of datasets.

### MegaDepth
First the MegaDepth dataset needs to be downloaded and preprocessed. This can be done by the following steps:
1. Download MegaDepth from here: https://www.cs.cornell.edu/projects/megadepth/
2. Extract and preprocess: See https://github.com/mihaidusmanu/d2-net
3. Download our prepared scene info from here: https://github.com/Parskatt/storage/releases/download/prep_scene_info/prep_scene_info.tar
4. File structure should be data/megadepth/phoenix, data/megadepth/Undistorted_SfM, data/megadepth/prep_scene_info.
Then run 
``` bash
python experiments/dkmv3/train_DKMv3_outdoor.py --gpus 4
```

## Megadepth + Scannet
First follow the steps outlined above.
Then, see https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md

Then run 
``` bash
python experiments/dkmv3/train_DKMv3_indoor.py --gpus 4
```
