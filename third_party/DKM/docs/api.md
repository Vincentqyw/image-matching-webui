## Creating a model
```python
from dkm import DKMv3_outdoor, DKMv3_indoor
DKMv3_outdoor() # creates an outdoor trained model
DKMv3_indoor() # creates an indoor trained model
```
## Model settings
Note: Non-exhaustive list
```python
model.upsample_preds = True/False # Whether to upsample the predictions to higher resolution
model.upsample_res = (H_big, W_big) # Which resolution to use for upsampling
model.symmetric = True/False # Whether to compute a bidirectional warp
model.w_resized = W # width of image used
model.h_resized = H # height of image used
model.sample_mode = "threshold_balanced" # method for sampling matches. threshold_balanced is what was used in the paper
model.sample_threshold = 0.05 # the threshold for sampling, 0.05 works well for megadepth, for IMC2022 we found 0.2 to work better.
```
## Running model
```python
warp, certainty = model.match(im_A, im_B) # produces a warp of shape [B,H,W,4] and certainty of shape [B,H,W]
matches, certainty = model.sample(warp, certainty) # samples from the warp using the certainty
kpts_A, kpts_B = model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B) # convenience function to convert normalized matches to pixel coordinates
```

