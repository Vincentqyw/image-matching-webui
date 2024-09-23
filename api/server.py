# server.py
import litserve as ls

import cv2
import warnings
import numpy as np
from pathlib import Path
import sys
import json

sys.path.append("..")
from PIL import Image

from ui.utils import (
    get_matcher_zoo,
    load_config,
    DEVICE,
    ROOT,
)

from ui.api import ImageMatchingAPI


# (STEP 1) - DEFINE THE API (compound AI system)
class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        conf = {
            "feature": {
                "output": "feats-superpoint-n4096-rmax1600",
                "model": {
                    "name": "superpoint",
                    "nms_radius": 3,
                    "max_keypoints": 4096,
                    "keypoint_threshold": 0.005,
                },
                "preprocessing": {
                    "grayscale": True,
                    "force_resize": True,
                    "resize_max": 1600,
                    "width": 640,
                    "height": 480,
                    "dfactor": 8,
                },
            },
            "matcher": {
                "output": "matches-NN-mutual",
                "model": {
                    "name": "nearest_neighbor",
                    "do_mutual_check": True,
                    "match_threshold": 0.2,
                },
            },
            "dense": False,
        }
        self.api = ImageMatchingAPI(conf=conf, device=DEVICE)

    def decode_request(self, request):
        # Convert the request payload to model input.
        return {
            "image0": request["image0"].file,
            "image1": request["image1"].file,
        }

    def predict(self, data):
        # Easily build compound systems. Run inference and return the output.
        image0 = np.array(Image.open(data["image0"]))
        image1 = np.array(Image.open(data["image1"]))
        output = self.api(image0, image1)
        print(output.keys())
        return output

    def encode_response(self, output):
        skip_keys = ["image0_orig", "image1_orig"]
        pred = {}
        for key, value in output.items():
            if key in skip_keys:
                continue
            if isinstance(value, np.ndarray):
                pred[key] = value.tolist()
        return json.dumps(pred)


# (STEP 2) - START THE SERVER
if __name__ == "__main__":
    server = ls.LitServer(
        SimpleLitAPI(),
        accelerator="auto",
        api_path="/v1/predict",
        max_batch_size=1,
    )
    server.run(port=8001)
