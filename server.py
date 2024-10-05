# server.py
from pathlib import Path
from typing import Union

import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from pydantic import BaseModel

from ui.api import ImageMatchingAPI
from ui.utils import DEVICE


class ImageInfo(BaseModel):
    image_path: str
    max_keypoints: int
    reference_points: list


class ImageMatchingService:
    def __init__(self, conf: dict, device: str):
        self.api = ImageMatchingAPI(conf=conf, device=device)
        self.app = FastAPI()
        self.register_routes()

    def register_routes(self):
        @self.app.post("/v1/match")
        async def match(
            image0: UploadFile = File(...), image1: UploadFile = File(...)
        ):
            try:
                image0_array = self.load_image(image0)
                image1_array = self.load_image(image1)

                output = self.api(image0_array, image1_array)

                skip_keys = ["image0_orig", "image1_orig"]
                pred = self.filter_output(output, skip_keys)

                return JSONResponse(content=pred)
            except Exception as e:
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.post("/v1/extract")
        async def extract(image: UploadFile = File(...)):
            try:
                image_array = self.load_image(image)
                output = self.api.extract(image_array)
                skip_keys = ["descriptors", "image", "image_orig"]
                pred = self.filter_output(output, skip_keys)
                return JSONResponse(content=pred)
            except Exception as e:
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.post("/v2/extract")
        async def extract_v2(image_path: ImageInfo):
            img_path = image_path.image_path
            try:
                safe_path = Path(img_path).resolve(strict=False)
                image_array = self.load_image(str(safe_path))
                output = self.api.extract(image_array)
                skip_keys = ["descriptors", "image", "image_orig"]
                pred = self.filter_output(output, skip_keys)
                return JSONResponse(content=pred)
            except Exception as e:
                return JSONResponse(content={"error": str(e)}, status_code=500)

    def load_image(self, file_path: Union[str, UploadFile]) -> np.ndarray:
        """
        Reads an image from a file path or an UploadFile object.

        Args:
            file_path: A file path or an UploadFile object.

        Returns:
            A numpy array representing the image.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path).resolve(strict=False)
        else:
            file_path = file_path.file
        with Image.open(file_path) as img:
            image_array = np.array(img)
        return image_array

    def filter_output(self, output: dict, skip_keys: list) -> dict:
        pred = {}
        for key, value in output.items():
            if key in skip_keys:
                continue
            if isinstance(value, np.ndarray):
                pred[key] = value.tolist()
        return pred

    def run(self, host: str = "0.0.0.0", port: int = 8001):
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
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

    service = ImageMatchingService(conf=conf, device=DEVICE)
    service.run()
