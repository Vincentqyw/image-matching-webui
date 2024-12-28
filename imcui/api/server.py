# server.py
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import ray
import torch
import yaml
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ray import serve

from imcui.api import ImageMatchingAPI, ImagesInput, to_base64_nparray
from imcui.hloc import DEVICE
from imcui.ui import get_version

warnings.simplefilter("ignore")
app = FastAPI()
if ray.is_initialized():
    ray.shutdown()
ray.init(
    dashboard_port=8265,
    ignore_reinit_error=True,
)
serve.start(
    http_options={"host": "0.0.0.0", "port": 8001},
)

num_gpus = 1 if torch.cuda.is_available() else 0


@serve.deployment(
    num_replicas=4, ray_actor_options={"num_cpus": 2, "num_gpus": num_gpus}
)
@serve.ingress(app)
class ImageMatchingService:
    def __init__(self, conf: dict, device: str):
        self.conf = conf
        self.api = ImageMatchingAPI(conf=conf, device=device)

    @app.get("/")
    def root(self):
        return "Hello, world!"

    @app.get("/version")
    async def version(self):
        return {"version": get_version()}

    @app.post("/v1/match")
    async def match(
        self, image0: UploadFile = File(...), image1: UploadFile = File(...)
    ):
        """
        Handle the image matching request and return the processed result.

        Args:
            image0 (UploadFile): The first image file for matching.
            image1 (UploadFile): The second image file for matching.

        Returns:
            JSONResponse: A JSON response containing the filtered match results
                            or an error message in case of failure.
        """
        try:
            # Load the images from the uploaded files
            image0_array = self.load_image(image0)
            image1_array = self.load_image(image1)

            # Perform image matching using the API
            output = self.api(image0_array, image1_array)

            # Keys to skip in the output
            skip_keys = ["image0_orig", "image1_orig"]

            # Postprocess the output to filter unwanted data
            pred = self.postprocess(output, skip_keys)

            # Return the filtered prediction as a JSON response
            return JSONResponse(content=pred)
        except Exception as e:
            # Return an error message with status code 500 in case of exception
            return JSONResponse(content={"error": str(e)}, status_code=500)

    @app.post("/v1/extract")
    async def extract(self, input_info: ImagesInput):
        """
        Extract keypoints and descriptors from images.

        Args:
            input_info: An object containing the image data and options.

        Returns:
            A list of dictionaries containing the keypoints and descriptors.
        """
        try:
            preds = []
            for i, input_image in enumerate(input_info.data):
                # Load the image from the input data
                image_array = to_base64_nparray(input_image)
                # Extract keypoints and descriptors
                output = self.api.extract(
                    image_array,
                    max_keypoints=input_info.max_keypoints[i],
                    binarize=input_info.binarize,
                )
                # Do not return the original image and image_orig
                # skip_keys = ["image", "image_orig"]
                skip_keys = []

                # Postprocess the output
                pred = self.postprocess(output, skip_keys)
                preds.append(pred)
            # Return the list of extracted features
            return JSONResponse(content=preds)
        except Exception as e:
            # Return an error message if an exception occurs
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

    def postprocess(self, output: dict, skip_keys: list, binarize: bool = True) -> dict:
        pred = {}
        for key, value in output.items():
            if key in skip_keys:
                continue
            if isinstance(value, np.ndarray):
                pred[key] = value.tolist()
        return pred

    def run(self, host: str = "0.0.0.0", port: int = 8001):
        import uvicorn

        uvicorn.run(app, host=host, port=port)


def read_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        conf = yaml.safe_load(f)
    return conf


# api server
conf = read_config(Path(__file__).parent / "config/api.yaml")
service = ImageMatchingService.bind(conf=conf["api"], device=DEVICE)
handle = serve.run(service, route_prefix="/")

# serve run api.server_ray:service

# build to generate config file
# serve build api.server_ray:service -o api/config/ray.yaml
# serve run api/config/ray.yaml
