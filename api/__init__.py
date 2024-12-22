import base64
import io
import sys
from pathlib import Path
from typing import List

import numpy as np
from fastapi.exceptions import HTTPException
from PIL import Image
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parents[1]))

from hloc import logger

from .core import ImageMatchingAPI


class ImagesInput(BaseModel):
    data: List[str] = []
    max_keypoints: List[int] = []
    timestamps: List[str] = []
    grayscale: bool = False
    image_hw: List[List[int]] = [[], []]
    feature_type: int = 0
    rotates: List[float] = []
    scales: List[float] = []
    reference_points: List[List[float]] = []
    binarize: bool = False


def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        image = Image.open(io.BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as e:
        logger.warning(f"API cannot decode image: {e}")
        raise HTTPException(status_code=500, detail="Invalid encoded image") from e


def to_base64_nparray(encoding: str) -> np.ndarray:
    return np.array(decode_base64_to_image(encoding)).astype("uint8")


__all__ = [
    "ImageMatchingAPI",
    "ImagesInput",
    "decode_base64_to_image",
    "to_base64_nparray",
]
