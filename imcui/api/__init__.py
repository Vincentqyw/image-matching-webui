import base64
import io
from typing import List

import numpy as np
from fastapi.exceptions import HTTPException
from PIL import Image
from pydantic import BaseModel

from ..hloc import logger
from .core import ImageMatchingAPI


class ImagesInput(BaseModel):
    data: list[str] = []
    max_keypoints: list[int] = []
    timestamps: list[str] = []
    grayscale: bool = False
    image_hw: list[list[int]] = [[], []]
    feature_type: int = 0
    rotates: list[float] = []
    scales: list[float] = []
    reference_points: list[list[float]] = []
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
