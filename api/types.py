from typing import List

from pydantic import BaseModel


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
