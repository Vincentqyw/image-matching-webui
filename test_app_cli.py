import cv2
import warnings
import numpy as np
from pathlib import Path
from hloc import logger
from common.utils import (
    get_matcher_zoo,
    load_config,
    device,
    ROOT,
)
from common.api import ImageMatchingAPI


def test_api(config: dict = None):
    img_path1 = ROOT / "datasets/sacre_coeur/mapping/02928139_3448003521.jpg"
    img_path2 = ROOT / "datasets/sacre_coeur/mapping/17295357_9106075285.jpg"
    image0 = cv2.imread(str(img_path1))[:, :, ::-1]
    image1 = cv2.imread(str(img_path2))[:, :, ::-1]

    matcher_zoo_restored = get_matcher_zoo(config["matcher_zoo"])
    for k, v in matcher_zoo_restored.items():
        if image0 is None or image1 is None:
            logger.error("Error: No images found! Please upload two images.")
        enable = config["matcher_zoo"][k].get("enable", True)
        if enable:
            logger.info(f"Testing {k} ...")
            api = ImageMatchingAPI(conf=v, device=device)
            api(image0, image1)
            log_path = ROOT / "experiments1"
            log_path.mkdir(exist_ok=True, parents=True)
            api.visualize(log_path=log_path)
        else:
            logger.info(f"Skipping {k} ...")


if __name__ == "__main__":
    import argparse

    config = load_config(ROOT / "common/config.yaml")
    test_api(config)
