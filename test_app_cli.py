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


def test_all(config: dict = None):
    img_path1 = ROOT / "datasets/sacre_coeur/mapping/02928139_3448003521.jpg"
    img_path2 = ROOT / "datasets/sacre_coeur/mapping/17295357_9106075285.jpg"
    image0 = cv2.imread(str(img_path1))[:, :, ::-1]  # RGB
    image1 = cv2.imread(str(img_path2))[:, :, ::-1]  # RGB

    matcher_zoo_restored = get_matcher_zoo(config["matcher_zoo"])
    for k, v in matcher_zoo_restored.items():
        if image0 is None or image1 is None:
            logger.error("Error: No images found! Please upload two images.")
        enable = config["matcher_zoo"][k].get("enable", True)
        if enable:
            logger.info(f"Testing {k} ...")
            api = ImageMatchingAPI(conf=v, device=device)
            api(image0, image1)
            log_path = ROOT / "experiments" / "all"
            log_path.mkdir(exist_ok=True, parents=True)
            api.visualize(log_path=log_path)
        else:
            logger.info(f"Skipping {k} ...")


def test_one():
    img_path1 = ROOT / "datasets/sacre_coeur/mapping/02928139_3448003521.jpg"
    img_path2 = ROOT / "datasets/sacre_coeur/mapping/17295357_9106075285.jpg"
    image0 = cv2.imread(str(img_path1))[:, :, ::-1]  # RGB
    image1 = cv2.imread(str(img_path2))[:, :, ::-1]  # RGB
    # sparse
    conf = {
        "dense": False,
        "matcher": {
            "model": {
                "name": "NN-mutual",
                "match_threshold": 0.2,
            }
        },
        "feature": {
            "model": {
                "name": "xfeat",
                "max_keypoints": 1024,
                "keypoint_threshold": 0.015,
            }
        },
        "ransac": {
            "enable": True,
            "estimator": "poselib",
            "geometry": "homography",
            "method": "RANSAC",
            "reproj_threshold": 3,
            "confidence": 0.9999,
            "max_iter": 10000,
        },
    }
    api = ImageMatchingAPI(conf=conf, device=device)
    api(image0, image1)
    log_path = ROOT / "experiments" / "one"
    log_path.mkdir(exist_ok=True, parents=True)
    api.visualize(log_path=log_path)

    # dense
    conf = {
        "dense": True,
        "matcher": {
            "model": {
                "name": "loftr",
                "match_threshold": 0.2,
            }
        },
        "feature": {},
        "ransac": {
            "enable": True,
            "estimator": "poselib",
            "geometry": "homography",
            "method": "RANSAC",
            "reproj_threshold": 3,
            "confidence": 0.9999,
            "max_iter": 10000,
        },
    }
    api = ImageMatchingAPI(conf=conf, device=device)
    api(image0, image1)
    log_path = ROOT / "experiments" / "one"
    log_path.mkdir(exist_ok=True, parents=True)
    api.visualize(log_path=log_path)


if __name__ == "__main__":
    import argparse

    config = load_config(ROOT / "common/config.yaml")
    test_one()
    test_all(config)
