import sys
from pathlib import Path

import cv2

from hloc import logger
from ui.utils import DEVICE, ROOT, get_matcher_zoo, load_config

sys.path.append(str(Path(__file__).parents[1]))
from api.server import ImageMatchingAPI


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
        skip_ci = config["matcher_zoo"][k].get("skip_ci", False)
        if enable and not skip_ci:
            logger.info(f"Testing {k} ...")
            api = ImageMatchingAPI(conf=v, device=DEVICE)
            api(image0, image1)
            log_path = ROOT / "experiments" / "all"
            log_path.mkdir(exist_ok=True, parents=True)
            api.visualize(log_path=log_path)
        else:
            logger.info(f"Skipping {k} ...")
    return 0


def test_one():
    img_path1 = ROOT / "datasets/sacre_coeur/mapping/02928139_3448003521.jpg"
    img_path2 = ROOT / "datasets/sacre_coeur/mapping/17295357_9106075285.jpg"
    image0 = cv2.imread(str(img_path1))[:, :, ::-1]  # RGB
    image1 = cv2.imread(str(img_path2))[:, :, ::-1]  # RGB
    # sparse
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
    api = ImageMatchingAPI(conf=conf, device=DEVICE)
    api(image0, image1)
    log_path = ROOT / "experiments" / "one"
    log_path.mkdir(exist_ok=True, parents=True)
    api.visualize(log_path=log_path)

    # dense
    conf = {
        "matcher": {
            "output": "matches-loftr",
            "model": {
                "name": "loftr",
                "weights": "outdoor",
                "max_keypoints": 2000,
                "match_threshold": 0.2,
            },
            "preprocessing": {
                "grayscale": True,
                "resize_max": 1024,
                "dfactor": 8,
                "width": 640,
                "height": 480,
                "force_resize": True,
            },
            "max_error": 1,
            "cell_size": 1,
        },
        "dense": True,
    }

    api = ImageMatchingAPI(conf=conf, device=DEVICE)
    api(image0, image1)
    log_path = ROOT / "experiments" / "one"
    log_path.mkdir(exist_ok=True, parents=True)
    api.visualize(log_path=log_path)
    return 0


if __name__ == "__main__":
    config = load_config(ROOT / "ui/config.yaml")
    test_one()
    test_all(config)
