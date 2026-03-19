import cv2
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from imcui.ui.utils import DEVICE, get_matcher_zoo
from imcui.api import ImageMatchingAPI


def load_image_rgb(path):
    """Load image as RGB float32 in [0, 1] range."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = img[:, :, ::-1].copy()  # BGR to RGB
    return img.astype(np.float32) / 255.0


def test_all():
    # matcher_zoo is dynamically loaded from vismatch
    img_path1 = ROOT / "tests/data/02928139_3448003521.jpg"
    img_path2 = ROOT / "tests/data/17295357_9106075285.jpg"
    image0 = load_image_rgb(img_path1)
    image1 = load_image_rgb(img_path2)

    # Get matcher zoo dynamically (no config needed)
    matcher_zoo = get_matcher_zoo()
    for k, v in list(matcher_zoo.items())[:3]:  # Test first 3 matchers
        if image0 is None or image1 is None:
            print(f"Error: No images found for {k}")
            continue
        print(f"Testing {k} ...")
        api = ImageMatchingAPI(conf=v, device=DEVICE)
        pred = api(image0, image1)
        assert pred is not None
    print("All tests passed!")


def test_one():
    img_path1 = ROOT / "tests/data/02928139_3448003521.jpg"
    img_path2 = ROOT / "tests/data/17295357_9106075285.jpg"

    image0 = load_image_rgb(img_path1)
    image1 = load_image_rgb(img_path2)
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
    pred = api(image0, image1)
    assert pred is not None
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
    pred = api(image0, image1)
    assert pred is not None
    log_path = ROOT / "experiments" / "one"
    log_path.mkdir(exist_ok=True, parents=True)
    api.visualize(log_path=log_path)


if __name__ == "__main__":
    test_one()
    test_all()
