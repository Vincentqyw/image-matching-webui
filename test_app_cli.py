import cv2
import warnings
from pathlib import Path
from hloc import logger
from hloc import matchers, extractors, logger
from hloc import match_dense, match_features, extract_features
from hloc.utils.viz import add_text, plot_keypoints
from common.utils import (
    load_config,
    get_model,
    get_feature_model,
    ransac_zoo,
    get_matcher_zoo,
    filter_matches,
    device,
    ROOT,
)
from common.viz import (
    fig2im,
    plot_images,
    display_matches,
    plot_color_line_matches,
)
import time
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")


def test_modules(config: dict):
    img_path1 = ROOT / "datasets/sacre_coeur/mapping/02928139_3448003521.jpg"
    img_path2 = ROOT / "datasets/sacre_coeur/mapping/17295357_9106075285.jpg"
    image0 = cv2.imread(str(img_path1))
    image1 = cv2.imread(str(img_path2))
    keypoint_threshold = 0.0
    extract_max_keypoints = 2000
    match_threshold = 0.2
    log_path = ROOT / "experiments"
    log_path.mkdir(exist_ok=True, parents=True)

    matcher_zoo_restored = get_matcher_zoo(config["matcher_zoo"])
    for k, v in matcher_zoo_restored.items():
        if image0 is None or image1 is None:
            logger.error("Error: No images found! Please upload two images.")
        # init output
        output_keypoints = None
        output_matches_raw = None
        output_matches_ransac = None
        match_conf = v["matcher"]

        # update match config
        match_conf["model"]["match_threshold"] = match_threshold
        match_conf["model"]["max_keypoints"] = extract_max_keypoints
        matcher = get_model(match_conf)
        t1 = time.time()
        if v["dense"]:
            pred = match_dense.match_images(
                matcher,
                image0,
                image1,
                match_conf["preprocessing"],
                device=device,
            )
            del matcher
            extract_conf = None
            last_fixed = "{}".format(match_conf["model"]["name"])
        else:
            extract_conf = v["feature"]

            # update extract config
            extract_conf["model"]["max_keypoints"] = extract_max_keypoints
            extract_conf["model"]["keypoint_threshold"] = keypoint_threshold
            extractor = get_feature_model(extract_conf)
            pred0 = extract_features.extract(
                extractor, image0, extract_conf["preprocessing"]
            )
            pred1 = extract_features.extract(
                extractor, image1, extract_conf["preprocessing"]
            )
            pred = match_features.match_images(matcher, pred0, pred1)
            del extractor
            last_fixed = "{}_{}".format(
                extract_conf["model"]["name"], match_conf["model"]["name"]
            )

        # keypoints on images
        logger.info(f"Match features done using: {time.time()-t1:.3f}s")
        t1 = time.time()
        texts = [
            f"image pairs: {img_path1.name} & {img_path2.name}",
            "",
        ]
        titles = [
            "Image 0 - Keypoints",
            "Image 1 - Keypoints",
        ]
        output_keypoints = plot_images([image0, image1], titles=titles, dpi=300)
        if "keypoints0" in pred.keys() and "keypoints1" in pred.keys():
            plot_keypoints([pred["keypoints0"], pred["keypoints1"]])
            text = (
                f"# keypoints0: {len(pred['keypoints0'])} \n"
                + f"# keypoints1: {len(pred['keypoints1'])}"
            )
            add_text(0, text, fs=15)
        output_keypoints = fig2im(output_keypoints)

        # plot images with raw matches
        titles = [
            "Image 0 - Raw matched keypoints",
            "Image 1 - Raw matched keypoints",
        ]
        output_matches_raw, num_matches_raw = display_matches(
            pred, titles=titles
        )
        logger.info(f"Plot keypoints done using: {time.time()-t1:.3f}s")
        t1 = time.time()

        filter_matches(
            pred,
            ransac_method=config["defaults"]["ransac_method"],
            ransac_reproj_threshold=config["defaults"][
                "ransac_reproj_threshold"
            ],
            ransac_confidence=config["defaults"]["ransac_confidence"],
            ransac_max_iter=config["defaults"]["ransac_max_iter"],
        )
        # plot images with ransac matches
        titles = [
            "Image 0 - Ransac matched keypoints",
            "Image 1 - Ransac matched keypoints",
        ]
        output_matches_ransac, num_matches_ransac = display_matches(
            pred, titles=titles
        )
        logger.info(f"RANSAC matches done using: {time.time()-t1:.3f}s")

        img_keypoints_path = log_path / f"img_keypoints_{last_fixed}.png"
        img_matches_raw_path = log_path / f"img_matches_raw_{last_fixed}.png"
        img_matches_ransac_path = (
            log_path / f"img_matches_ransac_{last_fixed}.png"
        )
        cv2.imwrite(str(img_keypoints_path), output_keypoints)
        cv2.imwrite(str(img_matches_raw_path), output_matches_raw)
        cv2.imwrite(str(img_matches_ransac_path), output_matches_ransac)

        plt.close("all")


if __name__ == "__main__":
    import argparse

    config = load_config(ROOT / "common/config.yaml")
    test_modules(config)
