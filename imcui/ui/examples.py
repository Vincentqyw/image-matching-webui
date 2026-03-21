"""Example image generation utilities for the UI."""

import os
import random
from itertools import combinations
from pathlib import Path
from typing import Any, List

import numpy as np

# Import defaults from config
from .config import (
    DEFAULT_DEFAULT_KEYPOINT_THRESHOLD,
    DEFAULT_RANSAC_CONFIDENCE,
    DEFAULT_RANSAC_MAX_ITER,
    DEFAULT_RANSAC_METHOD,
    DEFAULT_RANSAC_REPROJ_THRESHOLD,
    DEFAULT_SETTING_MAX_FEATURES,
    DEFAULT_SETTING_THRESHOLD,
)


def gen_examples(data_root: Path = None) -> List[List[Any]]:
    """Generate example image pairs for the UI.

    Args:
        data_root: Root directory containing example datasets.
                   If None, uses imcui.ui.config_utils.get_example_data_path()

    Returns:
        List of example input lists for Gradio Examples component.
    """
    # Import here to avoid circular dependency
    from .config_utils import get_example_data_path

    if data_root is None:
        data_root = get_example_data_path()
    else:
        data_root = Path(data_root)

    random.seed(1)
    # Use vismatch available models for examples
    example_algos = [
        "disk-lightglue",
        "xfeat",
        "dedode",
        "loftr",
        "roma",
        "sift-lightglue",
        "d2net",
        "aspanformer",
        "topicfm",
        "superpoint-lightglue",
    ]
    example_algos_rotation_robust = [
        "sift-nn",
        "orb-nn",
        "sift-lightglue",
        # "GIM(dkm)",
    ]

    def distribute_elements(A, B):
        new_B = np.array(B, copy=True).flatten()
        np.random.shuffle(new_B)
        new_B = np.resize(new_B, len(A))
        np.random.shuffle(new_B)
        return new_B.tolist()

    # normal examples
    def gen_images_pairs(count: int = 5):
        path = str(data_root / "sacre_coeur/mapping")
        imgs_list = [
            os.path.join(path, file)
            for file in os.listdir(path)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        pairs = list(combinations(imgs_list, 2))
        if len(pairs) < count:
            count = len(pairs)
        selected = random.sample(range(len(pairs)), count)
        return [pairs[i] for i in selected]

    # rotated examples
    def gen_rot_image_pairs(count: int = 5):
        path = data_root / "sacre_coeur/mapping"
        path_rot = data_root / "sacre_coeur/mapping_rot"
        rot_list = [45, 180, 90, 225, 270]
        pairs = []
        for file in os.listdir(path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                for rot in rot_list:
                    file_rot = "{}_rot{}.jpg".format(Path(file).stem, rot)
                    if (path_rot / file_rot).exists():
                        pairs.append(
                            [
                                path / file,
                                path_rot / file_rot,
                            ]
                        )
        if len(pairs) < count:
            count = len(pairs)
        selected = random.sample(range(len(pairs)), count)
        return [pairs[i] for i in selected]

    def gen_scale_image_pairs(count: int = 5):
        path = data_root / "sacre_coeur/mapping"
        path_scale = data_root / "sacre_coeur/mapping_scale"
        scale_list = [0.3, 0.5]
        pairs = []
        for file in os.listdir(path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                for scale in scale_list:
                    file_scale = "{}_scale{}.jpg".format(Path(file).stem, scale)
                    if (path_scale / file_scale).exists():
                        pairs.append(
                            [
                                path / file,
                                path_scale / file_scale,
                            ]
                        )
        if len(pairs) < count:
            count = len(pairs)
        selected = random.sample(range(len(pairs)), count)
        return [pairs[i] for i in selected]

    # extramely hard examples
    def gen_image_pairs_wxbs(count: int = None):
        prefix = "wxbs_benchmark/.WxBS/v1.1"
        wxbs_path = data_root / prefix
        pairs = []
        for catg in os.listdir(wxbs_path):
            catg_path = wxbs_path / catg
            if not catg_path.is_dir():
                continue
            for scene in os.listdir(catg_path):
                scene_path = catg_path / scene
                if not scene_path.is_dir():
                    continue
                img1_path = scene_path / "01.png"
                img2_path = scene_path / "02.png"
                if img1_path.exists() and img2_path.exists():
                    pairs.append([str(img1_path), str(img2_path)])
        return pairs

    # image pair path
    pairs = gen_images_pairs()
    # pairs += gen_rot_image_pairs()
    pairs += gen_scale_image_pairs()
    pairs += gen_image_pairs_wxbs()
    pairs_rotation = gen_rot_image_pairs()
    dist_examples = distribute_elements(pairs, example_algos)
    dist_examples_rotation = distribute_elements(
        pairs_rotation, example_algos_rotation_robust
    )
    pairs = pairs_rotation + pairs
    dist_examples = dist_examples_rotation + dist_examples
    match_setting_threshold = DEFAULT_SETTING_THRESHOLD
    match_setting_max_features = DEFAULT_SETTING_MAX_FEATURES
    detect_keypoints_threshold = DEFAULT_DEFAULT_KEYPOINT_THRESHOLD
    ransac_method = DEFAULT_RANSAC_METHOD
    ransac_reproj_threshold = DEFAULT_RANSAC_REPROJ_THRESHOLD
    ransac_confidence = DEFAULT_RANSAC_CONFIDENCE
    ransac_max_iter = DEFAULT_RANSAC_MAX_ITER
    input_lists = []

    for pair, mt in zip(pairs, dist_examples):
        input_lists.append(
            [
                pair[0],
                pair[1],
                match_setting_threshold,
                match_setting_max_features,
                detect_keypoints_threshold,
                mt,
                ransac_method,
                ransac_reproj_threshold,
                ransac_confidence,
                ransac_max_iter,
            ]
        )
    return input_lists
