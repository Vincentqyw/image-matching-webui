import argparse
import pprint
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import cv2
import h5py
import numpy as np
import torch
import torchvision.transforms.functional as F
from scipy.spatial import KDTree
from tqdm import tqdm

from . import logger, matchers
from .extract_features import read_image, resize_image
from .match_features import find_unique_new_pairs
from .utils.base_model import dynamic_load
from .utils.io import list_h5_names
from .utils.parsers import names_to_pair, parse_retrieval

device = "cuda" if torch.cuda.is_available() else "cpu"

confs = {
    # Best quality but loads of points. Only use for small scenes
    "loftr": {
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
        "max_error": 1,  # max error for assigned keypoints (in px)
        "cell_size": 1,  # size of quantization patch (max 1 kp/patch)
    },
    "eloftr": {
        "output": "matches-eloftr",
        "model": {
            "name": "eloftr",
            "weights": "weights/eloftr_outdoor.ckpt",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 32,
            "width": 640,
            "height": 480,
            "force_resize": True,
        },
        "max_error": 1,  # max error for assigned keypoints (in px)
        "cell_size": 1,  # size of quantization patch (max 1 kp/patch)
    },
    "xoftr": {
        "output": "matches-xoftr",
        "model": {
            "name": "xoftr",
            "weights": "weights_xoftr_640.ckpt",
            "max_keypoints": 2000,
            "match_threshold": 0.3,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 8,
            "width": 640,
            "height": 480,
            "force_resize": True,
        },
        "max_error": 1,  # max error for assigned keypoints (in px)
        "cell_size": 1,  # size of quantization patch (max 1 kp/patch)
    },
    # "loftr_quadtree": {
    #     "output": "matches-loftr-quadtree",
    #     "model": {
    #         "name": "quadtree",
    #         "weights": "outdoor",
    #         "max_keypoints": 2000,
    #         "match_threshold": 0.2,
    #     },
    #     "preprocessing": {
    #         "grayscale": True,
    #         "resize_max": 1024,
    #         "dfactor": 8,
    #         "width": 640,
    #         "height": 480,
    #         "force_resize": True,
    #     },
    #     "max_error": 1,  # max error for assigned keypoints (in px)
    #     "cell_size": 1,  # size of quantization patch (max 1 kp/patch)
    # },
    "cotr": {
        "output": "matches-cotr",
        "model": {
            "name": "cotr",
            "weights": "out/default",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1024,
            "dfactor": 8,
            "width": 640,
            "height": 480,
            "force_resize": True,
        },
        "max_error": 1,  # max error for assigned keypoints (in px)
        "cell_size": 1,  # size of quantization patch (max 1 kp/patch)
    },
    # Semi-scalable loftr which limits detected keypoints
    "loftr_aachen": {
        "output": "matches-loftr_aachen",
        "model": {
            "name": "loftr",
            "weights": "outdoor",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {"grayscale": True, "resize_max": 1024, "dfactor": 8},
        "max_error": 2,  # max error for assigned keypoints (in px)
        "cell_size": 8,  # size of quantization patch (max 1 kp/patch)
    },
    # Use for matching superpoint feats with loftr
    "loftr_superpoint": {
        "output": "matches-loftr_aachen",
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
        "max_error": 4,  # max error for assigned keypoints (in px)
        "cell_size": 4,  # size of quantization patch (max 1 kp/patch)
    },
    # Use topicfm for matching feats
    "topicfm": {
        "output": "matches-topicfm",
        "model": {
            "name": "topicfm",
            "weights": "outdoor",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": True,
            "force_resize": True,
            "resize_max": 1024,
            "dfactor": 8,
            "width": 640,
            "height": 480,
        },
    },
    # Use aspanformer for matching feats
    "aspanformer": {
        "output": "matches-aspanformer",
        "model": {
            "name": "aspanformer",
            "weights": "outdoor",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": True,
            "force_resize": True,
            "resize_max": 1024,
            "width": 640,
            "height": 480,
            "dfactor": 8,
        },
    },
    "duster": {
        "output": "matches-duster",
        "model": {
            "name": "duster",
            "weights": "vit_large",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 512,
            "dfactor": 16,
        },
    },
    "mast3r": {
        "output": "matches-mast3r",
        "model": {
            "name": "mast3r",
            "weights": "vit_large",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 512,
            "dfactor": 16,
        },
    },
    "xfeat_lightglue": {
        "output": "matches-xfeat_lightglue",
        "model": {
            "name": "xfeat_lightglue",
            "max_keypoints": 8000,
        },
        "preprocessing": {
            "grayscale": False,
            "force_resize": False,
            "resize_max": 1024,
            "width": 640,
            "height": 480,
            "dfactor": 8,
        },
    },
    "xfeat_dense": {
        "output": "matches-xfeat_dense",
        "model": {
            "name": "xfeat_dense",
            "max_keypoints": 8000,
        },
        "preprocessing": {
            "grayscale": False,
            "force_resize": False,
            "resize_max": 1024,
            "width": 640,
            "height": 480,
            "dfactor": 8,
        },
    },
    "dkm": {
        "output": "matches-dkm",
        "model": {
            "name": "dkm",
            "weights": "outdoor",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": False,
            "force_resize": True,
            "resize_max": 1024,
            "width": 80,
            "height": 60,
            "dfactor": 8,
        },
    },
    "roma": {
        "output": "matches-roma",
        "model": {
            "name": "roma",
            "weights": "outdoor",
            "model_name": "roma_outdoor.pth",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": False,
            "force_resize": True,
            "resize_max": 1024,
            "width": 320,
            "height": 240,
            "dfactor": 8,
        },
    },
    "gim(dkm)": {
        "output": "matches-gim",
        "model": {
            "name": "gim",
            "model_name": "gim_dkm_100h.ckpt",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": False,
            "force_resize": True,
            "resize_max": 1024,
            "width": 320,
            "height": 240,
            "dfactor": 8,
        },
    },
    "omniglue": {
        "output": "matches-omniglue",
        "model": {
            "name": "omniglue",
            "match_threshold": 0.2,
            "max_keypoints": 2000,
            "features": "null",
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1024,
            "dfactor": 8,
            "force_resize": False,
            "width": 640,
            "height": 480,
        },
    },
    "sold2": {
        "output": "matches-sold2",
        "model": {
            "name": "sold2",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": True,
            "force_resize": True,
            "resize_max": 1024,
            "width": 640,
            "height": 480,
            "dfactor": 8,
        },
    },
    "gluestick": {
        "output": "matches-gluestick",
        "model": {
            "name": "gluestick",
            "use_lines": True,
            "max_keypoints": 1000,
            "max_lines": 300,
            "force_num_keypoints": False,
        },
        "preprocessing": {
            "grayscale": True,
            "force_resize": True,
            "resize_max": 1024,
            "width": 640,
            "height": 480,
            "dfactor": 8,
        },
    },
}


def to_cpts(kpts, ps):
    if ps > 0.0:
        kpts = np.round(np.round((kpts + 0.5) / ps) * ps - 0.5, 2)
    return [tuple(cpt) for cpt in kpts]


def assign_keypoints(
    kpts: np.ndarray,
    other_cpts: Union[List[Tuple], np.ndarray],
    max_error: float,
    update: bool = False,
    ref_bins: Optional[List[Counter]] = None,
    scores: Optional[np.ndarray] = None,
    cell_size: Optional[int] = None,
):
    if not update:
        # Without update this is just a NN search
        if len(other_cpts) == 0 or len(kpts) == 0:
            return np.full(len(kpts), -1)
        dist, kpt_ids = KDTree(np.array(other_cpts)).query(kpts)
        valid = dist <= max_error
        kpt_ids[~valid] = -1
        return kpt_ids
    else:
        ps = cell_size if cell_size is not None else max_error
        ps = max(ps, max_error)
        # With update we quantize and bin (optionally)
        assert isinstance(other_cpts, list)
        kpt_ids = []
        cpts = to_cpts(kpts, ps)
        bpts = to_cpts(kpts, int(max_error))
        cp_to_id = {val: i for i, val in enumerate(other_cpts)}
        for i, (cpt, bpt) in enumerate(zip(cpts, bpts)):
            try:
                kid = cp_to_id[cpt]
            except KeyError:
                kid = len(cp_to_id)
                cp_to_id[cpt] = kid
                other_cpts.append(cpt)
                if ref_bins is not None:
                    ref_bins.append(Counter())
            if ref_bins is not None:
                score = scores[i] if scores is not None else 1
                ref_bins[cp_to_id[cpt]][bpt] += score
            kpt_ids.append(kid)
        return np.array(kpt_ids)


def get_grouped_ids(array):
    # Group array indices based on its values
    # all duplicates are grouped as a set
    idx_sort = np.argsort(array)
    sorted_array = array[idx_sort]
    _, ids, _ = np.unique(sorted_array, return_counts=True, return_index=True)
    res = np.split(idx_sort, ids[1:])
    return res


def get_unique_matches(match_ids, scores):
    if len(match_ids.shape) == 1:
        return [0]

    isets1 = get_grouped_ids(match_ids[:, 0])
    isets2 = get_grouped_ids(match_ids[:, 1])
    uid1s = [ids[scores[ids].argmax()] for ids in isets1 if len(ids) > 0]
    uid2s = [ids[scores[ids].argmax()] for ids in isets2 if len(ids) > 0]
    uids = list(set(uid1s).intersection(uid2s))
    return match_ids[uids], scores[uids]


def matches_to_matches0(matches, scores):
    if len(matches) == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float16)
    n_kps0 = np.max(matches[:, 0]) + 1
    matches0 = -np.ones((n_kps0,))
    scores0 = np.zeros((n_kps0,))
    matches0[matches[:, 0]] = matches[:, 1]
    scores0[matches[:, 0]] = scores
    return matches0.astype(np.int32), scores0.astype(np.float16)


def kpids_to_matches0(kpt_ids0, kpt_ids1, scores):
    valid = (kpt_ids0 != -1) & (kpt_ids1 != -1)
    matches = np.dstack([kpt_ids0[valid], kpt_ids1[valid]])
    matches = matches.reshape(-1, 2)
    scores = scores[valid]

    # Remove n-to-1 matches
    matches, scores = get_unique_matches(matches, scores)
    return matches_to_matches0(matches, scores)


def scale_keypoints(kpts, scale):
    if np.any(scale != 1.0):
        kpts *= kpts.new_tensor(scale)
    return kpts


class ImagePairDataset(torch.utils.data.Dataset):
    default_conf = {
        "grayscale": True,
        "resize_max": 1024,
        "dfactor": 8,
        "cache_images": False,
    }

    def __init__(self, image_dir, conf, pairs):
        self.image_dir = image_dir
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.pairs = pairs
        if self.conf.cache_images:
            image_names = set(sum(pairs, ()))  # unique image names in pairs
            logger.info(f"Loading and caching {len(image_names)} unique images.")
            self.images = {}
            self.scales = {}
            for name in tqdm(image_names):
                image = read_image(self.image_dir / name, self.conf.grayscale)
                self.images[name], self.scales[name] = self.preprocess(image)

    def preprocess(self, image: np.ndarray):
        image = image.astype(np.float32, copy=False)
        size = image.shape[:2][::-1]
        scale = np.array([1.0, 1.0])

        if self.conf.resize_max:
            scale = self.conf.resize_max / max(size)
            if scale < 1.0:
                size_new = tuple(int(round(x * scale)) for x in size)
                image = resize_image(image, size_new, "cv2_area")
                scale = np.array(size) / np.array(size_new)

        if self.conf.grayscale:
            assert image.ndim == 2, image.shape
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = torch.from_numpy(image / 255.0).float()

        # assure that the size is divisible by dfactor
        size_new = tuple(
            map(
                lambda x: int(x // self.conf.dfactor * self.conf.dfactor),
                image.shape[-2:],
            )
        )
        image = F.resize(image, size=size_new)
        scale = np.array(size) / np.array(size_new)[::-1]
        return image, scale

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        if self.conf.cache_images:
            image0, scale0 = self.images[name0], self.scales[name0]
            image1, scale1 = self.images[name1], self.scales[name1]
        else:
            image0 = read_image(self.image_dir / name0, self.conf.grayscale)
            image1 = read_image(self.image_dir / name1, self.conf.grayscale)
            image0, scale0 = self.preprocess(image0)
            image1, scale1 = self.preprocess(image1)
        return image0, image1, scale0, scale1, name0, name1


@torch.no_grad()
def match_dense(
    conf: Dict,
    pairs: List[Tuple[str, str]],
    image_dir: Path,
    match_path: Path,  # out
    existing_refs: Optional[List] = [],
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(matchers, conf["model"]["name"])
    model = Model(conf["model"]).eval().to(device)

    dataset = ImagePairDataset(image_dir, conf["preprocessing"], pairs)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=16, batch_size=1, shuffle=False
    )

    logger.info("Performing dense matching...")
    with h5py.File(str(match_path), "a") as fd:
        for data in tqdm(loader, smoothing=0.1):
            # load image-pair data
            image0, image1, scale0, scale1, (name0,), (name1,) = data
            scale0, scale1 = scale0[0].numpy(), scale1[0].numpy()
            image0, image1 = image0.to(device), image1.to(device)

            # match semi-dense
            # for consistency with pairs_from_*: refine kpts of image0
            if name0 in existing_refs:
                # special case: flip to enable refinement in query image
                pred = model({"image0": image1, "image1": image0})
                pred = {
                    **pred,
                    "keypoints0": pred["keypoints1"],
                    "keypoints1": pred["keypoints0"],
                }
            else:
                # usual case
                pred = model({"image0": image0, "image1": image1})

            # Rescale keypoints and move to cpu
            kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
            kpts0 = scale_keypoints(kpts0 + 0.5, scale0) - 0.5
            kpts1 = scale_keypoints(kpts1 + 0.5, scale1) - 0.5
            kpts0 = kpts0.cpu().numpy()
            kpts1 = kpts1.cpu().numpy()
            scores = pred["scores"].cpu().numpy()

            # Write matches and matching scores in hloc format
            pair = names_to_pair(name0, name1)
            if pair in fd:
                del fd[pair]
            grp = fd.create_group(pair)

            # Write dense matching output
            grp.create_dataset("keypoints0", data=kpts0)
            grp.create_dataset("keypoints1", data=kpts1)
            grp.create_dataset("scores", data=scores)
    del model, loader


# default: quantize all!
def load_keypoints(
    conf: Dict, feature_paths_refs: List[Path], quantize: Optional[set] = None
):
    name2ref = {
        n: i for i, p in enumerate(feature_paths_refs) for n in list_h5_names(p)
    }

    existing_refs = set(name2ref.keys())
    if quantize is None:
        quantize = existing_refs  # quantize all
    if len(existing_refs) > 0:
        logger.info(f"Loading keypoints from {len(existing_refs)} images.")

    # Load query keypoints
    cpdict = defaultdict(list)
    bindict = defaultdict(list)
    for name in existing_refs:
        with h5py.File(str(feature_paths_refs[name2ref[name]]), "r") as fd:
            kps = fd[name]["keypoints"].__array__()
            if name not in quantize:
                cpdict[name] = kps
            else:
                if "scores" in fd[name].keys():
                    kp_scores = fd[name]["scores"].__array__()
                else:
                    # we set the score to 1.0 if not provided
                    # increase for more weight on reference keypoints for
                    # stronger anchoring
                    kp_scores = [1.0 for _ in range(kps.shape[0])]
                # bin existing keypoints of reference images for association
                assign_keypoints(
                    kps,
                    cpdict[name],
                    conf["max_error"],
                    True,
                    bindict[name],
                    kp_scores,
                    conf["cell_size"],
                )
    return cpdict, bindict


def aggregate_matches(
    conf: Dict,
    pairs: List[Tuple[str, str]],
    match_path: Path,
    feature_path: Path,
    required_queries: Optional[Set[str]] = None,
    max_kps: Optional[int] = None,
    cpdict: Dict[str, Iterable] = defaultdict(list),
    bindict: Dict[str, List[Counter]] = defaultdict(list),
):
    if required_queries is None:
        required_queries = set(sum(pairs, ()))
        # default: do not overwrite existing features in feature_path!
        required_queries -= set(list_h5_names(feature_path))

    # if an entry in cpdict is provided as np.ndarray we assume it is fixed
    required_queries -= set([k for k, v in cpdict.items() if isinstance(v, np.ndarray)])

    # sort pairs for reduced RAM
    pairs_per_q = Counter(list(chain(*pairs)))
    pairs_score = [min(pairs_per_q[i], pairs_per_q[j]) for i, j in pairs]
    pairs = [p for _, p in sorted(zip(pairs_score, pairs))]

    if len(required_queries) > 0:
        logger.info(f"Aggregating keypoints for {len(required_queries)} images.")
    n_kps = 0
    with h5py.File(str(match_path), "a") as fd:
        for name0, name1 in tqdm(pairs, smoothing=0.1):
            pair = names_to_pair(name0, name1)
            grp = fd[pair]
            kpts0 = grp["keypoints0"].__array__()
            kpts1 = grp["keypoints1"].__array__()
            scores = grp["scores"].__array__()

            # Aggregate local features
            update0 = name0 in required_queries
            update1 = name1 in required_queries

            # in localization we do not want to bin the query kp
            # assumes that the query is name0!
            if update0 and not update1 and max_kps is None:
                max_error0 = cell_size0 = 0.0
            else:
                max_error0 = conf["max_error"]
                cell_size0 = conf["cell_size"]

            # Get match ids and extend query keypoints (cpdict)
            mkp_ids0 = assign_keypoints(
                kpts0,
                cpdict[name0],
                max_error0,
                update0,
                bindict[name0],
                scores,
                cell_size0,
            )
            mkp_ids1 = assign_keypoints(
                kpts1,
                cpdict[name1],
                conf["max_error"],
                update1,
                bindict[name1],
                scores,
                conf["cell_size"],
            )

            # Build matches from assignments
            matches0, scores0 = kpids_to_matches0(mkp_ids0, mkp_ids1, scores)

            assert kpts0.shape[0] == scores.shape[0]
            grp.create_dataset("matches0", data=matches0)
            grp.create_dataset("matching_scores0", data=scores0)

            # Convert bins to kps if finished, and store them
            for name in (name0, name1):
                pairs_per_q[name] -= 1
                if pairs_per_q[name] > 0 or name not in required_queries:
                    continue
                kp_score = [c.most_common(1)[0][1] for c in bindict[name]]
                cpdict[name] = [c.most_common(1)[0][0] for c in bindict[name]]
                cpdict[name] = np.array(cpdict[name], dtype=np.float32)

                # Select top-k query kps by score (reassign matches later)
                if max_kps:
                    top_k = min(max_kps, cpdict[name].shape[0])
                    top_k = np.argsort(kp_score)[::-1][:top_k]
                    cpdict[name] = cpdict[name][top_k]
                    kp_score = np.array(kp_score)[top_k]

                # Write query keypoints
                with h5py.File(feature_path, "a") as kfd:
                    if name in kfd:
                        del kfd[name]
                    kgrp = kfd.create_group(name)
                    kgrp.create_dataset("keypoints", data=cpdict[name])
                    kgrp.create_dataset("score", data=kp_score)
                    n_kps += cpdict[name].shape[0]
                del bindict[name]

    if len(required_queries) > 0:
        avg_kp_per_image = round(n_kps / len(required_queries), 1)
        logger.info(
            f"Finished assignment, found {avg_kp_per_image} "
            f"keypoints/image (avg.), total {n_kps}."
        )
    return cpdict


def assign_matches(
    pairs: List[Tuple[str, str]],
    match_path: Path,
    keypoints: Union[List[Path], Dict[str, np.array]],
    max_error: float,
):
    if isinstance(keypoints, list):
        keypoints = load_keypoints({}, keypoints, kpts_as_bin=set([]))
    assert len(set(sum(pairs, ())) - set(keypoints.keys())) == 0
    with h5py.File(str(match_path), "a") as fd:
        for name0, name1 in tqdm(pairs):
            pair = names_to_pair(name0, name1)
            grp = fd[pair]
            kpts0 = grp["keypoints0"].__array__()
            kpts1 = grp["keypoints1"].__array__()
            scores = grp["scores"].__array__()

            # NN search across cell boundaries
            mkp_ids0 = assign_keypoints(kpts0, keypoints[name0], max_error)
            mkp_ids1 = assign_keypoints(kpts1, keypoints[name1], max_error)

            matches0, scores0 = kpids_to_matches0(mkp_ids0, mkp_ids1, scores)

            # overwrite matches0 and matching_scores0
            del grp["matches0"], grp["matching_scores0"]
            grp.create_dataset("matches0", data=matches0)
            grp.create_dataset("matching_scores0", data=scores0)


@torch.no_grad()
def match_and_assign(
    conf: Dict,
    pairs_path: Path,
    image_dir: Path,
    match_path: Path,  # out
    feature_path_q: Path,  # out
    feature_paths_refs: Optional[List[Path]] = [],
    max_kps: Optional[int] = 8192,
    overwrite: bool = False,
) -> Path:
    for path in feature_paths_refs:
        if not path.exists():
            raise FileNotFoundError(f"Reference feature file {path}.")
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    required_queries = set(sum(pairs, ()))

    name2ref = {
        n: i for i, p in enumerate(feature_paths_refs) for n in list_h5_names(p)
    }
    existing_refs = required_queries.intersection(set(name2ref.keys()))

    # images which require feature extraction
    required_queries = required_queries - existing_refs

    if feature_path_q.exists():
        existing_queries = set(list_h5_names(feature_path_q))
        feature_paths_refs.append(feature_path_q)
        existing_refs = set.union(existing_refs, existing_queries)
        if not overwrite:
            required_queries = required_queries - existing_queries

    if len(pairs) == 0 and len(required_queries) == 0:
        logger.info("All pairs exist. Skipping dense matching.")
        return

    # extract semi-dense matches
    match_dense(conf, pairs, image_dir, match_path, existing_refs=existing_refs)

    logger.info("Assigning matches...")

    # Pre-load existing keypoints
    cpdict, bindict = load_keypoints(
        conf, feature_paths_refs, quantize=required_queries
    )

    # Reassign matches by aggregation
    cpdict = aggregate_matches(
        conf,
        pairs,
        match_path,
        feature_path=feature_path_q,
        required_queries=required_queries,
        max_kps=max_kps,
        cpdict=cpdict,
        bindict=bindict,
    )

    # Invalidate matches that are far from selected bin by reassignment
    if max_kps is not None:
        logger.info(f'Reassign matches with max_error={conf["max_error"]}.')
        assign_matches(pairs, match_path, cpdict, max_error=conf["max_error"])


def scale_lines(lines, scale):
    if np.any(scale != 1.0):
        lines *= lines.new_tensor(scale)
    return lines


def match(model, path_0, path_1, conf):
    default_conf = {
        "grayscale": True,
        "resize_max": 1024,
        "dfactor": 8,
        "cache_images": False,
        "force_resize": False,
        "width": 320,
        "height": 240,
    }

    def preprocess(image: np.ndarray):
        image = image.astype(np.float32, copy=False)
        size = image.shape[:2][::-1]
        scale = np.array([1.0, 1.0])
        if conf.resize_max:
            scale = conf.resize_max / max(size)
            if scale < 1.0:
                size_new = tuple(int(round(x * scale)) for x in size)
                image = resize_image(image, size_new, "cv2_area")
                scale = np.array(size) / np.array(size_new)
        if conf.force_resize:
            size = image.shape[:2][::-1]
            image = resize_image(image, (conf.width, conf.height), "cv2_area")
            size_new = (conf.width, conf.height)
            scale = np.array(size) / np.array(size_new)
        if conf.grayscale:
            assert image.ndim == 2, image.shape
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = torch.from_numpy(image / 255.0).float()
        # assure that the size is divisible by dfactor
        size_new = tuple(
            map(
                lambda x: int(x // conf.dfactor * conf.dfactor),
                image.shape[-2:],
            )
        )
        image = F.resize(image, size=size_new, antialias=True)
        scale = np.array(size) / np.array(size_new)[::-1]
        return image, scale

    conf = SimpleNamespace(**{**default_conf, **conf})
    image0 = read_image(path_0, conf.grayscale)
    image1 = read_image(path_1, conf.grayscale)
    image0, scale0 = preprocess(image0)
    image1, scale1 = preprocess(image1)
    image0 = image0.to(device)[None]
    image1 = image1.to(device)[None]
    pred = model({"image0": image0, "image1": image1})

    # Rescale keypoints and move to cpu
    kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
    kpts0 = scale_keypoints(kpts0 + 0.5, scale0) - 0.5
    kpts1 = scale_keypoints(kpts1 + 0.5, scale1) - 0.5

    ret = {
        "image0": image0.squeeze().cpu().numpy(),
        "image1": image1.squeeze().cpu().numpy(),
        "keypoints0": kpts0.cpu().numpy(),
        "keypoints1": kpts1.cpu().numpy(),
    }
    if "mconf" in pred.keys():
        ret["mconf"] = pred["mconf"].cpu().numpy()
    return ret


@torch.no_grad()
def match_images(model, image_0, image_1, conf, device="cpu"):
    default_conf = {
        "grayscale": True,
        "resize_max": 1024,
        "dfactor": 8,
        "cache_images": False,
        "force_resize": False,
        "width": 320,
        "height": 240,
    }

    def preprocess(image: np.ndarray):
        image = image.astype(np.float32, copy=False)
        size = image.shape[:2][::-1]
        scale = np.array([1.0, 1.0])
        if conf.resize_max:
            scale = conf.resize_max / max(size)
            if scale < 1.0:
                size_new = tuple(int(round(x * scale)) for x in size)
                image = resize_image(image, size_new, "cv2_area")
                scale = np.array(size) / np.array(size_new)
        if conf.force_resize:
            size = image.shape[:2][::-1]
            image = resize_image(image, (conf.width, conf.height), "cv2_area")
            size_new = (conf.width, conf.height)
            scale = np.array(size) / np.array(size_new)
        if conf.grayscale:
            assert image.ndim == 2, image.shape
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = torch.from_numpy(image / 255.0).float()

        # assure that the size is divisible by dfactor
        size_new = tuple(
            map(
                lambda x: int(x // conf.dfactor * conf.dfactor),
                image.shape[-2:],
            )
        )
        image = F.resize(image, size=size_new)
        scale = np.array(size) / np.array(size_new)[::-1]
        return image, scale

    conf = SimpleNamespace(**{**default_conf, **conf})

    if len(image_0.shape) == 3 and conf.grayscale:
        image0 = cv2.cvtColor(image_0, cv2.COLOR_RGB2GRAY)
    else:
        image0 = image_0
    if len(image_0.shape) == 3 and conf.grayscale:
        image1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
    else:
        image1 = image_1

    # comment following lines, image is always RGB mode
    # if not conf.grayscale and len(image0.shape) == 3:
    #     image0 = image0[:, :, ::-1]  # BGR to RGB
    # if not conf.grayscale and len(image1.shape) == 3:
    #     image1 = image1[:, :, ::-1]  # BGR to RGB

    image0, scale0 = preprocess(image0)
    image1, scale1 = preprocess(image1)
    image0 = image0.to(device)[None]
    image1 = image1.to(device)[None]
    pred = model({"image0": image0, "image1": image1})

    s0 = np.array(image_0.shape[:2][::-1]) / np.array(image0.shape[-2:][::-1])
    s1 = np.array(image_1.shape[:2][::-1]) / np.array(image1.shape[-2:][::-1])

    # Rescale keypoints and move to cpu
    if "keypoints0" in pred.keys() and "keypoints1" in pred.keys():
        kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
        kpts0_origin = scale_keypoints(kpts0 + 0.5, s0) - 0.5
        kpts1_origin = scale_keypoints(kpts1 + 0.5, s1) - 0.5

        ret = {
            "image0": image0.squeeze().cpu().numpy(),
            "image1": image1.squeeze().cpu().numpy(),
            "image0_orig": image_0,
            "image1_orig": image_1,
            "keypoints0": kpts0.cpu().numpy(),
            "keypoints1": kpts1.cpu().numpy(),
            "keypoints0_orig": kpts0_origin.cpu().numpy(),
            "keypoints1_orig": kpts1_origin.cpu().numpy(),
            "mkeypoints0": kpts0.cpu().numpy(),
            "mkeypoints1": kpts1.cpu().numpy(),
            "mkeypoints0_orig": kpts0_origin.cpu().numpy(),
            "mkeypoints1_orig": kpts1_origin.cpu().numpy(),
            "original_size0": np.array(image_0.shape[:2][::-1]),
            "original_size1": np.array(image_1.shape[:2][::-1]),
            "new_size0": np.array(image0.shape[-2:][::-1]),
            "new_size1": np.array(image1.shape[-2:][::-1]),
            "scale0": s0,
            "scale1": s1,
        }
        if "mconf" in pred.keys():
            ret["mconf"] = pred["mconf"].cpu().numpy()
        elif "scores" in pred.keys():  # adapting loftr
            ret["mconf"] = pred["scores"].cpu().numpy()
        else:
            ret["mconf"] = np.ones_like(kpts0.cpu().numpy()[:, 0])
    if "lines0" in pred.keys() and "lines1" in pred.keys():
        if "keypoints0" in pred.keys() and "keypoints1" in pred.keys():
            kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
            kpts0_origin = scale_keypoints(kpts0 + 0.5, s0) - 0.5
            kpts1_origin = scale_keypoints(kpts1 + 0.5, s1) - 0.5
            kpts0_origin = kpts0_origin.cpu().numpy()
            kpts1_origin = kpts1_origin.cpu().numpy()
        else:
            kpts0_origin, kpts1_origin = (
                None,
                None,
            )  # np.zeros([0]), np.zeros([0])
        lines0, lines1 = pred["lines0"], pred["lines1"]
        lines0_raw, lines1_raw = pred["raw_lines0"], pred["raw_lines1"]

        lines0_raw = torch.from_numpy(lines0_raw.copy())
        lines1_raw = torch.from_numpy(lines1_raw.copy())
        lines0_raw = scale_lines(lines0_raw + 0.5, s0) - 0.5
        lines1_raw = scale_lines(lines1_raw + 0.5, s1) - 0.5

        lines0 = torch.from_numpy(lines0.copy())
        lines1 = torch.from_numpy(lines1.copy())
        lines0 = scale_lines(lines0 + 0.5, s0) - 0.5
        lines1 = scale_lines(lines1 + 0.5, s1) - 0.5

        ret = {
            "image0_orig": image_0,
            "image1_orig": image_1,
            "line0": lines0_raw.cpu().numpy(),
            "line1": lines1_raw.cpu().numpy(),
            "line0_orig": lines0.cpu().numpy(),
            "line1_orig": lines1.cpu().numpy(),
            "line_keypoints0_orig": kpts0_origin,
            "line_keypoints1_orig": kpts1_origin,
        }
    del pred
    torch.cuda.empty_cache()
    return ret


@torch.no_grad()
def main(
    conf: Dict,
    pairs: Path,
    image_dir: Path,
    export_dir: Optional[Path] = None,
    matches: Optional[Path] = None,  # out
    features: Optional[Path] = None,  # out
    features_ref: Optional[Path] = None,
    max_kps: Optional[int] = 8192,
    overwrite: bool = False,
) -> Path:
    logger.info(
        "Extracting semi-dense features with configuration:" f"\n{pprint.pformat(conf)}"
    )

    if features is None:
        features = "feats_"

    if isinstance(features, Path):
        features_q = features
        if matches is None:
            raise ValueError(
                "Either provide both features and matches as Path" " or both as names."
            )
    else:
        if export_dir is None:
            raise ValueError(
                "Provide an export_dir if features and matches"
                f" are not file paths: {features}, {matches}."
            )
        features_q = Path(export_dir, f'{features}{conf["output"]}.h5')
        if matches is None:
            matches = Path(export_dir, f'{conf["output"]}_{pairs.stem}.h5')

    if features_ref is None:
        features_ref = []
    elif isinstance(features_ref, list):
        features_ref = list(features_ref)
    elif isinstance(features_ref, Path):
        features_ref = [features_ref]
    else:
        raise TypeError(str(features_ref))

    match_and_assign(
        conf,
        pairs,
        image_dir,
        matches,
        features_q,
        features_ref,
        max_kps,
        overwrite,
    )

    return features_q, matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--export_dir", type=Path, required=True)
    parser.add_argument("--matches", type=Path, default=confs["loftr"]["output"])
    parser.add_argument(
        "--features", type=str, default="feats_" + confs["loftr"]["output"]
    )
    parser.add_argument("--conf", type=str, default="loftr", choices=list(confs.keys()))
    args = parser.parse_args()
    main(
        confs[args.conf],
        args.pairs,
        args.image_dir,
        args.export_dir,
        args.matches,
        args.features,
    )
