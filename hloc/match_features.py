import argparse
import pprint
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from tqdm import tqdm

from . import logger, matchers
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval

"""
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the match file that will be generated.
    - model: the model configuration, as passed to a feature matcher.
"""
confs = {
    "superglue": {
        "output": "matches-superglue",
        "model": {
            "name": "superglue",
            "weights": "outdoor",
            "sinkhorn_iterations": 50,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 8,
            "force_resize": False,
        },
    },
    "superglue-fast": {
        "output": "matches-superglue-it5",
        "model": {
            "name": "superglue",
            "weights": "outdoor",
            "sinkhorn_iterations": 5,
            "match_threshold": 0.2,
        },
    },
    "superpoint-lightglue": {
        "output": "matches-lightglue",
        "model": {
            "name": "lightglue",
            "match_threshold": 0.2,
            "width_confidence": 0.99,  # for point pruning
            "depth_confidence": 0.95,  # for early stopping,
            "features": "superpoint",
            "model_name": "superpoint_lightglue.pth",
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 8,
            "force_resize": False,
        },
    },
    "disk-lightglue": {
        "output": "matches-disk-lightglue",
        "model": {
            "name": "lightglue",
            "match_threshold": 0.2,
            "width_confidence": 0.99,  # for point pruning
            "depth_confidence": 0.95,  # for early stopping,
            "features": "disk",
            "model_name": "disk_lightglue.pth",
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 8,
            "force_resize": False,
        },
    },
    "dedode-lightglue": {
        "output": "matches-dedode-lightglue",
        "model": {
            "name": "lightglue",
            "match_threshold": 0.2,
            "width_confidence": 0.99,  # for point pruning
            "depth_confidence": 0.95,  # for early stopping,
            "features": "dedodeg",
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 8,
            "force_resize": False,
        },
    },
    "dedodev2-lightglue": {
        "output": "matches-dedode-lightglue",
        "model": {
            "name": "lightglue",
            "match_threshold": 0.2,
            "width_confidence": 0.99,  # for point pruning
            "depth_confidence": 0.95,  # for early stopping,
            "features": "dedodeg-v2",
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 8,
            "force_resize": False,
        },
    },
    "aliked-lightglue": {
        "output": "matches-aliked-lightglue",
        "model": {
            "name": "lightglue",
            "match_threshold": 0.2,
            "width_confidence": 0.99,  # for point pruning
            "depth_confidence": 0.95,  # for early stopping,
            "features": "aliked",
            "model_name": "aliked_lightglue.pth",
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 8,
            "force_resize": False,
        },
    },
    "sift-lightglue": {
        "output": "matches-sift-lightglue",
        "model": {
            "name": "lightglue",
            "match_threshold": 0.2,
            "width_confidence": 0.99,  # for point pruning
            "depth_confidence": 0.95,  # for early stopping,
            "features": "sift",
            "add_scale_ori": True,
            "model_name": "sift_lightglue.pth",
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 8,
            "force_resize": False,
        },
    },
    "sgmnet": {
        "output": "matches-sgmnet",
        "model": {
            "name": "sgmnet",
            "seed_top_k": [256, 256],
            "seed_radius_coe": 0.01,
            "net_channels": 128,
            "layer_num": 9,
            "head": 4,
            "seedlayer": [0, 6],
            "use_mc_seeding": True,
            "use_score_encoding": False,
            "conf_bar": [1.11, 0.1],
            "sink_iter": [10, 100],
            "detach_iter": 1000000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
            "dfactor": 8,
            "force_resize": False,
        },
    },
    "NN-superpoint": {
        "output": "matches-NN-mutual-dist.7",
        "model": {
            "name": "nearest_neighbor",
            "do_mutual_check": True,
            "distance_threshold": 0.7,
            "match_threshold": 0.2,
        },
    },
    "NN-ratio": {
        "output": "matches-NN-mutual-ratio.8",
        "model": {
            "name": "nearest_neighbor",
            "do_mutual_check": True,
            "ratio_threshold": 0.8,
            "match_threshold": 0.2,
        },
    },
    "NN-mutual": {
        "output": "matches-NN-mutual",
        "model": {
            "name": "nearest_neighbor",
            "do_mutual_check": True,
            "match_threshold": 0.2,
        },
    },
    "Dual-Softmax": {
        "output": "matches-Dual-Softmax",
        "model": {
            "name": "dual_softmax",
            "match_threshold": 0.01,
            "inv_temperature": 20,
        },
    },
    "adalam": {
        "output": "matches-adalam",
        "model": {
            "name": "adalam",
            "match_threshold": 0.2,
        },
    },
    "imp": {
        "output": "matches-imp",
        "model": {
            "name": "imp",
            "match_threshold": 0.2,
        },
    },
}


class WorkQueue:
    def __init__(self, work_fn, num_threads=1):
        self.queue = Queue(num_threads)
        self.threads = [
            Thread(target=self.thread_fn, args=(work_fn,))
            for _ in range(num_threads)
        ]
        for thread in self.threads:
            thread.start()

    def join(self):
        for thread in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()

    def thread_fn(self, work_fn):
        item = self.queue.get()
        while item is not None:
            work_fn(item)
            item = self.queue.get()

    def put(self, data):
        self.queue.put(data)


class FeaturePairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, feature_path_q, feature_path_r):
        self.pairs = pairs
        self.feature_path_q = feature_path_q
        self.feature_path_r = feature_path_r

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        data = {}
        with h5py.File(self.feature_path_q, "r") as fd:
            grp = fd[name0]
            for k, v in grp.items():
                data[k + "0"] = torch.from_numpy(v.__array__()).float()
            # some matchers might expect an image but only use its size
            data["image0"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])
        with h5py.File(self.feature_path_r, "r") as fd:
            grp = fd[name1]
            for k, v in grp.items():
                data[k + "1"] = torch.from_numpy(v.__array__()).float()
            data["image1"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])
        return data

    def __len__(self):
        return len(self.pairs)


def writer_fn(inp, match_path):
    pair, pred = inp
    with h5py.File(str(match_path), "a", libver="latest") as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        matches = pred["matches0"][0].cpu().short().numpy()
        grp.create_dataset("matches0", data=matches)
        if "matching_scores0" in pred:
            scores = pred["matching_scores0"][0].cpu().half().numpy()
            grp.create_dataset("matching_scores0", data=scores)


def main(
    conf: Dict,
    pairs: Path,
    features: Union[Path, str],
    export_dir: Optional[Path] = None,
    matches: Optional[Path] = None,
    features_ref: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    if isinstance(features, Path) or Path(features).exists():
        features_q = features
        if matches is None:
            raise ValueError(
                "Either provide both features and matches as Path"
                " or both as names."
            )
    else:
        if export_dir is None:
            raise ValueError(
                "Provide an export_dir if features is not"
                f" a file path: {features}."
            )
        features_q = Path(export_dir, features + ".h5")
        if matches is None:
            matches = Path(
                export_dir, f'{features}_{conf["output"]}_{pairs.stem}.h5'
            )

    if features_ref is None:
        features_ref = features_q
    match_from_paths(conf, pairs, matches, features_q, features_ref, overwrite)

    return matches


def find_unique_new_pairs(pairs_all: List[Tuple[str]], match_path: Path = None):
    """Avoid to recompute duplicates to save time."""
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = list(pairs)
    if match_path is not None and match_path.exists():
        with h5py.File(str(match_path), "r", libver="latest") as fd:
            pairs_filtered = []
            for i, j in pairs:
                if (
                    names_to_pair(i, j) in fd
                    or names_to_pair(j, i) in fd
                    or names_to_pair_old(i, j) in fd
                    or names_to_pair_old(j, i) in fd
                ):
                    continue
                pairs_filtered.append((i, j))
        return pairs_filtered
    return pairs


@torch.no_grad()
def match_from_paths(
    conf: Dict,
    pairs_path: Path,
    match_path: Path,
    feature_path_q: Path,
    feature_path_ref: Path,
    overwrite: bool = False,
) -> Path:
    logger.info(
        "Matching local features with configuration:"
        f"\n{pprint.pformat(conf)}"
    )

    if not feature_path_q.exists():
        raise FileNotFoundError(f"Query feature file {feature_path_q}.")
    if not feature_path_ref.exists():
        raise FileNotFoundError(f"Reference feature file {feature_path_ref}.")
    match_path.parent.mkdir(exist_ok=True, parents=True)

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        logger.info("Skipping the matching.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(matchers, conf["model"]["name"])
    model = Model(conf["model"]).eval().to(device)

    dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=5, batch_size=1, shuffle=False, pin_memory=True
    )
    writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)

    for idx, data in enumerate(tqdm(loader, smoothing=0.1)):
        data = {
            k: v if k.startswith("image") else v.to(device, non_blocking=True)
            for k, v in data.items()
        }
        pred = model(data)
        pair = names_to_pair(*pairs[idx])
        writer_queue.put((pair, pred))
    writer_queue.join()
    logger.info("Finished exporting matches.")


def scale_keypoints(kpts, scale):
    if (
        isinstance(scale, (list, tuple, np.ndarray))
        and len(scale) == 2
        and np.any(scale != np.array([1.0, 1.0]))
    ):
        if isinstance(kpts, torch.Tensor):
            kpts[:, 0] *= scale[0]  # scale x-dimension
            kpts[:, 1] *= scale[1]  # scale y-dimension
        elif isinstance(kpts, np.ndarray):
            kpts[:, 0] *= scale[0]  # scale x-dimension
            kpts[:, 1] *= scale[1]  # scale y-dimension
    return kpts


@torch.no_grad()
def match_images(model, feat0, feat1):
    # forward pass to match keypoints
    desc0 = feat0["descriptors"][0]
    desc1 = feat1["descriptors"][0]
    if len(desc0.shape) == 2:
        desc0 = desc0.unsqueeze(0)
    if len(desc1.shape) == 2:
        desc1 = desc1.unsqueeze(0)
    if isinstance(feat0["keypoints"], list):
        feat0["keypoints"] = feat0["keypoints"][0][None]
    if isinstance(feat1["keypoints"], list):
        feat1["keypoints"] = feat1["keypoints"][0][None]
    input_dict = {
        "image0": feat0["image"],
        "keypoints0": feat0["keypoints"],
        "scores0": feat0["scores"][0].unsqueeze(0),
        "descriptors0": desc0,
        "image1": feat1["image"],
        "keypoints1": feat1["keypoints"],
        "scores1": feat1["scores"][0].unsqueeze(0),
        "descriptors1": desc1,
    }
    if "scales" in feat0:
        input_dict = {**input_dict, "scales0": feat0["scales"]}
    if "scales" in feat1:
        input_dict = {**input_dict, "scales1": feat1["scales"]}
    if "oris" in feat0:
        input_dict = {**input_dict, "oris0": feat0["oris"]}
    if "oris" in feat1:
        input_dict = {**input_dict, "oris1": feat1["oris"]}
    pred = model(input_dict)
    pred = {
        k: v.cpu().detach()[0] if isinstance(v, torch.Tensor) else v
        for k, v in pred.items()
    }
    kpts0, kpts1 = (
        feat0["keypoints"][0].cpu().numpy(),
        feat1["keypoints"][0].cpu().numpy(),
    )
    matches, confid = pred["matches0"], pred["matching_scores0"]
    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconfid = confid[valid]
    # rescale the keypoints to their original size
    s0 = feat0["original_size"] / feat0["size"]
    s1 = feat1["original_size"] / feat1["size"]
    kpts0_origin = scale_keypoints(torch.from_numpy(kpts0 + 0.5), s0) - 0.5
    kpts1_origin = scale_keypoints(torch.from_numpy(kpts1 + 0.5), s1) - 0.5

    mkpts0_origin = scale_keypoints(torch.from_numpy(mkpts0 + 0.5), s0) - 0.5
    mkpts1_origin = scale_keypoints(torch.from_numpy(mkpts1 + 0.5), s1) - 0.5

    ret = {
        "image0_orig": feat0["image_orig"],
        "image1_orig": feat1["image_orig"],
        "keypoints0": kpts0,
        "keypoints1": kpts1,
        "keypoints0_orig": kpts0_origin.numpy(),
        "keypoints1_orig": kpts1_origin.numpy(),
        "mkeypoints0": mkpts0,
        "mkeypoints1": mkpts1,
        "mkeypoints0_orig": mkpts0_origin.numpy(),
        "mkeypoints1_orig": mkpts1_origin.numpy(),
        "mconf": mconfid.numpy(),
    }
    del feat0, feat1, desc0, desc1, kpts0, kpts1, kpts0_origin, kpts1_origin
    torch.cuda.empty_cache()

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--export_dir", type=Path)
    parser.add_argument(
        "--features", type=str, default="feats-superpoint-n4096-r1024"
    )
    parser.add_argument("--matches", type=Path)
    parser.add_argument(
        "--conf", type=str, default="superglue", choices=list(confs.keys())
    )
    args = parser.parse_args()
    main(confs[args.conf], args.pairs, args.features, args.export_dir)
