import argparse
import collections.abc as collections
import pprint
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Union

import cv2
import h5py
import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm

from . import extractors, logger
from .utils.base_model import dynamic_load
from .utils.io import list_h5_names, read_image
from .utils.parsers import parse_image_lists

from .configs import confs_dict

confs = confs_dict["extractors"]


def resize_image(image, size, interp):
    if interp.startswith("cv2_"):
        interp = getattr(cv2, "INTER_" + interp[len("cv2_") :].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith("pil_"):
        interp = getattr(PIL.Image, interp[len("pil_") :].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(f"Unknown interpolation {interp}.")
    return resized


class ImageDataset(torch.utils.data.Dataset):
    default_conf = {
        "globs": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
        "grayscale": False,
        "resize_max": None,
        "force_resize": False,
        "interpolation": "cv2_area",  # pil_linear is more accurate but slower
    }

    def __init__(self, root, conf, paths=None):
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.root = root

        if paths is None:
            paths = []
            for g in conf.globs:
                paths += list(Path(root).glob("**/" + g))
            if len(paths) == 0:
                raise ValueError(f"Could not find any image in root: {root}.")
            paths = sorted(list(set(paths)))
            self.names = [i.relative_to(root).as_posix() for i in paths]
            logger.info(f"Found {len(self.names)} images in root {root}.")
        else:
            if isinstance(paths, (Path, str)):
                self.names = parse_image_lists(paths)
            elif isinstance(paths, collections.Iterable):
                self.names = [p.as_posix() if isinstance(p, Path) else p for p in paths]
            else:
                raise ValueError(f"Unknown format for path argument {paths}.")

            for name in self.names:
                if not (root / name).exists():
                    raise ValueError(f"Image {name} does not exists in root: {root}.")

    def __getitem__(self, idx):
        name = self.names[idx]
        image = read_image(self.root / name, self.conf.grayscale)
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]

        if self.conf.resize_max and (
            self.conf.force_resize or max(size) > self.conf.resize_max
        ):
            scale = self.conf.resize_max / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            image = resize_image(image, size_new, self.conf.interpolation)

        if self.conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.0

        data = {
            "image": image,
            "original_size": np.array(size),
        }
        return data

    def __len__(self):
        return len(self.names)


def extract(model, image_0, conf):
    default_conf = {
        "grayscale": True,
        "resize_max": 1024,
        "dfactor": 8,
        "cache_images": False,
        "force_resize": False,
        "width": 320,
        "height": 240,
        "interpolation": "cv2_area",
    }
    conf = SimpleNamespace(**{**default_conf, **conf})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def preprocess(image: np.ndarray, conf: SimpleNamespace):
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
        input_ = image.to(device, non_blocking=True)[None]
        data = {
            "image": input_,
            "image_orig": image_0,
            "original_size": np.array(size),
            "size": np.array(image.shape[1:][::-1]),
        }
        return data

    # convert to grayscale if needed
    if len(image_0.shape) == 3 and conf.grayscale:
        image0 = cv2.cvtColor(image_0, cv2.COLOR_RGB2GRAY)
    else:
        image0 = image_0
    # comment following lines, image is always RGB mode
    # if not conf.grayscale and len(image_0.shape) == 3:
    #     image0 = image_0[:, :, ::-1]  # BGR to RGB
    data = preprocess(image0, conf)
    pred = model({"image": data["image"]})
    pred["image_size"] = data["original_size"]
    pred = {**pred, **data}
    return pred


@torch.no_grad()
def main(
    conf: Dict,
    image_dir: Path,
    export_dir: Optional[Path] = None,
    as_half: bool = True,
    image_list: Optional[Union[Path, List[str]]] = None,
    feature_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    logger.info(
        f"Extracting local features with configuration:\n{pprint.pformat(conf)}"
    )

    dataset = ImageDataset(image_dir, conf["preprocessing"], image_list)
    if feature_path is None:
        feature_path = Path(export_dir, conf["output"] + ".h5")
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    skip_names = set(
        list_h5_names(feature_path) if feature_path.exists() and not overwrite else ()
    )
    dataset.names = [n for n in dataset.names if n not in skip_names]
    if len(dataset.names) == 0:
        logger.info("Skipping the extraction.")
        return feature_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(extractors, conf["model"]["name"])
    model = Model(conf["model"]).eval().to(device)

    loader = torch.utils.data.DataLoader(
        dataset, num_workers=1, shuffle=False, pin_memory=True
    )
    for idx, data in enumerate(tqdm(loader)):
        name = dataset.names[idx]
        pred = model({"image": data["image"].to(device, non_blocking=True)})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        pred["image_size"] = original_size = data["original_size"][0].numpy()
        if "keypoints" in pred:
            size = np.array(data["image"].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred["keypoints"] = (pred["keypoints"] + 0.5) * scales[None] - 0.5
            if "scales" in pred:
                pred["scales"] *= scales.mean()
            # add keypoint uncertainties scaled to the original resolution
            uncertainty = getattr(model, "detection_noise", 1) * scales.mean()

        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)

        with h5py.File(str(feature_path), "a", libver="latest") as fd:
            try:
                if name in fd:
                    del fd[name]
                grp = fd.create_group(name)
                for k, v in pred.items():
                    grp.create_dataset(k, data=v)
                if "keypoints" in pred:
                    grp["keypoints"].attrs["uncertainty"] = uncertainty
            except OSError as error:
                if "No space left on device" in error.args[0]:
                    logger.error(
                        "Out of disk space: storing features on disk can take "
                        "significant space, did you enable the as_half flag?"
                    )
                    del grp, fd[name]
                raise error

        del pred

    logger.info("Finished exporting features.")
    return feature_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--export_dir", type=Path, required=True)
    parser.add_argument(
        "--conf",
        type=str,
        default="superpoint_aachen",
        choices=list(confs.keys()),
    )
    parser.add_argument("--as_half", action="store_true")
    parser.add_argument("--image_list", type=Path)
    parser.add_argument("--feature_path", type=Path)
    args = parser.parse_args()
    main(confs[args.conf], args.image_dir, args.export_dir, args.as_half)
