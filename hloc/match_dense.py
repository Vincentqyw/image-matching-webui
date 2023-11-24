import numpy as np
import torch
import torchvision.transforms.functional as F
from types import SimpleNamespace
from .extract_features import read_image, resize_image
import cv2

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
    "loftr-quadtree": {
        "output": "matches-loftr-quadtree",
        "model": {
            "name": "quadtree",
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
        "preprocessing": {"grayscale": True, "resize_max": 1024, "dfactor": 8},
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
    # Use topicfm for matching feats
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
    "dedode_sparse": {
        "output": "matches-dedode",
        "model": {
            "name": "dedode",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
            "dense": False,
        },
        "preprocessing": {
            "grayscale": False,
            "force_resize": True,
            "resize_max": 1024,
            "width": 768,
            "height": 768,
            "dfactor": 8,
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


def scale_keypoints(kpts, scale):
    if np.any(scale != 1.0):
        kpts *= kpts.new_tensor(scale)
    return kpts


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
            "keypoints0": kpts0_origin.cpu().numpy(),
            "keypoints1": kpts1_origin.cpu().numpy(),
            "keypoints0_orig": kpts0_origin.cpu().numpy(),
            "keypoints1_orig": kpts1_origin.cpu().numpy(),
            "original_size0": np.array(image_0.shape[:2][::-1]),
            "original_size1": np.array(image_1.shape[:2][::-1]),
            "new_size0": np.array(image0.shape[-2:][::-1]),
            "new_size1": np.array(image1.shape[-2:][::-1]),
            "scale0": s0,
            "scale1": s1,
        }
        if "mconf" in pred.keys():
            ret["mconf"] = pred["mconf"].cpu().numpy()
        elif "scores" in pred.keys(): #adapting loftr
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
