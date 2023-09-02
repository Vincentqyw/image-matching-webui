import sys
from pathlib import Path
import subprocess
import torch
from PIL import Image
from ..utils.base_model import BaseModel
from .. import logger
import torchvision.transforms as transforms

dedode_path = Path(__file__).parent / "../../third_party/DeDoDe"
sys.path.append(str(dedode_path))

from DeDoDe import dedode_detector_L, dedode_descriptor_B
from DeDoDe.utils import to_pixel_coords

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeDoDe(BaseModel):
    default_conf = {
        "name": "dedode",
        "model_detector_name": "dedode_detector_L.pth",
        "model_descriptor_name": "dedode_descriptor_B.pth",
        "max_keypoints": 2000,
        "match_threshold": 0.2,
        "dense": False,  # Now fixed to be false
    }
    required_inputs = [
        "image",
    ]
    weight_urls = {
        "dedode_detector_L.pth": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_detector_L.pth",
        "dedode_descriptor_B.pth": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_B.pth",
    }

    # Initialize the line matcher
    def _init(self, conf):
        model_detector_path = (
            dedode_path / "pretrained" / conf["model_detector_name"]
        )
        model_descriptor_path = (
            dedode_path / "pretrained" / conf["model_descriptor_name"]
        )

        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        # Download the model.
        if not model_detector_path.exists():
            model_detector_path.parent.mkdir(exist_ok=True)
            link = self.weight_urls[conf["model_detector_name"]]
            cmd = ["wget", link, "-O", str(model_detector_path)]
            logger.info(f"Downloading the DeDoDe detector model with `{cmd}`.")
            subprocess.run(cmd, check=True)

        if not model_descriptor_path.exists():
            model_descriptor_path.parent.mkdir(exist_ok=True)
            link = self.weight_urls[conf["model_descriptor_name"]]
            cmd = ["wget", link, "-O", str(model_descriptor_path)]
            logger.info(
                f"Downloading the DeDoDe descriptor model with `{cmd}`."
            )
            subprocess.run(cmd, check=True)

        logger.info(f"Loading DeDoDe model...")

        # load the model
        weights_detector = torch.load(model_detector_path, map_location="cpu")
        weights_descriptor = torch.load(
            model_descriptor_path, map_location="cpu"
        )
        self.detector = dedode_detector_L(
            weights=weights_detector, device=device
        )
        self.descriptor = dedode_descriptor_B(
            weights=weights_descriptor, device=device
        )
        logger.info(f"Load DeDoDe model done.")

    def _forward(self, data):
        """
        data: dict, keys: {'image0','image1'}
        image shape: N x C x H x W
        color mode: RGB
        """
        img0 = self.normalizer(data["image"].squeeze()).float()[None]
        H_A, W_A = img0.shape[2:]

        # step 1: detect keypoints
        detections_A = None
        batch_A = {"image": img0}
        if self.conf["dense"]:
            detections_A = self.detector.detect_dense(batch_A)
        else:
            detections_A = self.detector.detect(
                batch_A, num_keypoints=self.conf["max_keypoints"]
            )
        keypoints_A, P_A = detections_A["keypoints"], detections_A["confidence"]

        # step 2: describe keypoints
        # dim: 1 x N x 256
        description_A = self.descriptor.describe_keypoints(
            batch_A, keypoints_A
        )["descriptions"]
        keypoints_A = to_pixel_coords(keypoints_A, H_A, W_A)

        return {
            "keypoints": keypoints_A,  # 1 x N x 2
            "descriptors": description_A.permute(0, 2, 1),  # 1 x 256 x N
            "scores": P_A,  # 1 x N
        }
