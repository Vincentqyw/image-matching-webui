import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms

from hloc import MODEL_REPO_ID, logger

from ..utils.base_model import BaseModel

dedode_path = Path(__file__).parent / "../../third_party/DeDoDe"
sys.path.append(str(dedode_path))

from DeDoDe import dedode_descriptor_B, dedode_detector_L
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

    # Initialize the line matcher
    def _init(self, conf):
        model_detector_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, conf["model_detector_name"]),
        )
        model_descriptor_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, conf["model_descriptor_name"]),
        )
        logger.info("Loaded DarkFeat model: {}".format(model_detector_path))
        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # load the model
        weights_detector = torch.load(model_detector_path, map_location="cpu")
        weights_descriptor = torch.load(model_descriptor_path, map_location="cpu")
        self.detector = dedode_detector_L(weights=weights_detector, device=device)
        self.descriptor = dedode_descriptor_B(weights=weights_descriptor, device=device)
        logger.info("Load DeDoDe model done.")

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
        description_A = self.descriptor.describe_keypoints(batch_A, keypoints_A)[
            "descriptions"
        ]
        keypoints_A = to_pixel_coords(keypoints_A, H_A, W_A)

        return {
            "keypoints": keypoints_A,  # 1 x N x 2
            "descriptors": description_A.permute(0, 2, 1),  # 1 x 256 x N
            "scores": P_A,  # 1 x N
        }
