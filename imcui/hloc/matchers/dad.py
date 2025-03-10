import sys
from pathlib import Path

import torch
from PIL import Image

from .. import MODEL_REPO_ID, logger
from ..utils.base_model import BaseModel

roma_path = Path(__file__).parent / "../../third_party/RoMa"
sys.path.append(str(roma_path))
from romatch.models.model_zoo import roma_model

dad_path = Path(__file__).parent / "../../third_party/dad"
sys.path.append(str(dad_path))
import dad as dad_detector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dad(BaseModel):
    default_conf = {
        "name": "two_view_pipeline",
        "model_name": "roma_outdoor.pth",
        "model_utils_name": "dinov2_vitl14_pretrain.pth",
        "max_keypoints": 3000,
    }
    required_inputs = [
        "image0",
        "image1",
    ]

    # Initialize the line matcher
    def _init(self, conf):
        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, self.conf["model_name"]),
        )

        dinov2_weights = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, self.conf["model_utils_name"]),
        )

        logger.info("Loading Roma model")
        # load the model
        weights = torch.load(model_path, map_location="cpu")
        dinov2_weights = torch.load(dinov2_weights, map_location="cpu")

        self.matcher = roma_model(
            resolution=(14 * 8 * 6, 14 * 8 * 6),
            upsample_preds=False,
            weights=weights,
            dinov2_weights=dinov2_weights,
            device=device,
            # temp fix issue: https://github.com/Parskatt/RoMa/issues/26
            amp_dtype=torch.float32,
        )
        self.detector = dad_detector.DaD()
        logger.info("Load Dad + Roma model done.")

    def _forward(self, data):
        img0 = data["image0"].cpu().numpy().squeeze() * 255
        img1 = data["image1"].cpu().numpy().squeeze() * 255
        img0 = img0.transpose(1, 2, 0)
        img1 = img1.transpose(1, 2, 0)
        img0 = Image.fromarray(img0.astype("uint8"))
        img1 = Image.fromarray(img1.astype("uint8"))
        W_A, H_A = img0.size
        W_B, H_B = img1.size

        # Match
        warp, certainty = self.matcher.match(img0, img1, device=device)

        # Sample matches for estimation
        keypoints_A = self.detector.detect(
            img0,
            num_keypoints=self.conf["max_keypoints"],
        )["keypoints"][0]
        keypoints_B = self.detector.detect(
            img1,
            num_keypoints=self.conf["max_keypoints"],
        )["keypoints"][0]
        matches = self.matcher.match_keypoints(
            keypoints_A,
            keypoints_B,
            warp,
            certainty,
            return_tuple=False,
        )

        # Sample matches for estimation
        kpts1, kpts2 = self.matcher.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
        pred = {
            "keypoints0": kpts1,
            "keypoints1": kpts2,
            "mconf": certainty,
        }
        return pred
