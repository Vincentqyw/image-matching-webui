import sys
from pathlib import Path
import tempfile
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
        "coarse_res": (560, 560),
        "upsample_res": (864, 1152),
    }
    required_inputs = [
        "image0",
        "image1",
    ]

    # Initialize the line matcher
    def _init(self, conf):
        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format("roma", self.conf["model_name"]),
        )

        dinov2_weights = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format("roma", self.conf["model_utils_name"]),
        )

        logger.info("Loading Dad + Roma model")
        # load the model
        weights = torch.load(model_path, map_location="cpu")
        dinov2_weights = torch.load(dinov2_weights, map_location="cpu")

        if str(device) == "cpu":
            amp_dtype = torch.float32
        else:
            amp_dtype = torch.float16

        self.matcher = roma_model(
            resolution=self.conf["coarse_res"],
            upsample_preds=True,
            weights=weights,
            dinov2_weights=dinov2_weights,
            device=device,
            amp_dtype=amp_dtype,
        )
        self.matcher.upsample_res = self.conf["upsample_res"]
        self.matcher.symmetric = False

        self.detector = dad_detector.load_DaD()
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

        # hack: bad way to save then match
        with (
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img0,
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img1,
        ):
            img0_path = temp_img0.name
            img1_path = temp_img1.name
            img0.save(img0_path)
            img1.save(img1_path)

        # Match
        warp, certainty = self.matcher.match(img0_path, img1_path, device=device)
        # Detect
        keypoints_A = self.detector.detect_from_path(
            img0_path,
            num_keypoints=self.conf["max_keypoints"],
        )["keypoints"][0]
        keypoints_B = self.detector.detect_from_path(
            img1_path,
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
        offset = self.detector.topleft - 0
        kpts1, kpts2 = kpts1 - offset, kpts2 - offset
        pred = {
            "keypoints0": self.matcher._to_pixel_coordinates(keypoints_A, H_A, W_A),
            "keypoints1": self.matcher._to_pixel_coordinates(keypoints_B, H_B, W_B),
            "mkeypoints0": kpts1,
            "mkeypoints1": kpts2,
            "mconf": torch.ones_like(kpts1[:, 0]),
        }
        return pred
