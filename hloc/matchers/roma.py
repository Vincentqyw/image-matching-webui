import sys
from pathlib import Path
import subprocess
import torch
from PIL import Image
from ..utils.base_model import BaseModel
from .. import logger

roma_path = Path(__file__).parent / "../../third_party/Roma"
sys.path.append(str(roma_path))

from roma.models.model_zoo.roma_models import roma_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Roma(BaseModel):
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
    weight_urls = {
        "roma": {
            "roma_outdoor.pth": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
            "roma_indoor.pth": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
        },
        "dinov2_vitl14_pretrain.pth": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
    }

    # Initialize the line matcher
    def _init(self, conf):
        model_path = roma_path / "pretrained" / conf["model_name"]
        dinov2_weights = roma_path / "pretrained" / conf["model_utils_name"]

        # Download the model.
        if not model_path.exists():
            model_path.parent.mkdir(exist_ok=True)
            link = self.weight_urls["roma"][conf["model_name"]]
            cmd = ["wget", link, "-O", str(model_path)]
            logger.info(f"Downloading the Roma model with `{cmd}`.")
            subprocess.run(cmd, check=True)

        if not dinov2_weights.exists():
            dinov2_weights.parent.mkdir(exist_ok=True)
            link = self.weight_urls[conf["model_utils_name"]]
            cmd = ["wget", link, "-O", str(dinov2_weights)]
            logger.info(f"Downloading the dinov2 model with `{cmd}`.")
            subprocess.run(cmd, check=True)

        logger.info(f"Loading Roma model...")
        # load the model
        weights = torch.load(model_path, map_location="cpu")
        dinov2_weights = torch.load(dinov2_weights, map_location="cpu")

        self.net = roma_model(
            resolution=(14 * 8 * 6, 14 * 8 * 6),
            upsample_preds=False,
            weights=weights,
            dinov2_weights=dinov2_weights,
            device=device,
        )
        logger.info(f"Load Roma model done.")

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
        warp, certainty = self.net.match(img0, img1, device=device)
        # Sample matches for estimation
        matches, certainty = self.net.sample(
            warp, certainty, num=self.conf["max_keypoints"]
        )
        kpts1, kpts2 = self.net.to_pixel_coordinates(
            matches, H_A, W_A, H_B, W_B
        )
        pred = {
            "keypoints0": kpts1,
            "keypoints1": kpts2,
            "mconf": certainty,
        }

        return pred
