import sys
from pathlib import Path
import torch
from PIL import Image
import subprocess
from ..utils.base_model import BaseModel
from .. import logger

sys.path.append(str(Path(__file__).parent / "../../third_party"))
from DKM.dkm import DKMv3_outdoor

dkm_path = Path(__file__).parent / "../../third_party/DKM"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DKMv3(BaseModel):
    default_conf = {
        "model_name": "DKMv3_outdoor.pth",
        "match_threshold": 0.2,
        "checkpoint_dir": dkm_path / "pretrained",
    }
    required_inputs = [
        "image0",
        "image1",
    ]
    # Models exported using
    dkm_models = {
        "DKMv3_outdoor.pth": "https://github.com/Parskatt/storage/releases/download/dkmv3/DKMv3_outdoor.pth",
        "DKMv3_indoor.pth": "https://github.com/Parskatt/storage/releases/download/dkmv3/DKMv3_indoor.pth",
    }

    def _init(self, conf):
        model_path = dkm_path / "pretrained" / conf["model_name"]

        # Download the model.
        if not model_path.exists():
            model_path.parent.mkdir(exist_ok=True)
            link = self.dkm_models[conf["model_name"]]
            cmd = ["wget", link, "-O", str(model_path)]
            logger.info(f"Downloading the DKMv3 model with `{cmd}`.")
            subprocess.run(cmd, check=True)
        logger.info(f"Loading DKMv3 model...")
        self.net = DKMv3_outdoor(path_to_weights=str(model_path), device=device)

    def _forward(self, data):
        img0 = data["image0"].cpu().numpy().squeeze() * 255
        img1 = data["image1"].cpu().numpy().squeeze() * 255
        img0 = img0.transpose(1, 2, 0)
        img1 = img1.transpose(1, 2, 0)
        img0 = Image.fromarray(img0.astype("uint8"))
        img1 = Image.fromarray(img1.astype("uint8"))
        W_A, H_A = img0.size
        W_B, H_B = img1.size

        warp, certainty = self.net.match(img0, img1, device=device)
        matches, certainty = self.net.sample(warp, certainty)
        kpts1, kpts2 = self.net.to_pixel_coordinates(
            matches, H_A, W_A, H_B, W_B
        )
        pred = {}
        pred["keypoints0"], pred["keypoints1"] = kpts1, kpts2
        return pred
