import sys
from pathlib import Path
import subprocess
import torch
from ..utils.base_model import BaseModel
from .. import logger

mickey_path = Path(__file__).parent / "../../third_party"
sys.path.append(str(mickey_path))

from mickey.lib.models.builder import build_model
from mickey.config.default import cfg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Mickey(BaseModel):
    default_conf = {
        "config_path": "config.yaml",
        "model_name": "mickey.ckpt",
        "max_keypoints": 3000,
    }
    required_inputs = [
        "image0",
        "image1",
    ]
    weight_urls = "https://storage.googleapis.com/niantic-lon-static/research/mickey/assets/mickey_weights.zip"

    # Initialize the line matcher
    def _init(self, conf):
        model_path = mickey_path / "mickey/mickey_weights" / conf["model_name"]
        zip_path = mickey_path / "mickey/mickey_weights.zip"
        config_path = model_path.parent / self.conf["config_path"]
        # Download the model.
        if not model_path.exists():
            model_path.parent.mkdir(exist_ok=True, parents=True)
            link = self.weight_urls
            if not zip_path.exists():
                cmd = ["wget", "--quiet", link, "-O", str(zip_path)]
                logger.info(f"Downloading the Mickey model with {cmd}.")
                subprocess.run(cmd, check=True)
            cmd = ["unzip", "-d", str(model_path.parent.parent), str(zip_path)]
            logger.info(f"Running {cmd}.")
            subprocess.run(cmd, check=True)

        logger.info("Loading mickey model...")
        cfg.merge_from_file(config_path)
        self.net = build_model(cfg, checkpoint=model_path)
        logger.info("Load Mickey model done.")

    def _forward(self, data):
        # data['K_color0'] = torch.from_numpy(K['im0.jpg']).unsqueeze(0).to(device)
        # data['K_color1'] = torch.from_numpy(K['im1.jpg']).unsqueeze(0).to(device)
        pred = self.net(data)
        pred = {
            **pred,
            **data,
        }
        inliers = data["inliers_list"]
        pred = {
            "keypoints0": inliers[:, :2],
            "keypoints1": inliers[:, 2:4],
        }

        return pred
