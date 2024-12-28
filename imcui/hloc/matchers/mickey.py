import sys
from pathlib import Path

from .. import MODEL_REPO_ID, logger
from ..utils.base_model import BaseModel

mickey_path = Path(__file__).parent / "../../third_party"
sys.path.append(str(mickey_path))

from mickey.config.default import cfg
from mickey.lib.models.builder import build_model


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

    # Initialize the line matcher
    def _init(self, conf):
        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, self.conf["model_name"]),
        )
        # TODO: config path of mickey
        config_path = model_path.parent / self.conf["config_path"]
        logger.info("Loading mickey model...")
        cfg.merge_from_file(config_path)
        self.net = build_model(cfg, checkpoint=model_path)
        logger.info("Load Mickey model done.")

    def _forward(self, data):
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
