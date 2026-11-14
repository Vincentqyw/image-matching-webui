import sys
import yaml
from pathlib import Path
from ..utils.base_model import BaseModel
from .. import logger, MODEL_REPO_ID, DEVICE

rdd_path = Path(__file__).parent / "../../third_party/rdd"
sys.path.append(str(rdd_path))

from RDD.RDD import build as build_rdd
from RDD.RDD_helper import RDD_helper


class RddDense(BaseModel):
    default_conf = {
        "keypoint_threshold": 0.1,
        "max_keypoints": 4096,
        "model_name": "RDD-v2.pth",
        "match_threshold": 0.1,
    }

    required_inputs = ["image0", "image1"]

    def _init(self, conf):
        logger.info("Loading RDD model...")
        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format("rdd", self.conf["model_name"]),
        )
        config_path = rdd_path / "configs/default.yaml"
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        config["top_k"] = conf["max_keypoints"]
        config["detection_threshold"] = conf["keypoint_threshold"]
        config["device"] = DEVICE
        # RDD's ResNet50 backbone requests torchvision's ImageNet weights.
        self._download_torch_checkpoint(
            "rdd/resnet50-0676ba61.pth", "resnet50-0676ba61.pth"
        )
        rdd_net = build_rdd(config=config, weights=model_path)
        rdd_net.eval()
        self.net = RDD_helper(rdd_net)
        logger.info("Loading RDD model done!")

    def _forward(self, data):
        # RDD_helper.match_dense hardcodes n_limit=30000; call the pieces so the
        # Max features slider actually caps the keypoints used for matching.
        out0 = self.net.RDD.extract_dense(
            data["image0"], n_limit=self.conf["max_keypoints"]
        )[0]
        out1 = self.net.RDD.extract_dense(
            data["image1"], n_limit=self.conf["max_keypoints"]
        )[0]
        mkpts_0, mkpts_1, mconf = self.net.dense_matcher(
            out0, out1, thr=self.conf["match_threshold"], err_thr=self.net.RDD.stride
        )
        return {"keypoints0": mkpts_0, "keypoints1": mkpts_1, "mconf": mconf}
