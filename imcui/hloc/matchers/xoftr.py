import sys
import warnings
from pathlib import Path

import torch

from hloc import DEVICE, MODEL_REPO_ID

tp_path = Path(__file__).parent / "../../third_party"
sys.path.append(str(tp_path))

from XoFTR.src.config.default import get_cfg_defaults
from XoFTR.src.utils.misc import lower_config
from XoFTR.src.xoftr import XoFTR as XoFTR_

from hloc import logger

from ..utils.base_model import BaseModel


class XoFTR(BaseModel):
    default_conf = {
        "model_name": "weights_xoftr_640.ckpt",
        "match_threshold": 0.3,
        "max_keypoints": -1,
    }
    required_inputs = ["image0", "image1"]

    def _init(self, conf):
        # Get default configurations
        config_ = get_cfg_defaults(inference=True)
        config_ = lower_config(config_)

        # Coarse level threshold
        config_["xoftr"]["match_coarse"]["thr"] = self.conf["match_threshold"]

        # Fine level threshold
        config_["xoftr"]["fine"]["thr"] = 0.1  # Default 0.1

        # It is posseble to get denser matches
        # If True, xoftr returns all fine-level matches for each fine-level window (at 1/2 resolution)
        config_["xoftr"]["fine"]["denser"] = False  # Default False

        # XoFTR model
        matcher = XoFTR_(config=config_["xoftr"])

        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, self.conf["model_name"]),
        )

        # Load model
        state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        matcher.load_state_dict(state_dict, strict=True)
        matcher = matcher.eval().to(DEVICE)
        self.net = matcher
        logger.info(f"Loaded XoFTR with weights {conf['model_name']}")

    def _forward(self, data):
        # For consistency with hloc pairs, we refine kpts in image0!
        rename = {
            "keypoints0": "keypoints1",
            "keypoints1": "keypoints0",
            "image0": "image1",
            "image1": "image0",
            "mask0": "mask1",
            "mask1": "mask0",
        }
        data_ = {rename[k]: v for k, v in data.items()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self.net(data_)
        pred = {
            "keypoints0": data_["mkpts0_f"],
            "keypoints1": data_["mkpts1_f"],
        }
        scores = data_["mconf_f"]

        top_k = self.conf["max_keypoints"]
        if top_k is not None and len(scores) > top_k:
            keep = torch.argsort(scores, descending=True)[:top_k]
            pred["keypoints0"], pred["keypoints1"] = (
                pred["keypoints0"][keep],
                pred["keypoints1"][keep],
            )
            scores = scores[keep]

        # Switch back indices
        pred = {(rename[k] if k in rename else k): v for k, v in pred.items()}
        pred["scores"] = scores
        return pred
