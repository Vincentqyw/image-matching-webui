import subprocess
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

tp_path = Path(__file__).parent / "../../third_party"
sys.path.append(str(tp_path))

from EfficientLoFTR.src.loftr import LoFTR as ELoFTR_
from EfficientLoFTR.src.loftr import (
    full_default_cfg,
    opt_default_cfg,
    reparameter,
)

from hloc import logger

from ..utils.base_model import BaseModel


class ELoFTR(BaseModel):
    default_conf = {
        "weights": "weights/eloftr_outdoor.ckpt",
        "match_threshold": 0.2,
        # "sinkhorn_iterations": 20,
        "max_keypoints": -1,
        # You can choose model type in ['full', 'opt']
        "model_type": "full",  # 'full' for best quality, 'opt' for best efficiency
        # You can choose numerical precision in ['fp32', 'mp', 'fp16']. 'fp16' for best efficiency
        "precision": "fp32",
    }
    required_inputs = ["image0", "image1"]

    def _init(self, conf):

        if self.conf["model_type"] == "full":
            _default_cfg = deepcopy(full_default_cfg)
        elif self.conf["model_type"] == "opt":
            _default_cfg = deepcopy(opt_default_cfg)

        if self.conf["precision"] == "mp":
            _default_cfg["mp"] = True
        elif self.conf["precision"] == "fp16":
            _default_cfg["half"] = True

        model_path = tp_path / "EfficientLoFTR" / self.conf["weights"]

        # Download the model.
        if not model_path.exists():
            model_path.parent.mkdir(exist_ok=True)
            cached_file = hf_hub_download(
                repo_type="space",
                repo_id="Realcat/image-matching-webui",
                filename="third_party/EfficientLoFTR/{}".format(
                    conf["weights"]
                ),
            )
            logger.info("Downloaded EfficientLoFTR model succeeed!")
            cmd = [
                "cp",
                str(cached_file),
                str(model_path),
            ]
            subprocess.run(cmd, check=True)
            logger.info(f"Copy model file `{cmd}`.")

        cfg = _default_cfg
        cfg["match_coarse"]["thr"] = conf["match_threshold"]
        # cfg["match_coarse"]["skh_iters"] = conf["sinkhorn_iterations"]
        state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        matcher = ELoFTR_(config=cfg)
        matcher.load_state_dict(state_dict)
        self.net = reparameter(matcher)

        if self.conf["precision"] == "fp16":
            self.net = self.net.half()
        logger.info(f"Loaded Efficient LoFTR with weights {conf['weights']}")

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
        scores = data_["mconf"]

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
