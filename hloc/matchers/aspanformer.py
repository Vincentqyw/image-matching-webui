import sys
from pathlib import Path

import torch

from hloc import MODEL_REPO_ID, logger
from hloc.utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / "../../third_party"))
from ASpanFormer.src.ASpanFormer.aspanformer import ASpanFormer as _ASpanFormer
from ASpanFormer.src.config.default import get_cfg_defaults
from ASpanFormer.src.utils.misc import lower_config

aspanformer_path = Path(__file__).parent / "../../third_party/ASpanFormer"


class ASpanFormer(BaseModel):
    default_conf = {
        "model_name": "outdoor.ckpt",
        "match_threshold": 0.2,
        "sinkhorn_iterations": 20,
        "max_keypoints": 2048,
        "config_path": aspanformer_path / "configs/aspan/outdoor/aspan_test.py",
    }
    required_inputs = ["image0", "image1"]

    def _init(self, conf):
        config = get_cfg_defaults()
        config.merge_from_file(conf["config_path"])
        _config = lower_config(config)

        # update: match threshold
        _config["aspan"]["match_coarse"]["thr"] = conf["match_threshold"]
        _config["aspan"]["match_coarse"]["skh_iters"] = conf["sinkhorn_iterations"]

        self.net = _ASpanFormer(config=_config["aspan"])
        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, self.conf["model_name"]),
        )
        state_dict = torch.load(str(model_path), map_location="cpu")["state_dict"]
        self.net.load_state_dict(state_dict, strict=False)
        logger.info("Loaded Aspanformer model")

    def _forward(self, data):
        data_ = {
            "image0": data["image0"],
            "image1": data["image1"],
        }
        self.net(data_, online_resize=True)
        pred = {
            "keypoints0": data_["mkpts0_f"],
            "keypoints1": data_["mkpts1_f"],
            "mconf": data_["mconf"],
        }
        scores = data_["mconf"]
        top_k = self.conf["max_keypoints"]
        if top_k is not None and len(scores) > top_k:
            keep = torch.argsort(scores, descending=True)[:top_k]
            scores = scores[keep]
            pred["keypoints0"], pred["keypoints1"], pred["mconf"] = (
                pred["keypoints0"][keep],
                pred["keypoints1"][keep],
                scores,
            )
        return pred
