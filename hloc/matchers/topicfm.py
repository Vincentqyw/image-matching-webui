import sys
from pathlib import Path

import torch

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / "../../third_party"))
from TopicFM.src import get_model_cfg
from TopicFM.src.models.topic_fm import TopicFM as _TopicFM

topicfm_path = Path(__file__).parent / "../../third_party/TopicFM"


class TopicFM(BaseModel):
    default_conf = {
        "weights": "outdoor",
        "match_threshold": 0.2,
        "n_sampling_topics": 4,
        "max_keypoints": -1,
    }
    required_inputs = ["image0", "image1"]

    def _init(self, conf):
        _conf = dict(get_model_cfg())
        _conf["match_coarse"]["thr"] = conf["match_threshold"]
        _conf["coarse"]["n_samples"] = conf["n_sampling_topics"]
        weight_path = topicfm_path / "pretrained/model_best.ckpt"
        self.net = _TopicFM(config=_conf)
        ckpt_dict = torch.load(weight_path, map_location="cpu")
        self.net.load_state_dict(ckpt_dict["state_dict"])

    def _forward(self, data):
        data_ = {
            "image0": data["image0"],
            "image1": data["image1"],
        }
        self.net(data_)
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
