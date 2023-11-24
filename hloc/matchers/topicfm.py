import torch
import warnings
from ..utils.base_model import BaseModel
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "../../third_party"))
from TopicFM.src.models.topic_fm import TopicFM as _TopicFM
from TopicFM.src import get_model_cfg

topicfm_path = Path(__file__).parent / "../../third_party/TopicFM"


class TopicFM(BaseModel):
    default_conf = {
        "weights": "outdoor",
        "match_threshold": 0.2,
        "n_sampling_topics": 4,
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
        return pred
