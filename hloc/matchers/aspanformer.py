import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from .. import logger
from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / "../../third_party"))
from ASpanFormer.src.ASpanFormer.aspanformer import ASpanFormer as _ASpanFormer
from ASpanFormer.src.config.default import get_cfg_defaults
from ASpanFormer.src.utils.misc import lower_config

aspanformer_path = Path(__file__).parent / "../../third_party/ASpanFormer"


class ASpanFormer(BaseModel):
    default_conf = {
        "weights": "outdoor",
        "match_threshold": 0.2,
        "sinkhorn_iterations": 20,
        "max_keypoints": 2048,
        "config_path": aspanformer_path / "configs/aspan/outdoor/aspan_test.py",
        "model_name": "weights_aspanformer.tar",
    }
    required_inputs = ["image0", "image1"]
    proxy = "http://localhost:1080"
    aspanformer_models = {
        "weights_aspanformer.tar": "https://drive.google.com/uc?id=1eavM9dTkw9nbc-JqlVVfGPU5UvTTfc6k&confirm=t"
    }

    def _init(self, conf):
        model_path = (
            aspanformer_path / "weights" / Path(conf["weights"] + ".ckpt")
        )
        # Download the model.
        if not model_path.exists():
            cached_file = hf_hub_download(
                repo_type="space",
                repo_id="Realcat/image-matching-webui",
                filename="third_party/ASpanFormer/weights_aspanformer.tar",
            )
            cmd = [
                "tar",
                "-xvf",
                str(cached_file),
                "-C",
                str(aspanformer_path / "weights"),
            ]
            logger.info(f"Unzip model file `{cmd}`.")
            subprocess.run(cmd, check=True)

        config = get_cfg_defaults()
        config.merge_from_file(conf["config_path"])
        _config = lower_config(config)

        # update: match threshold
        _config["aspan"]["match_coarse"]["thr"] = conf["match_threshold"]
        _config["aspan"]["match_coarse"]["skh_iters"] = conf[
            "sinkhorn_iterations"
        ]

        self.net = _ASpanFormer(config=_config["aspan"])
        weight_path = model_path
        state_dict = torch.load(str(weight_path), map_location="cpu")[
            "state_dict"
        ]
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
