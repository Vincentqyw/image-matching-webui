import sys
import torch
from ..utils.base_model import BaseModel
from ..utils import do_system
from pathlib import Path
import subprocess
from .. import logger

sys.path.append(str(Path(__file__).parent / "../../third_party"))
from ASpanFormer.src.ASpanFormer.aspanformer import ASpanFormer as _ASpanFormer
from ASpanFormer.src.config.default import get_cfg_defaults
from ASpanFormer.src.utils.misc import lower_config
from ASpanFormer.demo import demo_utils

aspanformer_path = Path(__file__).parent / "../../third_party/ASpanFormer"


class ASpanFormer(BaseModel):
    default_conf = {
        "weights": "outdoor",
        "match_threshold": 0.2,
        "sinkhorn_iterations": 20,
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
            # model_path.parent.mkdir(exist_ok=True)
            tar_path = aspanformer_path / conf["model_name"]
            if not tar_path.exists():
                link = self.aspanformer_models[conf["model_name"]]
                cmd = [
                    "gdown",
                    link,
                    "-O",
                    str(tar_path),
                    "--proxy",
                    self.proxy,
                ]
                cmd_wo_proxy = ["gdown", link, "-O", str(tar_path)]
                logger.info(
                    f"Downloading the Aspanformer model with `{cmd_wo_proxy}`."
                )
                try:
                    subprocess.run(cmd_wo_proxy, check=True)
                except subprocess.CalledProcessError as e:
                    logger.info(
                        f"Downloading the Aspanformer model with `{cmd}`."
                    )
                    try:
                        subprocess.run(cmd, check=True)
                    except subprocess.CalledProcessError as e:
                        logger.error(
                            f"Failed to download the Aspanformer model."
                        )
                        raise e

            do_system(f"cd {str(aspanformer_path)} & tar -xvf {str(tar_path)}")

        logger.info(f"Loading Aspanformer model...")

        config = get_cfg_defaults()
        config.merge_from_file(conf["config_path"])
        _config = lower_config(config)

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
        return pred
