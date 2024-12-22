import sys
from collections import OrderedDict, namedtuple
from pathlib import Path

import torch

from .. import MODEL_REPO_ID, logger
from ..utils.base_model import BaseModel

sgmnet_path = Path(__file__).parent / "../../third_party/SGMNet"
sys.path.append(str(sgmnet_path))

from sgmnet import matcher as SGM_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SGMNet(BaseModel):
    default_conf = {
        "name": "SGM",
        "model_name": "weights/sgm/root/model_best.pth",
        "seed_top_k": [256, 256],
        "seed_radius_coe": 0.01,
        "net_channels": 128,
        "layer_num": 9,
        "head": 4,
        "seedlayer": [0, 6],
        "use_mc_seeding": True,
        "use_score_encoding": False,
        "conf_bar": [1.11, 0.1],
        "sink_iter": [10, 100],
        "detach_iter": 1000000,
        "match_threshold": 0.2,
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

        # config
        config = namedtuple("config", conf.keys())(*conf.values())
        self.net = SGM_Model(config)
        checkpoint = torch.load(model_path, map_location="cpu")
        # for ddp model
        if list(checkpoint["state_dict"].items())[0][0].split(".")[0] == "module":
            new_stat_dict = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                new_stat_dict[key[7:]] = value
            checkpoint["state_dict"] = new_stat_dict
        self.net.load_state_dict(checkpoint["state_dict"])
        logger.info("Load SGMNet model done.")

    def _forward(self, data):
        x1 = data["keypoints0"].squeeze()  # N x 2
        x2 = data["keypoints1"].squeeze()
        score1 = data["scores0"].reshape(-1, 1)  # N x 1
        score2 = data["scores1"].reshape(-1, 1)
        desc1 = data["descriptors0"].permute(0, 2, 1)  # 1 x N x 128
        desc2 = data["descriptors1"].permute(0, 2, 1)
        size1 = (
            torch.tensor(data["image0"].shape[2:]).flip(0).to(x1.device)
        )  # W x H -> x & y
        size2 = torch.tensor(data["image1"].shape[2:]).flip(0).to(x2.device)  # W x H
        norm_x1 = self.normalize_size(x1, size1)
        norm_x2 = self.normalize_size(x2, size2)

        x1 = torch.cat((norm_x1, score1), dim=-1)  # N x 3
        x2 = torch.cat((norm_x2, score2), dim=-1)
        input = {"x1": x1[None], "x2": x2[None], "desc1": desc1, "desc2": desc2}
        input = {
            k: v.to(device).float() if isinstance(v, torch.Tensor) else v
            for k, v in input.items()
        }
        pred = self.net(input, test_mode=True)

        p = pred["p"]  # shape: N * M
        indices0 = self.match_p(p[0, :-1, :-1])
        pred = {
            "matches0": indices0.unsqueeze(0),
            "matching_scores0": torch.zeros(indices0.size(0)).unsqueeze(0),
        }
        return pred

    def match_p(self, p):
        score, index = torch.topk(p, k=1, dim=-1)
        _, index2 = torch.topk(p, k=1, dim=-2)
        mask_th, index, index2 = (
            score[:, 0] > self.conf["match_threshold"],
            index[:, 0],
            index2.squeeze(0),
        )
        mask_mc = index2[index] == torch.arange(len(p)).to(device)
        mask = mask_th & mask_mc
        indices0 = torch.where(mask, index, index.new_tensor(-1))
        return indices0

    def normalize_size(self, x, size, scale=1):
        norm_fac = size.max()
        return (x - size / 2 + 0.5) / (norm_fac * scale)
