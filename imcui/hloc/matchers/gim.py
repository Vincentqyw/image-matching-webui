import sys
from pathlib import Path

import torch

from .. import DEVICE, MODEL_REPO_ID, logger
from ..utils.base_model import BaseModel

gim_path = Path(__file__).parents[2] / "third_party/gim"
sys.path.append(str(gim_path))


def load_model(weight_name, checkpoints_path):
    # load model
    model = None
    detector = None
    if weight_name == "gim_dkm":
        from networks.dkm.models.model_zoo.DKMv3 import DKMv3

        model = DKMv3(weights=None, h=672, w=896)
    elif weight_name == "gim_loftr":
        from networks.loftr.config import get_cfg_defaults
        from networks.loftr.loftr import LoFTR
        from networks.loftr.misc import lower_config

        model = LoFTR(lower_config(get_cfg_defaults())["loftr"])
    elif weight_name == "gim_lightglue":
        from networks.lightglue.models.matchers.lightglue import LightGlue
        from networks.lightglue.superpoint import SuperPoint

        detector = SuperPoint(
            {
                "max_num_keypoints": 2048,
                "force_num_keypoints": True,
                "detection_threshold": 0.0,
                "nms_radius": 3,
                "trainable": False,
            }
        )
        model = LightGlue(
            {
                "filter_threshold": 0.1,
                "flash": False,
                "checkpointed": True,
            }
        )

    # load state dict
    if weight_name == "gim_dkm":
        state_dict = torch.load(checkpoints_path, map_location="cpu")
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        for k in list(state_dict.keys()):
            if k.startswith("model."):
                state_dict[k.replace("model.", "", 1)] = state_dict.pop(k)
            if "encoder.net.fc" in k:
                state_dict.pop(k)
        model.load_state_dict(state_dict)

    elif weight_name == "gim_loftr":
        state_dict = torch.load(checkpoints_path, map_location="cpu")
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)

    elif weight_name == "gim_lightglue":
        state_dict = torch.load(checkpoints_path, map_location="cpu")
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        for k in list(state_dict.keys()):
            if k.startswith("model."):
                state_dict.pop(k)
            if k.startswith("superpoint."):
                state_dict[k.replace("superpoint.", "", 1)] = state_dict.pop(k)
        detector.load_state_dict(state_dict)

        state_dict = torch.load(checkpoints_path, map_location="cpu")
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        for k in list(state_dict.keys()):
            if k.startswith("superpoint."):
                state_dict.pop(k)
            if k.startswith("model."):
                state_dict[k.replace("model.", "", 1)] = state_dict.pop(k)
        model.load_state_dict(state_dict)

    # eval mode
    if detector is not None:
        detector = detector.eval().to(DEVICE)
    model = model.eval().to(DEVICE)
    return model


class GIM(BaseModel):
    default_conf = {
        "match_threshold": 0.2,
        "checkpoint_dir": gim_path / "weights",
        "weights": "gim_dkm",
    }
    required_inputs = [
        "image0",
        "image1",
    ]
    ckpt_name_dict = {
        "gim_dkm": "gim_dkm_100h.ckpt",
        "gim_loftr": "gim_loftr_50h.ckpt",
        "gim_lightglue": "gim_lightglue_100h.ckpt",
    }

    def _init(self, conf):
        ckpt_name = self.ckpt_name_dict[conf["weights"]]
        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, ckpt_name),
        )
        self.aspect_ratio = 896 / 672
        model = load_model(conf["weights"], model_path)
        self.net = model
        logger.info("Loaded GIM model")

    def pad_image(self, image, aspect_ratio):
        new_width = max(image.shape[3], int(image.shape[2] * aspect_ratio))
        new_height = max(image.shape[2], int(image.shape[3] / aspect_ratio))
        pad_width = new_width - image.shape[3]
        pad_height = new_height - image.shape[2]
        return torch.nn.functional.pad(
            image,
            (
                pad_width // 2,
                pad_width - pad_width // 2,
                pad_height // 2,
                pad_height - pad_height // 2,
            ),
        )

    def rescale_kpts(self, sparse_matches, shape0, shape1):
        kpts0 = torch.stack(
            (
                shape0[1] * (sparse_matches[:, 0] + 1) / 2,
                shape0[0] * (sparse_matches[:, 1] + 1) / 2,
            ),
            dim=-1,
        )
        kpts1 = torch.stack(
            (
                shape1[1] * (sparse_matches[:, 2] + 1) / 2,
                shape1[0] * (sparse_matches[:, 3] + 1) / 2,
            ),
            dim=-1,
        )
        return kpts0, kpts1

    def compute_mask(self, kpts0, kpts1, orig_shape0, orig_shape1):
        mask = (
            (kpts0[:, 0] > 0)
            & (kpts0[:, 1] > 0)
            & (kpts1[:, 0] > 0)
            & (kpts1[:, 1] > 0)
        )
        mask &= (
            (kpts0[:, 0] <= (orig_shape0[1] - 1))
            & (kpts1[:, 0] <= (orig_shape1[1] - 1))
            & (kpts0[:, 1] <= (orig_shape0[0] - 1))
            & (kpts1[:, 1] <= (orig_shape1[0] - 1))
        )
        return mask

    def _forward(self, data):
        # TODO: only support dkm+gim
        image0, image1 = (
            self.pad_image(data["image0"], self.aspect_ratio),
            self.pad_image(data["image1"], self.aspect_ratio),
        )
        dense_matches, dense_certainty = self.net.match(image0, image1)
        sparse_matches, mconf = self.net.sample(
            dense_matches, dense_certainty, self.conf["max_keypoints"]
        )
        kpts0, kpts1 = self.rescale_kpts(
            sparse_matches, image0.shape[-2:], image1.shape[-2:]
        )
        mask = self.compute_mask(
            kpts0, kpts1, data["image0"].shape[-2:], data["image1"].shape[-2:]
        )
        b_ids, i_ids = torch.where(mconf[None])
        pred = {
            "keypoints0": kpts0[i_ids],
            "keypoints1": kpts1[i_ids],
            "confidence": mconf[i_ids],
            "batch_indexes": b_ids,
        }
        scores, b_ids = pred["confidence"], pred["batch_indexes"]
        kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
        pred["confidence"], pred["batch_indexes"] = scores[mask], b_ids[mask]
        pred["keypoints0"], pred["keypoints1"] = kpts0[mask], kpts1[mask]

        out = {
            "keypoints0": pred["keypoints0"],
            "keypoints1": pred["keypoints1"],
        }
        return out
