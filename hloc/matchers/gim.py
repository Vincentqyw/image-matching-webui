import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from .. import logger
from ..utils.base_model import BaseModel

gim_path = Path(__file__).parent / "../../third_party/gim"
sys.path.append(str(gim_path))

from dkm.models.model_zoo.DKMv3 import DKMv3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GIM(BaseModel):
    default_conf = {
        "model_name": "gim_dkm_100h.ckpt",
        "match_threshold": 0.2,
        "checkpoint_dir": gim_path / "weights",
    }
    required_inputs = [
        "image0",
        "image1",
    ]
    model_list = ["gim_lightglue_100h.ckpt", "gim_dkm_100h.ckpt"]
    model_dict = {
        "gim_lightglue_100h.ckpt": "https://github.com/xuelunshen/gim/blob/main/weights/gim_lightglue_100h.ckpt",
        "gim_dkm_100h.ckpt": "https://drive.google.com/file/d/1gk97V4IROnR1Nprq10W9NCFUv2mxXR_-/view",
    }

    def _init(self, conf):
        conf["model_name"] = str(conf["weights"])
        if conf["model_name"] not in self.model_list:
            raise ValueError(f"Unknown GIM model {conf['model_name']}.")
        model_path = conf["checkpoint_dir"] / conf["model_name"]

        # Download the model.
        if not model_path.exists():
            model_path.parent.mkdir(exist_ok=True)
            cached_file = hf_hub_download(
                repo_type="space",
                repo_id="Realcat/image-matching-webui",
                filename="third_party/gim/weights/{}".format(
                    conf["model_name"]
                ),
            )
            logger.info("Downloaded GIM model succeeed!")
            cmd = [
                "cp",
                str(cached_file),
                str(conf["checkpoint_dir"]),
            ]
            subprocess.run(cmd, check=True)
            logger.info(f"Copy model file `{cmd}`.")

        self.aspect_ratio = 896 / 672
        model = DKMv3(None, 672, 896, upsample_preds=True)
        state_dict = torch.load(str(model_path), map_location="cpu")
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        for k in list(state_dict.keys()):
            if k.startswith("model."):
                state_dict[k.replace("model.", "", 1)] = state_dict.pop(k)
            if "encoder.net.fc" in k:
                state_dict.pop(k)
        model.load_state_dict(state_dict)

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
        image0, image1 = self.pad_image(
            data["image0"], self.aspect_ratio
        ), self.pad_image(data["image1"], self.aspect_ratio)
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
