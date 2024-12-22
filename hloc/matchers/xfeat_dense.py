import torch

from hloc import logger

from ..utils.base_model import BaseModel


class XFeatDense(BaseModel):
    default_conf = {
        "keypoint_threshold": 0.005,
        "max_keypoints": 8000,
    }
    required_inputs = [
        "image0",
        "image1",
    ]

    def _init(self, conf):
        self.net = torch.hub.load(
            "verlab/accelerated_features",
            "XFeat",
            pretrained=True,
            top_k=self.conf["max_keypoints"],
        )
        logger.info("Load XFeat(dense) model done.")

    def _forward(self, data):
        # Compute coarse feats
        out0 = self.net.detectAndComputeDense(
            data["image0"], top_k=self.conf["max_keypoints"]
        )
        out1 = self.net.detectAndComputeDense(
            data["image1"], top_k=self.conf["max_keypoints"]
        )

        # Match batches of pairs
        idxs_list = self.net.batch_match(out0["descriptors"], out1["descriptors"])
        B = len(data["image0"])

        # Refine coarse matches
        # this part is harder to batch, currently iterate
        matches = []
        for b in range(B):
            matches.append(
                self.net.refine_matches(out0, out1, matches=idxs_list, batch_idx=b)
            )
        # we use results from one batch
        matches = matches[0]
        pred = {
            "keypoints0": matches[:, :2],
            "keypoints1": matches[:, 2:],
            "mconf": torch.ones_like(matches[:, 0]),
        }
        return pred
