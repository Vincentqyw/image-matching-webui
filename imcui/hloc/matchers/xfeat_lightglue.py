import torch

from hloc import logger

from ..utils.base_model import BaseModel


class XFeatLightGlue(BaseModel):
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
        # we use results from one batch
        im0 = data["image0"]
        im1 = data["image1"]
        # Compute coarse feats
        out0 = self.net.detectAndCompute(im0, top_k=self.conf["max_keypoints"])[0]
        out1 = self.net.detectAndCompute(im1, top_k=self.conf["max_keypoints"])[0]
        out0.update({"image_size": (im0.shape[-1], im0.shape[-2])})  # W H
        out1.update({"image_size": (im1.shape[-1], im1.shape[-2])})  # W H
        pred = self.net.match_lighterglue(out0, out1)
        if len(pred) == 3:
            mkpts_0, mkpts_1, _ = pred
        else:
            mkpts_0, mkpts_1 = pred
        mkpts_0 = torch.from_numpy(mkpts_0)  # n x 2
        mkpts_1 = torch.from_numpy(mkpts_1)  # n x 2
        pred = {
            "keypoints0": mkpts_0,
            "keypoints1": mkpts_1,
            "mconf": torch.ones_like(mkpts_0[:, 0]),
        }
        return pred
