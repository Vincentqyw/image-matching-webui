import sys
from pathlib import Path

from PIL import Image

from hloc import DEVICE, MODEL_REPO_ID, logger
from hloc.utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / "../../third_party"))
from DKM.dkm import DKMv3_outdoor


class DKMv3(BaseModel):
    default_conf = {
        "model_name": "DKMv3_outdoor.pth",
        "match_threshold": 0.2,
        "max_keypoints": -1,
    }
    required_inputs = [
        "image0",
        "image1",
    ]

    def _init(self, conf):
        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, self.conf["model_name"]),
        )

        self.net = DKMv3_outdoor(path_to_weights=str(model_path), device=DEVICE)
        logger.info("Loading DKMv3 model done")

    def _forward(self, data):
        img0 = data["image0"].cpu().numpy().squeeze() * 255
        img1 = data["image1"].cpu().numpy().squeeze() * 255
        img0 = img0.transpose(1, 2, 0)
        img1 = img1.transpose(1, 2, 0)
        img0 = Image.fromarray(img0.astype("uint8"))
        img1 = Image.fromarray(img1.astype("uint8"))
        W_A, H_A = img0.size
        W_B, H_B = img1.size

        warp, certainty = self.net.match(img0, img1, device=DEVICE)
        matches, certainty = self.net.sample(
            warp, certainty, num=self.conf["max_keypoints"]
        )
        kpts1, kpts2 = self.net.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
        pred = {
            "keypoints0": kpts1,
            "keypoints1": kpts2,
            "mconf": certainty,
        }
        return pred
