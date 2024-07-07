import sys
from pathlib import Path

import torch

from .. import logger
from ..utils.base_model import BaseModel

example_path = Path(__file__).parent / "../../third_party/example"
sys.path.append(str(example_path))

# import some modules here

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Example(BaseModel):
    # change to your default configs
    default_conf = {
        "name": "example",
        "keypoint_threshold": 0.1,
        "max_keypoints": 2000,
        "model_name": "model.pth",
    }
    required_inputs = ["image"]

    def _init(self, conf):
        # set checkpoints paths if needed
        model_path = example_path / "checkpoints" / f'{conf["model_name"]}'
        if not model_path.exists():
            logger.info(f"No model found at {model_path}")

        # init model
        self.net = callable
        # self.net = ExampleNet(is_test=True)
        state_dict = torch.load(model_path, map_location="cpu")
        self.net.load_state_dict(state_dict["model_state"])
        logger.info("Load example model done.")

    def _forward(self, data):
        # data: dict, keys: 'image'
        # image color mode: RGB
        # image value range in [0, 1]
        image = data["image"]

        # B: batch size, N: number of keypoints
        # keypoints shape: B x N x 2, type: torch tensor
        # scores shape: B x N, type: torch tensor
        # descriptors shape: B x 128 x N, type: torch tensor
        keypoints, scores, descriptors = self.net(image)

        return {
            "keypoints": keypoints,
            "scores": scores,
            "descriptors": descriptors,
        }
