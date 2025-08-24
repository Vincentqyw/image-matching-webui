import sys
from pathlib import Path

from .. import MODEL_REPO_ID, logger
from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / "../../third_party"))
from SuperGluePretrainedNetwork.models.superglue import (  # noqa: E402
    SuperGlue as SG,
)


class SuperGlue(BaseModel):
    default_conf = {
        "weights": "outdoor",
        "model_name": "superglue_outdoor.pth",
        "sinkhorn_iterations": 100,
        "match_threshold": 0.2,
    }
    required_inputs = [
        "image0",
        "keypoints0",
        "scores0",
        "descriptors0",
        "image1",
        "keypoints1",
        "scores1",
        "descriptors1",
    ]

    def _init(self, conf):
        weights_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format("superglue", self.conf["model_name"]),
        )
        conf["weights_path"] = str(weights_path)
        self.net = SG(conf)
        logger.info(
            'Loaded SuperGlue model ("{}" weights)'.format(self.conf["weights"])
        )

    def _forward(self, data):
        return self.net(data)
