import sys
from pathlib import Path

from .. import MODEL_REPO_ID, logger
from ..utils.base_model import BaseModel

ripepp_path = Path(__file__).parent / "../../third_party/RIPEpp"
sys.path.append(str(ripepp_path))

from ripepp import load_model_from_checkpoint

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RIPEpp(BaseModel):
    default_conf = {
        "model_name": "ripe++.ckpt",
        "keypoint_threshold": 0.5,
        "max_keypoints": 2048,
    }

    required_inputs = ["image"]

    def _init(self, conf):
        logger.info("Loading RIPE++ model...")
        # The backbone is instantiated with torchvision's ImageNet VGG19-BN
        # weights (immediately overwritten by the checkpoint), so seed the
        # torch.hub cache from HuggingFace rather than download.pytorch.org.
        self._download_torch_checkpoint(
            "ripepp/vgg19_bn-c79401a0.pth", "vgg19_bn-c79401a0.pth"
        )
        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename=f"ripepp/{self.conf['model_name']}",
        )
        self.net = load_model_from_checkpoint(Path(model_path), device="cpu")
        logger.info("Loading RIPE++ model done!")

    def _forward(self, data):
        image = data["image"]
        mean = image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        # detectAndCompute runs under torch.inference_mode(); clone so the
        # descriptors can be consumed outside it.
        keypoints, descriptors, scores = self.net.detectAndCompute(
            (image - mean) / std,
            threshold=self.conf["keypoint_threshold"],
            top_k=self.conf["max_keypoints"],
        )
        return {
            "keypoints": keypoints.clone(),
            "descriptors": descriptors.permute(0, 2, 1).clone(),
            "scores": scores.clone(),
        }
