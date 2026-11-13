import cv2
import numpy as np
import onnxruntime as ort
import torch

from .. import MODEL_REPO_ID, logger
from ..utils.base_model import BaseModel

# Sessions are shared across matcher-zoo entries, which only differ in the
# keypoint extractor used.
_sessions = {}


class FlexGlue(BaseModel):
    """Sparse matcher: keypoints from any detector, matched with FlexGlue ONNX."""

    default_conf = {
        "match_threshold": 0.1,
        "max_distance_px": 3.0,
        "min_covisibility": 0.0,
    }
    required_inputs = [
        "image0_orig",
        "image1_orig",
        "keypoints0",
        "keypoints1",
        "size0",
        "size1",
        "original_size0",
        "original_size1",
    ]

    def _init(self, _conf):
        providers = ["CPUExecutionProvider"]
        matcher_name = "flexglue_matcher.onnx"
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
            matcher_name = "flexglue_matcher_fp16.onnx"
        logger.info(f"Loading FlexGlue ONNX models ({matcher_name})")
        self.descriptor = self._session("flexglue_descriptor.onnx", providers)
        self.matcher = self._session(matcher_name, providers)
        logger.info("Load FlexGlue model done.")

    def _session(self, name, providers):
        if name not in _sessions:
            options = ort.SessionOptions()
            options.log_severity_level = 3  # errors only
            path = self._download_model(MODEL_REPO_ID, f"flexglue/{name}")
            _sessions[name] = ort.InferenceSession(
                str(path), sess_options=options, providers=providers
            )
        return _sessions[name]

    def _describe(self, image):
        assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
        _, _, height, width = self.descriptor.get_inputs()[0].shape
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        image = image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return self.descriptor.run(None, {"image": image})[0]

    def _forward(self, data):
        image0, image1 = data["image0_orig"], data["image1_orig"]
        # Keypoints are in the preprocessed image frame; move them to the
        # original frame FlexGlue expects (corner convention, center of the
        # top-left pixel at (0.5, 0.5)).
        kpts0 = (data["keypoints0"][0].cpu().numpy() + 0.5) * (
            np.asarray(data["original_size0"]) / np.asarray(data["size0"])
        )
        kpts1 = (data["keypoints1"][0].cpu().numpy() + 0.5) * (
            np.asarray(data["original_size1"]) / np.asarray(data["size1"])
        )
        if len(kpts0) == 0 or len(kpts1) == 0:
            return {
                "matches0": torch.full((1, len(kpts0)), -1, dtype=torch.long),
                "matching_scores0": torch.zeros((1, len(kpts0))),
            }
        feeds = {
            "features0": self._describe(image0),
            "features1": self._describe(image1),
            "kpts0": kpts0[None].astype(np.float32),
            "kpts1": kpts1[None].astype(np.float32),
            "image_size0": np.array([image0.shape[1::-1]], dtype=np.float32),
            "image_size1": np.array([image1.shape[1::-1]], dtype=np.float32),
            "filter_threshold": np.array(self.conf["match_threshold"], np.float32),
            "max_distance_px": np.array(self.conf["max_distance_px"], np.float32),
            "min_covisibility": np.array(self.conf["min_covisibility"], np.float32),
        }
        matches0, mscores0, _ = self.matcher.run(None, feeds)
        return {
            "matches0": torch.from_numpy(matches0),
            "matching_scores0": torch.from_numpy(mscores0),
        }
