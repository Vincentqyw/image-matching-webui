import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).parents[1]))

from hloc import (
    extract_features,
    logger,
    match_features,
    pairs_from_retrieval,
    reconstruction,
    visualization,
)

try:
    import pycolmap
except ImportError:
    logger.warning("pycolmap not installed, some features may not work")

from ui.viz import fig2im


class SfmEngine:
    def __init__(self, cfg: Dict[str, Any] = None):
        self.cfg = cfg
        if "outputs" in cfg and Path(cfg["outputs"]):
            outputs = Path(cfg["outputs"])
            outputs.mkdir(parents=True, exist_ok=True)
        else:
            outputs = tempfile.mkdtemp()
        self.outputs = Path(outputs)

    def call(
        self,
        key: str,
        images: Path,
        camera_model: str,
        camera_params: List[float],
        max_keypoints: int,
        keypoint_threshold: float,
        match_threshold: float,
        ransac_threshold: int,
        ransac_confidence: float,
        ransac_max_iter: int,
        scene_graph: bool,
        global_feature: str,
        top_k: int = 10,
        mapper_refine_focal_length: bool = False,
        mapper_refine_principle_points: bool = False,
        mapper_refine_extra_params: bool = False,
    ):
        """
        Call a list of functions to perform feature extraction, matching, and reconstruction.

        Args:
            key (str): The key to retrieve the matcher and feature models.
            images (Path): The directory containing the images.
            outputs (Path): The directory to store the outputs.
            camera_model (str): The camera model.
            camera_params (List[float]): The camera parameters.
            max_keypoints (int): The maximum number of features.
            match_threshold (float): The match threshold.
            ransac_threshold (int): The RANSAC threshold.
            ransac_confidence (float): The RANSAC confidence.
            ransac_max_iter (int): The maximum number of RANSAC iterations.
            scene_graph (bool): Whether to compute the scene graph.
            global_feature (str): Whether to compute the global feature.
            top_k (int): The number of image-pair to use.
            mapper_refine_focal_length (bool): Whether to refine the focal length.
            mapper_refine_principle_points (bool): Whether to refine the principle points.
            mapper_refine_extra_params (bool): Whether to refine the extra parameters.

        Returns:
            Path: The directory containing the SfM results.
        """
        if len(images) == 0:
            logger.error(f"{images} does not exist.")

        temp_images = Path(tempfile.mkdtemp())
        # copy images
        logger.info(f"Copying images to {temp_images}.")
        for image in images:
            shutil.copy(image, temp_images)

        matcher_zoo = self.cfg["matcher_zoo"]
        model = matcher_zoo[key]
        match_conf = model["matcher"]
        match_conf["model"]["max_keypoints"] = max_keypoints
        match_conf["model"]["match_threshold"] = match_threshold

        feature_conf = model["feature"]
        feature_conf["model"]["max_keypoints"] = max_keypoints
        feature_conf["model"]["keypoint_threshold"] = keypoint_threshold

        # retrieval
        retrieval_name = self.cfg.get("retrieval_name", "netvlad")
        retrieval_conf = extract_features.confs[retrieval_name]

        mapper_options = {
            "ba_refine_extra_params": mapper_refine_extra_params,
            "ba_refine_focal_length": mapper_refine_focal_length,
            "ba_refine_principal_point": mapper_refine_principle_points,
            "ba_local_max_num_iterations": 40,
            "ba_local_max_refinements": 3,
            "ba_global_max_num_iterations": 100,
            # below 3 options are for individual/video data, for internet photos, they should be left
            # default
            "min_focal_length_ratio": 0.1,
            "max_focal_length_ratio": 10,
            "max_extra_param": 1e15,
        }

        sfm_dir = self.outputs / "sfm_{}".format(key)
        sfm_pairs = self.outputs / "pairs-sfm.txt"
        sfm_dir.mkdir(exist_ok=True, parents=True)

        # extract features
        retrieval_path = extract_features.main(
            retrieval_conf, temp_images, self.outputs
        )
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=top_k)

        feature_path = extract_features.main(feature_conf, temp_images, self.outputs)
        # match features
        match_path = match_features.main(
            match_conf, sfm_pairs, feature_conf["output"], self.outputs
        )
        # reconstruction
        already_sfm = False
        if sfm_dir.exists():
            try:
                model = pycolmap.Reconstruction(str(sfm_dir))
                already_sfm = True
            except ValueError:
                logger.info(f"sfm_dir not exists model: {sfm_dir}")
        if not already_sfm:
            model = reconstruction.main(
                sfm_dir,
                temp_images,
                sfm_pairs,
                feature_path,
                match_path,
                mapper_options=mapper_options,
            )

        vertices = []
        for point3D_id, point3D in model.points3D.items():
            vertices.append([point3D.xyz, point3D.color])

        model_3d = sfm_dir / "points3D.obj"
        with open(model_3d, "w") as f:
            for p, c in vertices:
                # Write vertex position
                f.write("v {} {} {}\n".format(p[0], p[1], p[2]))
                # Write vertex normal (color)
                f.write(
                    "vn {} {} {}\n".format(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
                )
        viz_2d = visualization.visualize_sfm_2d(
            model, temp_images, color_by="visibility", n=2, dpi=300
        )

        return model_3d, fig2im(viz_2d) / 255.0
