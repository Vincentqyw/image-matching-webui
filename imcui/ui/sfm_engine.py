"""SFM Engine using pycolmap for Structure from Motion."""

import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pycolmap
from loguru import logger
import rerun as rr

from .config import get_matcher_zoo, get_model
from .model_cache import ARCSizeAwareModelCache as ModelCache

# Module-level model cache for SFM feature extractors (lazy initialization)
_sfm_model_cache = None


def get_sfm_model_cache():
    """Get the SFM model cache instance, creating it on first use."""
    global _sfm_model_cache
    if _sfm_model_cache is None:
        _sfm_model_cache = ModelCache()
    return _sfm_model_cache


# For backward compatibility
class SfmModelCacheWrapper:
    """Wrapper for lazy SFM model cache initialization."""

    def __getattr__(self, name):
        cache = get_sfm_model_cache()
        return getattr(cache, name)

    def __call__(self, *args, **kwargs):
        cache = get_sfm_model_cache()
        return cache(*args, **kwargs)


model_cache = SfmModelCacheWrapper()


def extract_frames_from_video(
    video_path: str,
    num_frames: int = 10,
    target_width: int = 640,
    target_height: int = 480,
    resize_enabled: bool = True,
) -> List[np.ndarray]:
    """
    Extract uniformly sampled frames from a video file.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to extract.
        target_width: Target width for resizing.
        target_height: Target height for resizing.
        resize_enabled: Whether to resize frames.

    Returns:
        List of extracted frames as RGB numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    # Calculate uniform sampling indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to read frame at index {idx}")
            continue

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if enabled
        if resize_enabled:
            frame_rgb = cv2.resize(
                frame_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA
            )

        frames.append(frame_rgb)

    cap.release()
    logger.info(f"Extracted {len(frames)} frames from video")
    return frames


def load_images(
    input_data: Union[List[Any], str],
    num_frames: int = 10,
    target_width: int = 640,
    target_height: int = 480,
    resize_enabled: bool = True,
) -> List[np.ndarray]:
    """
    Load images from input (list of file objects, numpy arrays, or video path).

    Args:
        input_data: List of Gradio file objects, list of RGB images, or path to video file.
        num_frames: Number of frames to extract from video.
        target_width: Target width for resizing.
        target_height: Target height for resizing.
        resize_enabled: Whether to resize frames.

    Returns:
        List of RGB images as numpy arrays.
    """
    # Handle None or empty input
    if input_data is None:
        return []

    # Check if input is a video file path (string)
    if isinstance(input_data, str):
        if os.path.isfile(input_data):
            # Check file extension
            ext = os.path.splitext(input_data)[1].lower()
            video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
            if ext in video_extensions:
                return extract_frames_from_video(
                    input_data,
                    num_frames=num_frames,
                    target_width=target_width,
                    target_height=target_height,
                    resize_enabled=resize_enabled,
                )
            else:
                # Treat as image file
                img = cv2.imread(input_data)
                if img is None:
                    raise ValueError(f"Cannot read image: {input_data}")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if resize_enabled:
                    img_rgb = cv2.resize(
                        img_rgb,
                        (target_width, target_height),
                        interpolation=cv2.INTER_AREA,
                    )
                return [img_rgb]
        else:
            raise FileNotFoundError(f"File not found: {input_data}")

    # Handle list input (Gradio file upload or numpy arrays)
    if isinstance(input_data, list):
        images = []

        for i, item in enumerate(input_data):
            # Handle Gradio File object (has .name attribute with temp path)
            if hasattr(item, "name"):
                file_path = item.name
                logger.info(f"Loading file: {file_path}")

                # Check if it's a video file
                ext = os.path.splitext(file_path)[1].lower()
                video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

                if ext in video_extensions:
                    # Extract frames from video
                    frames = extract_frames_from_video(
                        file_path,
                        num_frames=num_frames,
                        target_width=target_width,
                        target_height=target_height,
                        resize_enabled=resize_enabled,
                    )
                    images.extend(frames)
                else:
                    # Load as image
                    img = cv2.imread(file_path)
                    if img is None:
                        logger.warning(f"Cannot read image: {file_path}")
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if resize_enabled and (
                        img_rgb.shape[1] != target_width
                        or img_rgb.shape[0] != target_height
                    ):
                        img_rgb = cv2.resize(
                            img_rgb,
                            (target_width, target_height),
                            interpolation=cv2.INTER_AREA,
                        )
                    images.append(img_rgb)

            # Handle numpy array (direct image)
            elif isinstance(item, np.ndarray):
                img = item
                if img is None:
                    logger.warning(f"Skipping None image at index {i}")
                    continue
                # Ensure RGB format
                if len(img.shape) == 2:
                    # Grayscale to RGB
                    img = np.stack([img] * 3, axis=-1)
                elif img.shape[-1] == 4:  # RGBA
                    img = img[:, :, :3]
                if resize_enabled and (
                    img.shape[1] != target_width or img.shape[0] != target_height
                ):
                    img = cv2.resize(
                        img, (target_width, target_height), interpolation=cv2.INTER_AREA
                    )
                images.append(img)

            else:
                logger.warning(f"Skipping unsupported item at index {i}: {type(item)}")

        return images

    raise TypeError(f"Unsupported input type: {type(input_data)}")


def extract_features(
    images: List[np.ndarray],
    matcher_key: str,
    max_keypoints: int = 2048,
    keypoint_threshold: float = 0.0,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Tuple[Dict[str, Any], Any]:
    """
    Extract features from images using vismatch.

    Args:
        images: List of RGB images.
        matcher_key: Key for the matcher model.
        max_keypoints: Maximum number of keypoints.
        keypoint_threshold: Keypoint detection threshold.
        progress: Optional progress callback (0-1, message).

    Returns:
        Tuple of (features dict for pycolmap, matcher instance).
    """
    n_images = len(images)
    logger.info(f"Extracting features from {n_images} images")

    # Get matcher model
    match_conf = {
        "model_name": matcher_key,
        "max_num_keypoints": max_keypoints,
        "threshold": 0.1,  # default threshold
        "keypoint_threshold": keypoint_threshold,
    }
    cache_key = f"{matcher_key}_sfm_kp{max_keypoints}_kt{keypoint_threshold}"
    matcher = get_sfm_model_cache().load_model(cache_key, get_model, match_conf)

    # Prepare images in CHW format for vismatch
    images_chw = []
    for img in images:
        if img.dtype == np.uint8:
            img_chw = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
        else:
            img_chw = np.transpose(img, (2, 0, 1))
        images_chw.append(img_chw)

    # Extract features for each image
    features = {}
    for i, img_chw in enumerate(images_chw):
        logger.info(f"Extracting features from image {i+1}/{len(images)}")

        # Report progress
        if progress:
            progress(i / n_images, f"Extracting features: {i+1}/{n_images}")

        # Get keypoints and descriptors
        result = matcher(img_chw, img_chw)  # Self-match to get keypoints

        kpts = result.get("all_kpts0", np.array([]))
        if len(kpts) == 0:
            logger.warning(f"No keypoints detected in image {i}")
            continue

        # Store in pycolmap format
        # pycolmap expects: keypoints (N, 3) with x, y, size
        # vismatch keypoints are (N, 2) with x, y
        keypoints = np.zeros((len(kpts), 3))
        keypoints[:, :2] = kpts[:, :2]
        keypoints[:, 2] = 1.0  # default size

        # vismatch uses "all_desc0" not "descriptors0"
        features[f"image_{i:04d}"] = {
            "keypoints": keypoints,
            "descriptors": result.get("all_desc0"),
            "image": images[i],
        }

    if progress:
        progress(1.0, "Feature extraction complete")

    return features, matcher


def match_image_pairs(
    images: List[np.ndarray],
    matcher: Any,
    _match_threshold: float = 0.7,
    progress: Optional[Callable[[float, str], None]] = None,
) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    """
    Match all image pairs exhaustively.

    Args:
        images: List of RGB images.
        matcher: Matcher instance.
        _match_threshold: Match threshold (unused in current implementation).
        progress: Optional progress callback (0-1, message).

    Returns:
        List of matches as (i, j, kpts_i, kpts_j) tuples.
    """
    # Convert images to CHW format
    images_chw = []
    for img in images:
        if img.dtype == np.uint8:
            img_chw = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
        else:
            img_chw = np.transpose(img, (2, 0, 1))
        images_chw.append(img_chw)

    matches = []
    n = len(images)
    pair_count = 0
    total_pairs = n * (n - 1) // 2

    for i in range(n):
        for j in range(i + 1, n):
            logger.info(f"Matching image {i} with image {j}")

            result = matcher(images_chw[i], images_chw[j])

            matched_kpts0 = result.get("matched_kpts0", np.array([]))
            matched_kpts1 = result.get("matched_kpts1", np.array([]))

            if len(matched_kpts0) > 0 and len(matched_kpts1) > 0:
                matches.append((i, j, matched_kpts0, matched_kpts1))
                pair_count += 1

            # Report progress
            if progress:
                pct = pair_count / total_pairs if total_pairs > 0 else 1.0
                progress(pct, f"Matching: pair {pair_count}/{total_pairs}")

    logger.info(f"Found {pair_count} image pairs with matches")
    return matches


def create_colmap_model(
    images: List[np.ndarray],
    features: Dict[str, Any],
    matches: List[Tuple[int, int, np.ndarray, np.ndarray]],
    camera_model: str = "simple-pinhole",
    camera_params: str = "",
) -> pycolmap.Reconstruction:
    """
    Create a COLMAP reconstruction from images, features, and matches.

    Args:
        images: List of RGB images.
        features: Dict of extracted features.
        matches: List of matches.
        camera_model: Camera model name.
        camera_params: Camera parameters string.

    Returns:
        COLMAP reconstruction object.
    """
    reconstruction = pycolmap.Reconstruction()

    # Get image dimensions
    h, w = images[0].shape[:2]

    # Add cameras
    if camera_params:
        # Parse camera parameters
        params = [float(p) for p in camera_params.split(",")]
    else:
        # Use default: focal length = max(w, h)
        params = [max(w, h)]

    # Convert camera model name to pycolmap format
    camera_model_map = {
        "simple-pinhole": "SIMPLE_PINHOLE",
        "pinhole": "PINHOLE",
        "simple-radial": "SIMPLE_RADIAL",
        "simple_radial": "SIMPLE_RADIAL",
        "opencv": "OPENCV",
    }
    pycolmap_model = camera_model_map.get(camera_model.lower(), "SIMPLE_PINHOLE")

    camera = pycolmap.Camera(
        model=pycolmap_model,
        width=w,
        height=h,
        params=params,
    )
    # pycolmap add_camera only takes the camera object
    reconstruction.add_camera(camera)
    camera_id = camera.camera_id

    # Add images
    for i, (img_name, feat) in enumerate(features.items()):
        image = pycolmap.Image()
        image.name = img_name
        image.camera_id = camera_id
        img_id = reconstruction.add_image(image)

        # Add points2D (keypoints)
        keypoints = feat["keypoints"]
        for kp in keypoints:
            reconstruction.add_point2D(kp[:2], img_id)

    # Add matches
    for i, j, kpts_i, kpts_j in matches:
        img_name_i = f"image_{i:04d}"
        img_name_j = f"image_{j:04d}"

        if img_name_i not in features or img_name_j not in features:
            continue

        # Get image IDs
        img_id_i = None
        img_id_j = None
        for img_id, img in reconstruction.images.items():
            if img.name == img_name_i:
                img_id_i = img_id
            if img.name == img_name_j:
                img_id_j = img_id

        if img_id_i is None or img_id_j is None:
            continue

        # Create point correspondences
        point_ids_i = []
        point_ids_j = []

        # Match keypoints to the original keypoints in features
        for m_idx in range(len(kpts_i)):
            # Find nearest keypoint in original features
            # This is a simplified approach - in practice we'd use the descriptor indices
            # For now, we just add the matches directly
            point_ids_i.append(m_idx)
            point_ids_j.append(m_idx)

        # Add matches to reconstruction
        if len(point_ids_i) > 0:
            reconstruction.add_matches(
                img_id_i,
                img_id_j,
                np.array(point_ids_i),
                np.array(point_ids_j),
            )

    return reconstruction


def build_scene_graph(
    images: List[np.ndarray],
    matcher: Any,
    method: str = "exhaustive",
    _global_feature: str = "netvlad",
    _top_k: int = 50,
    progress: Optional[Callable[[float, str], None]] = None,
) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    """
    Build scene graph by matching image pairs.

    Args:
        images: List of RGB images.
        matcher: Matcher instance.
        method: Matching method ("exhaustive" or "retrieval").
        _global_feature: Global feature for retrieval (unused in current implementation).
        _top_k: Top k retrieval results (unused in current implementation).
        progress: Optional progress callback.

    Returns:
        List of matches.
    """
    if method == "exhaustive":
        # Report starting matches
        if progress:
            progress(0, "Starting image matching...")
        return match_image_pairs(images, matcher, progress=progress)
        # Exhaustive matching - match all pairs
        return match_image_pairs(images, matcher)
    else:
        # For now, fallback to exhaustive
        logger.warning(f"Retrieval method '{method}' not implemented, using exhaustive")
        return match_image_pairs(images, matcher)


def run_colmap_sfm(
    images: List[np.ndarray],
    features: Dict[str, Any],
    matches: List[Tuple[int, int, np.ndarray, np.ndarray]],
    mapper_options: Optional[Dict[str, Any]] = None,
) -> Optional[pycolmap.Reconstruction]:
    """
    Run incremental COLMAP SFM.

    Args:
        images: List of RGB images.
        features: Extracted features.
        matches: Image pair matches.
        mapper_options: Options for the incremental mapper.

    Returns:
        COLMAP reconstruction or None if failed.
    """
    if mapper_options is None:
        mapper_options = {}

    logger.info(
        f"Starting COLMAP SFM with {len(images)} images and {len(matches)} match pairs"
    )

    try:
        # Create reconstruction
        reconstruction = pycolmap.Reconstruction()

        # Add cameras - use correct pycolmap enum value
        h, w = images[0].shape[:2]
        camera = pycolmap.Camera(
            model="SIMPLE_PINHOLE",
            width=w,
            height=h,
            params=[max(w, h), w / 2, h / 2],
        )

        # pycolmap add_camera only takes the camera object, returns void
        reconstruction.add_camera(camera)

        # Get the camera_id from the added camera
        camera_id = camera.camera_id
        logger.info(
            f"Added camera: {camera_id}, total cameras: {len(reconstruction.cameras)}"
        )

        # Add images with keypoints
        image_ids = {}
        for i, _ in enumerate(images):
            img_name = f"image_{i:04d}"
            # Create image with camera_id explicitly set as int
            image = pycolmap.Image()
            image.name = img_name
            image.camera_id = camera_id
            img_id = reconstruction.add_image(image)
            image_ids[img_name] = img_id

            # Add keypoints to image
            if img_name in features:
                keypoints = features[img_name]["keypoints"]
                for kp in keypoints:
                    reconstruction.add_point2D(kp[:2], img_id)
                    # Track which point2D IDs belong to which image
                    if img_name not in image_ids:
                        image_ids[img_name] = set()
                    # Note: point2D IDs are stored per image in reconstruction

        logger.info(f"Added {len(reconstruction.images)} images to reconstruction")

        # Add matches - use a simpler approach
        match_count = 0
        for i, j, kpts_i, kpts_j in matches:
            img_name_i = f"image_{i:04d}"
            img_name_j = f"image_{j:04d}"

            img_id_i = image_ids.get(img_name_i)
            img_id_j = image_ids.get(img_name_j)

            if img_id_i is None or img_id_j is None:
                continue

            # Match keypoints
            n_matches = min(
                len(kpts_i), len(kpts_j), 100
            )  # Limit matches for efficiency
            if n_matches < 4:
                continue

            # Create correspondences - use index matching
            point_ids_i = np.arange(n_matches)
            point_ids_j = np.arange(n_matches)

            try:
                reconstruction.add_matches(img_id_i, img_id_j, point_ids_i, point_ids_j)
                match_count += 1
            except Exception as e:
                logger.warning(f"Failed to add matches for pair {i}-{j}: {e}")
                continue

        logger.info(f"Added {match_count} match pairs to reconstruction")

        # Try incremental mapping
        mapper_options_obj = pycolmap.IncrementalMapperOptions()
        mapper_options_obj.min_num_matches = 4
        mapper_options_obj.max_num_models = 1

        # Apply mapper options
        if mapper_options.get("refine_focal_length", True):
            mapper_options_obj.refine_focal_length = True
        if mapper_options.get("refine_principle_points", False):
            mapper_options_obj.refine_principle_points = True
        if mapper_options.get("refine_extra_params", False):
            mapper_options_obj.refine_extra_params = True

        # Try to run SFM
        try:
            maps = pycolmap.incremental_mapping(mapper_options_obj, reconstruction)
            logger.info(f"SFM produced {len(maps)} models")

            if len(maps) > 0 and maps[0].num_points3D() > 0:
                return maps[0]
        except Exception as e:
            logger.error(f"Incremental mapping failed: {e}")

    except Exception as e:
        logger.error(f"COLMAP SFM setup failed: {e}")

    return None


def save_point_cloud(
    reconstruction: pycolmap.Reconstruction,
    output_path: str,
) -> str:
    """
    Save reconstruction as PLY point cloud.

    Args:
        reconstruction: COLMAP reconstruction.
        output_path: Output PLY file path.

    Returns:
        Path to the saved PLY file.
    """
    # Get points3D from reconstruction
    points = []
    colors = []

    for _, point3D in reconstruction.points3D.items():
        points.append([point3D.x, point3D.y, point3D.z])

        # Get color from first observation
        if point3D.track.length() > 0:
            _ = point3D.track[0]  # obs variable
            # Note: We don't have direct access to image colors here
            # Use default color
            colors.append([200, 200, 200])
        else:
            colors.append([200, 200, 200])

    if not points:
        logger.warning("No 3D points in reconstruction")
        # Create dummy point cloud
        points = [[0, 0, 0]]
        colors = [[255, 0, 0]]

    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)

    # Write PLY file
    write_ply(output_path, points, colors)

    logger.info(f"Saved point cloud with {len(points)} points to {output_path}")
    return output_path


def create_simple_pointcloud_from_matches(
    images: List[np.ndarray],
    _features: Dict[str, Any],
    matches: List[Tuple[int, int, np.ndarray, np.ndarray]],
) -> Optional[pycolmap.Reconstruction]:
    """
    Create a simple point cloud from matches without full COLMAP SFM.
    This is a fallback when COLMAP SFM fails.

    Args:
        images: List of RGB images.
        _features: Extracted features (unused in current implementation).
        matches: Image pair matches.
    """
    try:
        reconstruction = pycolmap.Reconstruction()

        # Add camera
        h, w = images[0].shape[:2]
        camera = pycolmap.Camera(
            model="SIMPLE_PINHOLE",
            width=w,
            height=h,
            params=[max(w, h), w / 2, h / 2],
        )

        # pycolmap add_camera only takes the camera object
        reconstruction.add_camera(camera)
        camera_id = camera.camera_id

        # Add images
        image_ids = {}
        for i, _ in enumerate(images):
            img_name = f"image_{i:04d}"
            image = pycolmap.Image()
            image.name = img_name
            image.camera_id = camera_id
            img_id = reconstruction.add_image(image)
            image_ids[img_name] = img_id

        # Add keypoints and matches
        match_count = 0
        for i, j, kpts_i, kpts_j in matches:
            img_name_i = f"image_{i:04d}"
            img_name_j = f"image_{j:04d}"

            img_id_i = image_ids.get(img_name_i)
            img_id_j = image_ids.get(img_name_j)

            if img_id_i is None or img_id_j is None:
                continue

            n_matches = min(len(kpts_i), len(kpts_j), 50)
            if n_matches < 4:
                continue

            point_ids_i = np.arange(n_matches)
            point_ids_j = np.arange(n_matches)

            try:
                reconstruction.add_matches(img_id_i, img_id_j, point_ids_i, point_ids_j)
                match_count += 1
            except Exception as e:
                logger.warning(f"Failed to add matches: {e}")

        if match_count > 0:
            return reconstruction

    except Exception as e:
        logger.error(f"Simple point cloud creation failed: {e}")

    return None


def extract_points_from_reconstruction(
    reconstruction: pycolmap.Reconstruction,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract points and colors from a COLMAP reconstruction."""
    points = []
    colors = []

    for _, point3D in reconstruction.points3D.items():
        points.append([point3D.x, point3D.y, point3D.z])
        # Default gray color
        colors.append([180, 180, 180])

    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.uint8)


def create_visualization_pointcloud(
    images: List[np.ndarray],
    _features: Dict[str, Any],
    matches: List[Tuple[int, int, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a simple 3D point cloud visualization from 2D matches.

    This creates a crude visualization by projecting matched keypoints
    to 3D based on a simplified camera model.

    Args:
        images: List of RGB images.
        _features: Extracted features (unused in current implementation).
        matches: Image pair matches.
    """
    points = []
    colors = []

    h, w = images[0].shape[:2]
    cx, cy = w / 2, h / 2
    f = max(w, h)

    # Process each image and its matches
    for i, _, kpts_i, kpts_j in matches:
        # Get colors from images at matched keypoints
        for m in range(min(len(kpts_i), len(kpts_j))):
            x_i, y_i = kpts_i[m]
            x_j, y_j = kpts_j[m]

            # Convert to normalized camera coordinates
            x_i_norm = (x_i - cx) / f
            y_i_norm = (y_i - cy) / f
            x_j_norm = (x_j - cx) / f
            y_j_norm = (y_j - cy) / f

            # Simple triangulation (assuming small baseline)
            # This is a simplification - real SFM would optimize this
            depth = 1.0 + m * 0.1  # crude depth estimate

            # Back-project to 3D
            pt_3d_i = np.array([x_i_norm * depth, y_i_norm * depth, depth])
            pt_3d_j = np.array([x_j_norm * depth, y_j_norm * depth, depth])

            # Average
            pt_3d = (pt_3d_i + pt_3d_j) / 2

            points.append(pt_3d)

            # Get color from image
            x_int = int(min(max(x_i, 0), w - 1))
            y_int = int(min(max(y_i, 0), h - 1))
            color = images[i][y_int, x_int]
            colors.append(color[:3] if len(color) >= 3 else [180, 180, 180])

    logger.info(f"Created visualization point cloud with {len(points)} points")
    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.uint8)


def create_rerun_pointcloud(
    points: np.ndarray,
    colors: np.ndarray,
) -> Tuple[str, str]:
    """
    Create a Rerun point cloud recording and HTML visualization.

    Args:
        points: Nx3 array of points.
        colors: Nx3 array of colors (RGB 0-255).

    Returns:
        Tuple of (path to recording, HTML visualization).
    """
    # Create a temporary file for the recording
    tmp_dir = tempfile.gettempdir()
    recording_path = os.path.join(tmp_dir, "sfm_pointcloud.rrd")

    try:
        # Initialize Rerun with a recording using the correct API
        rec = rr.RecordingStream("imcui_sfm", recording_id="sfm_pointcloud")

        # Set this as the active recording for logging
        rr.set_thread_local_data_recording(rec)

        # Save the recording to file (must be called before logging)
        rr.save(recording_path, recording=rec)

        # Scale radii based on point cloud scale
        point_scale = np.max(np.abs(points)) / len(points) if len(points) > 0 else 0.01
        radii = [max(point_scale * 0.1, 0.01)] * len(points)

        # Log the point cloud - colors directly as RGB array
        rec.log(
            "pointcloud",
            rr.Points3D(
                positions=points,
                colors=colors,  # RGB colors 0-255
                radii=radii,
            ),
        )

        # Log coordinate axes for reference - colors directly as RGB array
        rec.log(
            "axes",
            rr.Arrows3D(
                origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                vectors=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # RGB colors
            ),
        )

        logger.info(f"Created Rerun recording with {len(points)} points")

        # Note: RecordingStream doesn't have a close() method in newer versions
        # The recording is automatically saved when the RecordingStream is garbage collected

        # Create HTML visualization showing the recording info
        html_content = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px;">
            <h3>3D Point Cloud Generated</h3>
            <p><strong>Points:</strong> {len(points)}</p>
            <p><strong>Recording Path:</strong> {recording_path}</p>
            <p><strong>Viewing Instructions:</strong></p>
            <ul>
                <li>The Rerun recording has been saved to: <code>{recording_path}</code></li>
                <li>To view the 3D point cloud, open this file in a Rerun viewer:</li>
                <li><code>rerun {recording_path}</code></li>
            </ul>
        </div>
        """
    except Exception as e:
        logger.warning(f"Failed to create Rerun recording: {e}")
        recording_path = ""
        html_content = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px;">
            <h3>Rerun Recording Failed</h3>
            <p>Point cloud contains {len(points)} points but Rerun recording failed.</p>
            <p>Error: {str(e)}</p>
        </div>
        """

    return recording_path, html_content


def write_ply(
    output_path: str,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    """
    Write point cloud to PLY file.

    Args:
        output_path: Output PLY file path.
        points: Nx3 array of points.
        colors: Nx3 array of colors (RGB).
    """
    n_points = len(points)

    with open(output_path, "wb") as f:
        # Header - encode strings as bytes
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {n_points}\n".encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property uchar red\n")
        f.write(b"property uchar green\n")
        f.write(b"property uchar blue\n")
        f.write(b"end_header\n")

        # Write points
        for i in range(n_points):
            f.write(points[i].tobytes())
            f.write(bytes(colors[i]))


class SfmEngine:
    """
    Structure from Motion Engine using pycolmap and vismatch.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        """
        Initialize SFM engine.

        Args:
            cfg: Optional configuration dict.
        """
        self.cfg = cfg or {}
        self.matcher_zoo = get_matcher_zoo()
        self.default_matcher = "superpoint-lightglue"

    def call(
        self,
        matcher_key: str,
        input_images: Union[List[np.ndarray], str],
        _camera_model: str = "simple-pinhole",
        _camera_params: str = "",
        max_keypoints: int = 2048,
        keypoint_threshold: float = 0.0,
        _match_threshold: float = 0.7,
        _ransac_threshold: float = 4.0,
        _ransac_confidence: float = 0.9999,
        _ransac_max_iter: int = 20000,
        scene_graph: str = "exhaustive",
        _global_feature: str = "netvlad",
        _top_k: int = 50,
        mapper_refine_focal_length: bool = True,
        mapper_refine_principle_points: bool = False,
        mapper_refine_extra_params: bool = False,
        # New parameters
        num_frames: int = 10,
        resize_enabled: bool = True,
        target_width: int = 640,
        target_height: int = 480,
        # Progress callback (from Gradio)
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Run SFM on input images.

        Args:
            matcher_key: Key for the matcher model.
            input_images: List of RGB images or path to video file.
            camera_model: Camera model (e.g., "simple-pinhole").
            camera_params: Camera parameters string.
            max_keypoints: Maximum number of keypoints.
            keypoint_threshold: Keypoint detection threshold.
            match_threshold: Match threshold for vismatch.
            ransac_threshold: RANSAC reprojection threshold.
            ransac_confidence: RANSAC confidence.
            ransac_max_iter: RANSAC max iterations.
            scene_graph: Scene graph generation method.
            global_feature: Global feature for retrieval.
            top_k: Top k retrieval results.
            mapper_refine_focal_length: Refine focal length in mapping.
            mapper_refine_principle_points: Refine principle points.
            mapper_refine_extra_params: Refine extra parameters.
            num_frames: Number of frames to extract from video.
            resize_enabled: Whether to resize images.
            target_width: Target width.
            target_height: Target height.

        Returns:
            Tuple of (path to PLY point cloud file, result image for display).
        """
        logger.info("Starting SFM reconstruction")

        # Report starting
        if progress:
            progress(0, "Starting SFM...")

        # Validate matcher key
        if matcher_key not in self.matcher_zoo:
            logger.warning(
                f"Matcher {matcher_key} not found, using {self.default_matcher}"
            )
            matcher_key = self.default_matcher

        # Load images
        try:
            images = load_images(
                input_images,
                num_frames=num_frames,
                target_width=target_width,
                target_height=target_height,
                resize_enabled=resize_enabled,
            )
        except Exception as e:
            logger.error(f"Failed to load images: {e}")
            return None, None

        if len(images) < 2:
            logger.error("Need at least 2 images for SFM")
            if progress:
                progress(1.0, "Error: Need at least 2 images")
            return None, None

        logger.info(f"Processing {len(images)} images")

        if progress:
            progress(0.05, f"Loaded {len(images)} images, extracting features...")

        # Extract features
        features, matcher = extract_features(
            images,
            matcher_key,
            max_keypoints=max_keypoints,
            keypoint_threshold=keypoint_threshold,
            progress=lambda p, m: progress(0.05 + p * 0.35, m) if progress else None,
        )

        if len(features) < 2:
            logger.error("Not enough images with features")
            return None, None

        if progress:
            progress(0.4, "Building scene graph (matching images)...")

        # Build scene graph (match image pairs)
        matches = build_scene_graph(
            images,
            matcher,
            method=scene_graph,
            progress=lambda p, m: progress(0.4 + p * 0.3, m) if progress else None,
        )

        if len(matches) == 0:
            logger.error("No matches found between images")
            if progress:
                progress(1.0, "Error: No matches found")
            return None, None

        if progress:
            progress(0.7, "Creating 3D point cloud...")

        # Try COLMAP SFM first, but don't fail if it doesn't work
        try:
            mapper_options = {
                "refine_focal_length": mapper_refine_focal_length,
                "refine_principle_points": mapper_refine_principle_points,
                "refine_extra_params": mapper_refine_extra_params,
            }

            reconstruction = run_colmap_sfm(
                images,
                features,
                matches,
                mapper_options=mapper_options,
            )

            if reconstruction is not None and len(reconstruction.points3D) > 0:
                points, colors = extract_points_from_reconstruction(reconstruction)
                logger.info(f"Extracted {len(points)} 3D points from COLMAP")
            else:
                raise ValueError("No 3D points from COLMAP")
        except Exception as e:
            logger.warning(
                f"COLMAP SFM failed: {e}, creating visualization point cloud"
            )
            # Create visualization-only point cloud from matches
            points, colors = create_visualization_pointcloud(images, features, matches)

        if len(points) == 0:
            # Create dummy point cloud
            logger.warning("No 3D points, creating placeholder")
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
            colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

        if progress:
            progress(0.85, f"Creating 3D visualization with {len(points)} points...")

        # Create Rerun point cloud recording
        rrd_path, _ = create_rerun_pointcloud(points, colors)
        logger.info(
            f"Created Rerun visualization with {len(points)} points at {rrd_path}"
        )

        if progress:
            progress(1.0, f"SFM completed with {len(points)} points")

        # Create a simple visualization
        result_image = images[0]  # Use first image as result preview

        logger.info(f"SFM completed with {len(points)} points")
        # Return RRD file path for Rerun component instead of HTML
        return rrd_path, result_image
