"""
Some useful functions for dataset pre-processing
"""
import cv2
import numpy as np
import shapely.geometry as sg

from ..synthetic_util import get_line_map
from . import homographic_transforms as homoaug


def random_scaling(image, junctions, line_map, scale=1., h_crop=0, w_crop=0):
    H, W = image.shape[:2]
    H_scale, W_scale = round(H * scale), round(W * scale)

    # Nothing to do if the scale is too close to 1
    if H_scale == H and W_scale == W:
        return (image, junctions, line_map, np.ones([H, W], dtype=np.int))

    # Zoom-in => resize and random crop
    if scale >= 1.:
        image_big = cv2.resize(image, (W_scale, H_scale),
                               interpolation=cv2.INTER_LINEAR)
        # Crop the image
        image = image_big[h_crop:h_crop+H, w_crop:w_crop+W, ...]
        valid_mask = np.ones([H, W], dtype=np.int)

        # Process junctions
        junctions, line_map = process_junctions_and_line_map(
            h_crop, w_crop, H, W, H_scale, W_scale,
            junctions, line_map, "zoom-in")
    # Zoom-out => resize and pad
    else:
        image_shape_raw = image.shape
        image_small = cv2.resize(image, (W_scale, H_scale),
                                 interpolation=cv2.INTER_AREA)
        # Decide the pasting location
        h_start = round((H - H_scale) / 2)
        w_start = round((W - W_scale) / 2)
        # Paste the image to the middle
        image = np.zeros(image_shape_raw, dtype=np.float)
        image[h_start:h_start+H_scale,
              w_start:w_start+W_scale, ...] = image_small
        valid_mask = np.zeros([H, W], dtype=np.int)
        valid_mask[h_start:h_start+H_scale, w_start:w_start+W_scale] = 1

        # Process the junctions
        junctions, line_map = process_junctions_and_line_map(
            h_start, w_start, H, W, H_scale, W_scale,
            junctions, line_map, "zoom-out")

    return image, junctions, line_map, valid_mask


def process_junctions_and_line_map(h_start, w_start, H, W, H_scale, W_scale,
                                   junctions, line_map, mode="zoom-in"):
    if mode == "zoom-in":
        junctions[:, 0] = junctions[:, 0] * H_scale / H
        junctions[:, 1] = junctions[:, 1] * W_scale / W
        line_segments = homoaug.convert_to_line_segments(junctions, line_map)
        # Crop segments to the new boundaries
        line_segments_new = np.zeros([0, 4])
        image_poly = sg.Polygon(
            [[w_start, h_start],
            [w_start+W, h_start],
            [w_start+W, h_start+H],
            [w_start, h_start+H]
            ])
        for idx in range(line_segments.shape[0]):
            # Get the line segment
            seg_raw = line_segments[idx, :]   # in HW format.
            # Convert to shapely line (flip to xy format)
            seg = sg.LineString([np.flip(seg_raw[:2]), 
                                np.flip(seg_raw[2:])])
            # The line segment is just inside the image.
            if seg.intersection(image_poly) == seg:
                line_segments_new = np.concatenate(
                    (line_segments_new, seg_raw[None, ...]), axis=0)
            # Intersect with the image.
            elif seg.intersects(image_poly):
                # Check intersection
                try:
                    p = np.array(
                        seg.intersection(image_poly).coords).reshape([-1, 4])
                # If intersect at exact one point, just continue.
                except:
                    continue
                segment = np.concatenate([np.flip(p[0, :2]), np.flip(p[0, 2:],
                                         axis=0)])[None, ...]
                line_segments_new = np.concatenate(
                    (line_segments_new, segment), axis=0)
            else:
                continue
        line_segments_new = (np.round(line_segments_new)).astype(np.int)
        # Filter segments with 0 length
        segment_lens = np.linalg.norm(
            line_segments_new[:, :2] - line_segments_new[:, 2:], axis=-1)
        seg_mask = segment_lens != 0
        line_segments_new = line_segments_new[seg_mask, :]
        # Convert back to junctions and line_map
        junctions_new = np.concatenate(
            (line_segments_new[:, :2], line_segments_new[:, 2:]), axis=0)
        if junctions_new.shape[0] == 0:
            junctions_new = np.zeros([0, 2])
            line_map = np.zeros([0, 0])
        else:
            junctions_new = np.unique(junctions_new, axis=0)
            # Generate line map from points and segments
            line_map = get_line_map(junctions_new,
                                    line_segments_new).astype(np.int)
        junctions_new[:, 0] -= h_start
        junctions_new[:, 1] -= w_start
        junctions = junctions_new
    elif mode == "zoom-out":
        # Process the junctions
        junctions[:, 0] = (junctions[:, 0] * H_scale / H) + h_start
        junctions[:, 1] = (junctions[:, 1] * W_scale / W) + w_start
    else:
        raise ValueError("[Error] unknown mode...")

    return junctions, line_map
