from PIL import Image
import numpy as np

import os

from tqdm import tqdm
from dkm.utils import pose_auc
import cv2


class HpatchesHomogBenchmark:
    """Hpatches grid goes from [0,n-1] instead of [0.5,n-0.5]"""

    def __init__(self, dataset_path) -> None:
        seqs_dir = "hpatches-sequences-release"
        self.seqs_path = os.path.join(dataset_path, seqs_dir)
        self.seq_names = sorted(os.listdir(self.seqs_path))
        # Ignore seqs is same as LoFTR.
        self.ignore_seqs = set(
            [
                "i_contruction",
                "i_crownnight",
                "i_dc",
                "i_pencils",
                "i_whitebuilding",
                "v_artisans",
                "v_astronautis",
                "v_talent",
            ]
        )

    def convert_coordinates(self, query_coords, query_to_support, wq, hq, wsup, hsup):
        offset = 0.5  # Hpatches assumes that the center of the top-left pixel is at [0,0] (I think)
        query_coords = (
            np.stack(
                (
                    wq * (query_coords[..., 0] + 1) / 2,
                    hq * (query_coords[..., 1] + 1) / 2,
                ),
                axis=-1,
            )
            - offset
        )
        query_to_support = (
            np.stack(
                (
                    wsup * (query_to_support[..., 0] + 1) / 2,
                    hsup * (query_to_support[..., 1] + 1) / 2,
                ),
                axis=-1,
            )
            - offset
        )
        return query_coords, query_to_support

    def benchmark(self, model, model_name = None):
        n_matches = []
        homog_dists = []
        for seq_idx, seq_name in tqdm(
            enumerate(self.seq_names), total=len(self.seq_names)
        ):
            if seq_name in self.ignore_seqs:
                continue
            im1_path = os.path.join(self.seqs_path, seq_name, "1.ppm")
            im1 = Image.open(im1_path)
            w1, h1 = im1.size
            for im_idx in range(2, 7):
                im2_path = os.path.join(self.seqs_path, seq_name, f"{im_idx}.ppm")
                im2 = Image.open(im2_path)
                w2, h2 = im2.size
                H = np.loadtxt(
                    os.path.join(self.seqs_path, seq_name, "H_1_" + str(im_idx))
                )
                dense_matches, dense_certainty = model.match(
                    im1_path, im2_path
                )
                good_matches, _ = model.sample(dense_matches, dense_certainty, 5000)
                pos_a, pos_b = self.convert_coordinates(
                    good_matches[:, :2], good_matches[:, 2:], w1, h1, w2, h2
                )
                try:
                    H_pred, inliers = cv2.findHomography(
                        pos_a,
                        pos_b,
                        method = cv2.RANSAC,
                        confidence = 0.99999,
                        ransacReprojThreshold = 3 * min(w2, h2) / 480,
                    )
                except:
                    H_pred = None
                if H_pred is None:
                    H_pred = np.zeros((3, 3))
                    H_pred[2, 2] = 1.0
                corners = np.array(
                    [[0, 0, 1], [0, h1 - 1, 1], [w1 - 1, 0, 1], [w1 - 1, h1 - 1, 1]]
                )
                real_warped_corners = np.dot(corners, np.transpose(H))
                real_warped_corners = (
                    real_warped_corners[:, :2] / real_warped_corners[:, 2:]
                )
                warped_corners = np.dot(corners, np.transpose(H_pred))
                warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
                mean_dist = np.mean(
                    np.linalg.norm(real_warped_corners - warped_corners, axis=1)
                ) / (min(w2, h2) / 480.0)
                homog_dists.append(mean_dist)
        n_matches = np.array(n_matches)
        thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        auc = pose_auc(np.array(homog_dists), thresholds)
        return {
            "hpatches_homog_auc_3": auc[2],
            "hpatches_homog_auc_5": auc[4],
            "hpatches_homog_auc_10": auc[9],
        }
