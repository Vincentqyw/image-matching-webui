import pickle
import h5py
import numpy as np
import torch
from dkm.utils import *
from PIL import Image
from tqdm import tqdm


class Yfcc100mBenchmark:
    def __init__(self, data_root="data/yfcc100m_test") -> None:
        self.scenes = [
            "buckingham_palace",
            "notre_dame_front_facade",
            "reichstag",
            "sacre_coeur",
        ]
        self.data_root = data_root

    def benchmark(self, model, r=2):
        model.train(False)
        with torch.no_grad():
            data_root = self.data_root
            meta_info = open(
                f"{data_root}/yfcc_test_pairs_with_gt.txt", "r"
            ).readlines()
            tot_e_t, tot_e_R, tot_e_pose = [], [], []
            for scene_ind in range(len(self.scenes)):
                scene = self.scenes[scene_ind]
                pairs = np.array(
                    pickle.load(
                        open(f"{data_root}/pairs/{scene}-te-1000-pairs.pkl", "rb")
                    )
                )
                scene_dir = f"{data_root}/yfcc100m/{scene}/test/"
                calibs = open(scene_dir + "calibration.txt", "r").read().split("\n")
                images = open(scene_dir + "images.txt", "r").read().split("\n")
                pair_inds = np.random.choice(
                    range(len(pairs)), size=len(pairs), replace=False
                )
                for pairind in tqdm(pair_inds):
                    idx1, idx2 = pairs[pairind]
                    params = meta_info[1000 * scene_ind + pairind].split()
                    rot1, rot2 = int(params[2]), int(params[3])
                    calib1 = h5py.File(scene_dir + calibs[idx1], "r")
                    K1, R1, t1, _, _ = get_pose(calib1)
                    calib2 = h5py.File(scene_dir + calibs[idx2], "r")
                    K2, R2, t2, _, _ = get_pose(calib2)

                    R, t = compute_relative_pose(R1, t1, R2, t2)
                    im1 = images[idx1]
                    im2 = images[idx2]
                    im1 = Image.open(scene_dir + im1).rotate(rot1 * 90, expand=True)
                    w1, h1 = im1.size
                    im2 = Image.open(scene_dir + im2).rotate(rot2 * 90, expand=True)
                    w2, h2 = im2.size
                    K1 = rotate_intrinsic(K1, rot1)
                    K2 = rotate_intrinsic(K2, rot2)

                    dense_matches, dense_certainty = model.match(im1, im2)
                    dense_certainty = dense_certainty ** (1 / r)
                    sparse_matches, sparse_confidence = model.sample(
                        dense_matches, dense_certainty, 10000
                    )
                    scale1 = 480 / min(w1, h1)
                    scale2 = 480 / min(w2, h2)
                    w1, h1 = scale1 * w1, scale1 * h1
                    w2, h2 = scale2 * w2, scale2 * h2
                    K1 = K1 * scale1
                    K2 = K2 * scale2

                    kpts1 = sparse_matches[:, :2]
                    kpts1 = np.stack(
                        (w1 * kpts1[:, 0] / 2, h1 * kpts1[:, 1] / 2), axis=-1
                    )
                    kpts2 = sparse_matches[:, 2:]
                    kpts2 = np.stack(
                        (w2 * kpts2[:, 0] / 2, h2 * kpts2[:, 1] / 2), axis=-1
                    )
                    try:
                        threshold = 1.0
                        norm_threshold = threshold / (
                            np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2]))
                        )
                        R_est, t_est, mask = estimate_pose(
                            kpts1,
                            kpts2,
                            K1[:2, :2],
                            K2[:2, :2],
                            norm_threshold,
                            conf=0.9999999,
                        )
                        T1_to_2 = np.concatenate((R_est, t_est), axis=-1)  #
                        e_t, e_R = compute_pose_error(T1_to_2, R, t)
                        e_pose = max(e_t, e_R)
                    except:
                        e_t, e_R = 90, 90
                        e_pose = max(e_t, e_R)
                    tot_e_t.append(e_t)
                    tot_e_R.append(e_R)
                    tot_e_pose.append(e_pose)
            tot_e_pose = np.array(tot_e_pose)
            thresholds = [5, 10, 20]
            auc = pose_auc(tot_e_pose, thresholds)
            acc_5 = (tot_e_pose < 5).mean()
            acc_10 = (tot_e_pose < 10).mean()
            acc_15 = (tot_e_pose < 15).mean()
            acc_20 = (tot_e_pose < 20).mean()
            map_5 = acc_5
            map_10 = np.mean([acc_5, acc_10])
            map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
            return {
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }
