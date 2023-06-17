import numpy as np
import torch
from dkm.utils import *
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

class Megadepth1500Benchmark:
    def __init__(self, data_root="data/megadepth", scene_names = None) -> None:
        if scene_names is None:
            self.scene_names = [
                "0015_0.1_0.3.npz",
                "0015_0.3_0.5.npz",
                "0022_0.1_0.3.npz",
                "0022_0.3_0.5.npz",
                "0022_0.5_0.7.npz",
            ]
        else:
            self.scene_names = scene_names
        self.scenes = [
            np.load(f"{data_root}/{scene}", allow_pickle=True)
            for scene in self.scene_names
        ]
        self.data_root = data_root

    def benchmark(self, model):
        with torch.no_grad():
            data_root = self.data_root
            tot_e_t, tot_e_R, tot_e_pose = [], [], []
            for scene_ind in range(len(self.scenes)):
                scene = self.scenes[scene_ind]
                pairs = scene["pair_infos"]
                intrinsics = scene["intrinsics"]
                poses = scene["poses"]
                im_paths = scene["image_paths"]
                pair_inds = range(len(pairs))
                for pairind in tqdm(pair_inds):
                    idx1, idx2 = pairs[pairind][0]
                    K1 = intrinsics[idx1].copy()
                    T1 = poses[idx1].copy()
                    R1, t1 = T1[:3, :3], T1[:3, 3]
                    K2 = intrinsics[idx2].copy()
                    T2 = poses[idx2].copy()
                    R2, t2 = T2[:3, :3], T2[:3, 3]
                    R, t = compute_relative_pose(R1, t1, R2, t2)
                    im1_path = f"{data_root}/{im_paths[idx1]}"
                    im2_path = f"{data_root}/{im_paths[idx2]}"
                    im1 = Image.open(im1_path)
                    w1, h1 = im1.size
                    im2 = Image.open(im2_path)
                    w2, h2 = im2.size
                    scale1 = 1200 / max(w1, h1)
                    scale2 = 1200 / max(w2, h2)
                    w1, h1 = scale1 * w1, scale1 * h1
                    w2, h2 = scale2 * w2, scale2 * h2
                    K1[:2] = K1[:2] * scale1
                    K2[:2] = K2[:2] * scale2
                    dense_matches, dense_certainty = model.match(im1_path, im2_path)
                    sparse_matches,_ = model.sample(
                        dense_matches, dense_certainty, 5000
                    )
                    kpts1 = sparse_matches[:, :2]
                    kpts1 = (
                        torch.stack(
                            (
                                w1 * (kpts1[:, 0] + 1) / 2,
                                h1 * (kpts1[:, 1] + 1) / 2,
                            ),
                            axis=-1,
                        )
                    )
                    kpts2 = sparse_matches[:, 2:]
                    kpts2 = (
                        torch.stack(
                            (
                                w2 * (kpts2[:, 0] + 1) / 2,
                                h2 * (kpts2[:, 1] + 1) / 2,
                            ),
                            axis=-1,
                        )
                    )
                    for _ in range(5):
                        shuffling = np.random.permutation(np.arange(len(kpts1)))
                        kpts1 = kpts1[shuffling]
                        kpts2 = kpts2[shuffling]
                        try:
                            norm_threshold = 0.5 / (
                            np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2])))
                            R_est, t_est, mask = estimate_pose(
                                kpts1.cpu().numpy(),
                                kpts2.cpu().numpy(),
                                K1,
                                K2,
                                norm_threshold,
                                conf=0.99999,
                            )
                            T1_to_2_est = np.concatenate((R_est, t_est), axis=-1)  #
                            e_t, e_R = compute_pose_error(T1_to_2_est, R, t)
                            e_pose = max(e_t, e_R)
                        except Exception as e:
                            print(repr(e))
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
