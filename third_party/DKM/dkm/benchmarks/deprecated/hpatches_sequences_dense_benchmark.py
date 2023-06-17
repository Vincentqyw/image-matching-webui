from PIL import Image
import numpy as np

import os

import torch
from tqdm import tqdm

from dkm.utils import *


class HpatchesDenseBenchmark:
    """WARNING: HPATCHES grid goes from [0,n-1] instead of [0.5,n-0.5]"""

    def __init__(self, dataset_path) -> None:
        seqs_dir = "hpatches-sequences-release"
        self.seqs_path = os.path.join(dataset_path, seqs_dir)
        self.seq_names = sorted(os.listdir(self.seqs_path))

    def convert_coordinates(self, query_coords, query_to_support, wq, hq, wsup, hsup):
        # Get matches in output format on the grid [0, n] where the center of the top-left coordinate is [0.5, 0.5]
        offset = (
            0.5  # Hpatches assumes that the center of the top-left pixel is at [0,0]
        )
        query_coords = (
            torch.stack(
                (
                    wq * (query_coords[..., 0] + 1) / 2,
                    hq * (query_coords[..., 1] + 1) / 2,
                ),
                axis=-1,
            )
            - offset
        )
        query_to_support = (
            torch.stack(
                (
                    wsup * (query_to_support[..., 0] + 1) / 2,
                    hsup * (query_to_support[..., 1] + 1) / 2,
                ),
                axis=-1,
            )
            - offset
        )
        return query_coords, query_to_support

    def inside_image(self, x, w, h):
        return torch.logical_and(
            x[:, 0] < (w - 1),
            torch.logical_and(x[:, 1] < (h - 1), (x > 0).prod(dim=-1)),
        )

    def benchmark(self, model):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        aepes = []
        pcks = []
        for seq_idx, seq_name in tqdm(
            enumerate(self.seq_names), total=len(self.seq_names)
        ):
            if seq_name[0] == "i":
                continue
            im1_path = os.path.join(self.seqs_path, seq_name, "1.ppm")
            im1 = Image.open(im1_path)
            w1, h1 = im1.size
            for im_idx in range(2, 7):
                im2_path = os.path.join(self.seqs_path, seq_name, f"{im_idx}.ppm")
                im2 = Image.open(im2_path)
                w2, h2 = im2.size
                matches, certainty = model.match(im2, im1, do_pred_in_og_res=True)
                matches, certainty = matches.reshape(-1, 4), certainty.reshape(-1)
                inv_homography = torch.from_numpy(
                    np.loadtxt(
                        os.path.join(self.seqs_path, seq_name, "H_1_" + str(im_idx))
                    )
                ).to(device)
                homography = torch.linalg.inv(inv_homography)
                pos_a, pos_b = self.convert_coordinates(
                    matches[:, :2], matches[:, 2:], w2, h2, w1, h1
                )
                pos_a, pos_b = pos_a.double(), pos_b.double()
                pos_a_h = torch.cat(
                    [pos_a, torch.ones([pos_a.shape[0], 1], device=device)], dim=1
                )
                pos_b_proj_h = (homography @ pos_a_h.t()).t()
                pos_b_proj = pos_b_proj_h[:, :2] / pos_b_proj_h[:, 2:]
                mask = self.inside_image(pos_b_proj, w1, h1)
                residual = pos_b - pos_b_proj
                dist = (residual**2).sum(dim=1).sqrt()[mask]
                aepes.append(torch.mean(dist).item())
                pck1 = (dist < 1.0).float().mean().item()
                pck3 = (dist < 3.0).float().mean().item()
                pck5 = (dist < 5.0).float().mean().item()
                pcks.append([pck1, pck3, pck5])
        m_pcks = np.mean(np.array(pcks), axis=0)
        return {
            "hp_pck1": m_pcks[0],
            "hp_pck3": m_pcks[1],
            "hp_pck5": m_pcks[2],
        }
