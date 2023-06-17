import torch
import numpy as np
import tqdm
from dkm.datasets import MegadepthBuilder
from dkm.utils import warp_kpts
from torch.utils.data import ConcatDataset


class MegadepthDenseBenchmark:
    def __init__(self, data_root="data/megadepth", h = 384, w = 512, num_samples = 2000, device=None) -> None:
        mega = MegadepthBuilder(data_root=data_root)
        self.dataset = ConcatDataset(
            mega.build_scenes(split="test_loftr", ht=h, wt=w)
        )  # fixed resolution of 384,512
        self.num_samples = num_samples
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

    def geometric_dist(self, depth1, depth2, T_1to2, K1, K2, dense_matches):
        b, h1, w1, d = dense_matches.shape
        with torch.no_grad():
            x1 = dense_matches[..., :2].reshape(b, h1 * w1, 2)
            # x1 = torch.stack((2*x1[...,0]/w1-1,2*x1[...,1]/h1-1),dim=-1)
            mask, x2 = warp_kpts(
                x1.double(),
                depth1.double(),
                depth2.double(),
                T_1to2.double(),
                K1.double(),
                K2.double(),
            )
            x2 = torch.stack(
                (w1 * (x2[..., 0] + 1) / 2, h1 * (x2[..., 1] + 1) / 2), dim=-1
            )
            prob = mask.float().reshape(b, h1, w1)
        x2_hat = dense_matches[..., 2:]
        x2_hat = torch.stack(
            (w1 * (x2_hat[..., 0] + 1) / 2, h1 * (x2_hat[..., 1] + 1) / 2), dim=-1
        )
        gd = (x2_hat - x2.reshape(b, h1, w1, 2)).norm(dim=-1)
        gd = gd[prob == 1]
        pck_1 = (gd < 1.0).float().mean()
        pck_3 = (gd < 3.0).float().mean()
        pck_5 = (gd < 5.0).float().mean()
        gd = gd.mean()
        return gd, pck_1, pck_3, pck_5

    def benchmark(self, model, batch_size=8):
        model.train(False)
        with torch.no_grad():
            gd_tot = 0.0
            pck_1_tot = 0.0
            pck_3_tot = 0.0
            pck_5_tot = 0.0
            sampler = torch.utils.data.WeightedRandomSampler(
                torch.ones(len(self.dataset)), replacement=False, num_samples=self.num_samples
            )
            dataloader = torch.utils.data.DataLoader(
                self.dataset, batch_size=8, num_workers=batch_size, sampler=sampler
            )
            for data in tqdm.tqdm(dataloader):
                im1, im2, depth1, depth2, T_1to2, K1, K2 = (
                    data["query"],
                    data["support"],
                    data["query_depth"].to(self.device),
                    data["support_depth"].to(self.device),
                    data["T_1to2"].to(self.device),
                    data["K1"].to(self.device),
                    data["K2"].to(self.device),
                )
                matches, certainty = model.match(im1, im2, batched=True)
                gd, pck_1, pck_3, pck_5 = self.geometric_dist(
                    depth1, depth2, T_1to2, K1, K2, matches
                )
                gd_tot, pck_1_tot, pck_3_tot, pck_5_tot = (
                    gd_tot + gd,
                    pck_1_tot + pck_1,
                    pck_3_tot + pck_3,
                    pck_5_tot + pck_5,
                )
        return {
            "mega_pck_1": pck_1_tot.item() / len(dataloader),
            "mega_pck_3": pck_3_tot.item() / len(dataloader),
            "mega_pck_5": pck_5_tot.item() / len(dataloader),
        }
