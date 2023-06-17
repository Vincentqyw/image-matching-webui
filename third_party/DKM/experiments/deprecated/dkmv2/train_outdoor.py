import os
import torch
from argparse import ArgumentParser

from torch import nn
from dkm.datasets.megadepth import MegadepthBuilder
from dkm.losses import DepthRegressionLoss
from torch.utils.data import ConcatDataset

from dkm.train.train import train_k_steps
from dkm.checkpointing.checkpoint import CheckPoint
from dkm import DKMv2


def run(gpus=1):
    experiment_name = os.path.splitext(os.path.basename(__file__))[0]
    checkpoint_dir = "workspace/checkpoints"
    h, w = 384, 512
    model = DKMv2(
        pretrained=False,
    )
    # Num steps
    n0 = 0
    batch_size = gpus * 8
    N = (32 * 250000) // batch_size  # 250k steps of batch size 32
    # checkpoint every
    k = 150000 // batch_size

    # Data
    mega = MegadepthBuilder(data_root="data/megadepth")
    megadepth_train1 = mega.build_scenes(
        split="train_loftr", min_overlap=0.01, ht=h, wt=w, shake_t=32
    )
    megadepth_train2 = mega.build_scenes(
        split="train_loftr", min_overlap=0.35, ht=h, wt=w, shake_t=32
    )
    megadepth_train = ConcatDataset(megadepth_train1 + megadepth_train2)
    mega_ws = mega.weight_scenes(megadepth_train, alpha=0.75)
    # Loss and optimizer
    depth_loss = DepthRegressionLoss(ce_weight=0.01)

    parameters = [
        {"params": model.encoder.parameters(), "lr": gpus * 1e-6},
        {"params": model.decoder.parameters(), "lr": gpus * 1e-4},
    ]
    optimizer = torch.optim.AdamW(parameters, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[N // 3, (2 * N) // 3], gamma=0.2
    )
    checkpointer = CheckPoint(checkpoint_dir, experiment_name)
    checkpoint_name = checkpoint_dir + experiment_name + "_latest.pth"
    states = {}
    # Load checkpoint (if existing)
    if os.path.exists(checkpoint_name):
        states = torch.load(checkpoint_name)
        if "model" in states:
            model.load_state_dict(states["model"])
        if "n" in states:
            n0 = states["n"] if states["n"] else 0
        if "optimizer" in states:
            optimizer.load_state_dict(states["optimizer"])
        if "lr_scheduler" in states:
            lr_scheduler.load_state_dict(states["lr_scheduler"])
    if states:
        print(f"Loaded states {list(states.keys())}, at step {n0}")

    dp_model = nn.DataParallel(model)
    # Train
    for n in range(n0, N, k):
        mega_sampler = torch.utils.data.WeightedRandomSampler(
            mega_ws, num_samples=batch_size * k, replacement=False
        )
        mega_dataloader = iter(
            torch.utils.data.DataLoader(
                megadepth_train,
                batch_size=batch_size,
                sampler=mega_sampler,
                num_workers=gpus * 8,
            )
        )
        train_k_steps(
            n, k, mega_dataloader, dp_model, depth_loss, optimizer, lr_scheduler
        )
        checkpointer(model, optimizer, lr_scheduler, n)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", default=1, type=int)
    args, _ = parser.parse_known_args()
    run(gpus=args.gpus)
