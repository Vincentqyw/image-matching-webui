import os
import cv2
import torch
from argparse import ArgumentParser
from torch import nn
import albumentations as A
from torch.utils.data import ConcatDataset
import kornia.augmentation as K
from tqdm import tqdm

from dkm.datasets.megadepth import MegadepthBuilder
from dkm.datasets.synthetic import (
    build_ade20k,
    build_cityscapes,
    build_coco,
    build_coco_foreground,
    SampleObject,
    MovingObjectsDataset,
)
from dkm.losses import DepthRegressionLoss, HomographyRegressionLoss
from dkm.train.train import train_k_steps
from dkm.checkpointing.checkpoint import CheckPoint
from dkm.utils.transforms import GeometricSequential, RandomPerspective
from dkm import DKM


def run(gpus=1):
    experiment_name = os.path.splitext(os.path.basename(__file__))[0]
    checkpoint_dir = "workspace/checkpoints"
    h, w = 384, 512
    model = DKM(
        pretrained=False,
    )
    # Num steps
    n0 = 0
    batch_size = gpus * 8
    N = (32 * 250000) // batch_size  # 250k steps of batch size 32
    # checkpoint every
    k = 150000 // batch_size

    # Data

    # MegaDepth
    mega = MegadepthBuilder(data_root="data/megadepth")
    megadepth_train1 = mega.build_scenes(
        split="train_loftr", min_overlap=0.01, ht=h, wt=w, shake_t=32
    )
    megadepth_train2 = mega.build_scenes(
        split="train_loftr", min_overlap=0.35, ht=h, wt=w, shake_t=32
    )
    megadepth_train = ConcatDataset(megadepth_train1 + megadepth_train2)
    mega_ws = mega.weight_scenes(megadepth_train, alpha=0.75)
    # Synthetic
    root_dir = "data/homog_data"
    cityscapes_ims = build_cityscapes()
    ade20k_ims = build_ade20k()
    coco_ims = build_coco()
    homog_ims = cityscapes_ims + ade20k_ims + coco_ims
    H_generator = GeometricSequential(
        RandomPerspective(0.6, p=1),
        K.RandomAffine(degrees=35, p=1, scale=(1, 1.6)),
        align_corners=True,
    )
    photometric_distortion = A.Compose(
        [
            A.Resize(h, w, cv2.INTER_CUBIC),
            A.RandomBrightnessContrast(p=1.0),
            A.Normalize(),
        ]
    )

    coco_instances = build_coco_foreground(root_dir)
    obj_sampler = SampleObject(root_dir, coco_instances)
    homog_train = MovingObjectsDataset(
        homog_ims,
        H_generator=H_generator,
        photometric_distortion=photometric_distortion,
        object_sampler=obj_sampler,
        num_objects=3,
        h=h,
        w=w,
    )
    # Loss and optimizer
    depth_loss = DepthRegressionLoss(ce_weight=0.01)
    homog_loss = HomographyRegressionLoss(ce_weight=0.0)
    parameters = [
        {"params": model.encoder.parameters(), "lr": gpus * 1e-4},
        {"params": model.decoder.parameters(), "lr": gpus * 1e-6},
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
        homog_sampler = torch.utils.data.WeightedRandomSampler(
            torch.ones(len(homog_train)),
            num_samples=batch_size * (k // 2),
            replacement=False,
        )
        homog_dataloader = iter(
            torch.utils.data.DataLoader(
                homog_train,
                batch_size=batch_size,
                sampler=homog_sampler,
                num_workers=gpus * 8,
            )
        )
        mega_sampler = torch.utils.data.WeightedRandomSampler(
            mega_ws, num_samples=batch_size * (k // 2), replacement=False
        )
        mega_dataloader = iter(
            torch.utils.data.DataLoader(
                megadepth_train,
                batch_size=batch_size,
                sampler=mega_sampler,
                num_workers=gpus * 8,
            )
        )
        for nk in tqdm(range(n, n + k, 2)):
            train_k_steps(
                nk,
                1,
                mega_dataloader,
                dp_model,
                depth_loss,
                optimizer,
                lr_scheduler,
                progress_bar=False,
            )
            train_k_steps(
                nk + 1,
                1,
                homog_dataloader,
                dp_model,
                homog_loss,
                optimizer,
                lr_scheduler,
                progress_bar=False,
            )
        checkpointer(model, optimizer, lr_scheduler, nk + 2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", default=1, type=int)
    args, _ = parser.parse_known_args()
    run(gpus=args.gpus)
