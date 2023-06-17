import os
import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from loguru import logger


class CheckPoint:
    def __init__(self, dir=None, name="tmp"):
        self.name = name
        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)

    def __call__(
        self,
        model,
        optimizer,
        lr_scheduler,
        n,
    ):
        assert model is not None
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            model = model.module
        states = {
            "model": model.state_dict(),
            "n": n,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }
        torch.save(states, self.dir + self.name + f"_latest.pth")
        logger.info(f"Saved states {list(states.keys())}, at step {n}")
