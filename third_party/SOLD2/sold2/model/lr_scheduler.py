"""
This file implements different learning rate schedulers
"""
import torch


def get_lr_scheduler(lr_decay, lr_decay_cfg, optimizer):
    """ Get the learning rate scheduler according to the config. """
    # If no lr_decay is specified => return None
    if (lr_decay == False) or (lr_decay_cfg is None):
        schduler = None
    # Exponential decay
    elif (lr_decay == True) and (lr_decay_cfg["policy"] == "exp"):
        schduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=lr_decay_cfg["gamma"]
        )
    # Unknown policy
    else:
        raise ValueError("[Error] Unknow learning rate decay policy!")

    return schduler