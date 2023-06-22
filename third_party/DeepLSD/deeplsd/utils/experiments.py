"""
A set of utilities to manage and load checkpoints of training experiments.
"""

from pathlib import Path
import logging
import re
import shutil
from omegaconf import OmegaConf
import torch
import os

from ..settings import EXPER_PATH
from ..models import get_model


def list_checkpoints(dir_):
    """List all valid checkpoints in a given directory."""
    checkpoints = []
    for p in dir_.glob('checkpoint_*.tar'):
        numbers = re.findall(r'(\d+)', p.name)
        assert len(numbers) <= 2
        if len(numbers) == 0:
            continue
        if len(numbers) == 1:
            checkpoints.append((int(numbers[0]), p))
        else:
            checkpoints.append((int(numbers[1]), p))
    return checkpoints


def get_last_checkpoint(exper, allow_interrupted=True):
    """Get the last saved checkpoint for a given experiment name."""
    ckpts = list_checkpoints(Path(EXPER_PATH, exper))
    if not allow_interrupted:
        ckpts = [(n, p) for (n, p) in ckpts if '_interrupted' not in p.name]
    assert len(ckpts) > 0
    return sorted(ckpts)[-1][1]


def get_best_checkpoint(exper):
    """Get the checkpoint with the best loss, for a given experiment name."""
    p = Path(EXPER_PATH, exper, 'checkpoint_best.tar')
    return p


def delete_old_checkpoints(dir_, num_keep):
    """Delete all but the num_keep last saved checkpoints."""
    ckpts = list_checkpoints(dir_)
    ckpts = sorted(ckpts)[::-1]
    kept = 0
    for ckpt in ckpts:
        if ('_interrupted' in str(ckpt[1]) and kept > 0) or kept >= num_keep:
            logging.info(f'Deleting checkpoint {ckpt[1].name}')
            ckpt[1].unlink()
        else:
            kept += 1


def load_experiment(exper, conf={}):
    """Load and return the model of a given experiment."""
    ckpt = get_best_checkpoint(exper)
    logging.info(f'Loading checkpoint {ckpt.name}')
    ckpt = torch.load(str(ckpt), map_location='cpu')

    conf = OmegaConf.merge(ckpt['conf'].model, OmegaConf.create(conf))
    model = get_model(conf.name)(conf).eval()
    model.load_state_dict(ckpt['model'])
    return model


def flexible_load(state_dict, model, verbose=False):
    """ Load the state dict of a previous experiments even if
        all the parameters do not match and handles different GPU modes,
        such as with/without DataParallel. """
    dict_params = set(state_dict.keys())
    model_params = set(map(lambda n: n[0], model.named_parameters()))

    if dict_params == model_params:  # prefect fit
        logging.info('Loading all parameters of the checkpoint.')
        model.load_state_dict(state_dict, strict=True)
        return
    elif len(dict_params & model_params) == 0:  # perfect mismatch
        state_dict = {'.'.join(n.split('.')[1:]): p
                      for n, p in state_dict.items()}
        dict_params = set(state_dict.keys())
        if len(dict_params & model_params) == 0:
            raise ValueError('Could not manage to load the checkpoint with'
                             'parameters:' + '\n\t'.join(sorted(dict_params)))
    common_params = dict_params & model_params
    left_params = dict_params - model_params
    if verbose:
        logging.info('Loading parameters:\n\t'+'\n\t'.join(
            sorted(common_params)))
        if len(left_params) > 0:
            logging.info('Could not load parameters:\n\t'
                         + '\n\t'.join(sorted(left_params)))
    model.load_state_dict(state_dict, strict=False)


def save_experiment(model, optimizer, lr_scheduler, conf, losses, results,
                    best_eval, epoch, iter, output_dir, stop=False):
    """ Save the current model to a checkpoint
        and return the best result so far. """
    state = model.state_dict()
    checkpoint = {
        'model': state,
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'conf': OmegaConf.to_container(conf, resolve=True),
        'epoch': epoch,
        'losses': losses,
        'eval': results,
    }
    cp_name = f'checkpoint_{epoch}_{iter}'+('_interrupted'
                                            if stop else '')+'.tar'
    logging.info(f'Saving checkpoint {cp_name}')
    cp_path = str(output_dir / cp_name)
    torch.save(checkpoint, cp_path)
    if results[conf.train.best_key] < best_eval:
        best_eval = results[conf.train.best_key]
        logging.info(
            f'New best checkpoint: {conf.train.best_key}={best_eval}')
        shutil.copy(cp_path, str(output_dir / 'checkpoint_best.tar'))
    delete_old_checkpoints(output_dir, conf.train.keep_last_checkpoints)
    return best_eval
