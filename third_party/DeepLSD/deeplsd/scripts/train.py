"""
A generic training script that works with any model and dataset.
"""

import sys
import argparse
import logging
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import random
from tqdm import tqdm
import signal
import multiprocessing as mp

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..datasets import get_dataset
from ..models import get_model
from ..utils.stdout_capturing import capture_outputs
from ..utils.tools import AverageMetric, MedianMetric
from ..utils.tensor import batch_to_device
from ..utils.experiments import (
    get_last_checkpoint, get_best_checkpoint, save_experiment)
from ..settings import EXPER_PATH


default_train_conf = {
    'seed': '???',  # training seed
    'epochs': 1,  # number of epochs
    'optimizer': 'adam',  # name of optimizer in [adam, sgd, rmsprop]
    'lr': 0.001,  # learning rate
    'optimizer_options': {},  # optional arguments passed to the optimizer
    'scheduler': 'ReduceLROnPlateau',  # scheduler of the learning rate
    'patience': 10,  # patience of the scheduler (in epochs)
    'eval_every_iter': 1000,  # interval for evaluation on the validation set
    'log_every_iter': 200,  # interval for logging the loss to the console
    'save_every_iter': 2000,  # interval for model saving
    'keep_last_checkpoints': 10,  # keep only the last X checkpoints
    'load_experiment': None,  # initialize the model from a previous experiment
    'median_metrics': [],  # display the median (not the mean) for some metrics
    'best_key': 'loss/total',  # key to use to select the best checkpoint
    'dataset_callback_fn': None,  # callback to call at the start of each epoch
}
default_train_conf = OmegaConf.create(default_train_conf)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def do_evaluation(model, loader, device, loss_fn, metrics_fn, conf):
    model.eval()
    averages = {}
    for data in tqdm(loader, desc='Evaluation', ascii=True):
        data = batch_to_device(data, device, non_blocking=True)
        with torch.no_grad():
            pred = model(data)
            losses = loss_fn(pred, data)
            metrics = metrics_fn(pred, data)
            del pred, data
        results = {**metrics, **{'loss/'+k: v for k, v in losses.items()}}
        for k, v in results.items():
            if k not in averages:
                if k in conf.median_metrics:
                    averages[k] = MedianMetric()
                else:
                    averages[k] = AverageMetric()
            averages[k].update(v)
    results = {k: averages[k].compute() for k in results}
    return results


def training(conf, output_dir, args):
    if args.restore:
        logging.info(f'Restoring from previous training of {args.experiment}')
        init_cp = get_last_checkpoint(args.experiment, allow_interrupted=False)
        logging.info(f'Restoring from checkpoint {init_cp.name}')
        init_cp = torch.load(str(init_cp), map_location='cpu')
        conf = OmegaConf.merge(OmegaConf.create(init_cp['conf']), conf)
        epoch = init_cp['epoch'] + 1

        # get the best loss or eval metric from the previous best checkpoint
        best_cp = get_best_checkpoint(args.experiment)
        best_cp = torch.load(str(best_cp), map_location='cpu')
        best_eval = best_cp['eval'][conf.train.best_key]
        del best_cp
    else:
        # we start a new, fresh training
        conf.train = OmegaConf.merge(default_train_conf, conf.train)
        epoch = 0
        best_eval = float('inf')
        if conf.train.load_experiment:
            logging.info(
                'Will fine-tune from weights of {conf.train.load_experiment}')
            # the user has to make sure that the weights are compatible
            init_cp = get_last_checkpoint(conf.train.load_experiment)
            init_cp = torch.load(str(init_cp), map_location='cpu')
        else:
            init_cp = None

    OmegaConf.set_struct(conf, True)  # prevent access to unknown entries
    writer = SummaryWriter(log_dir=str(output_dir))

    dataset = get_dataset(conf.data.name)(conf.data)
    if args.overfit:
        # we train and eval with the same single training batch
        logging.info('Data in overfitting mode')
        train_loader = dataset.get_overfit_loader('train')
        val_loader = dataset.get_overfit_loader('val')
    else:
        train_loader = dataset.get_data_loader('train')
        val_loader = dataset.get_data_loader('val')
    logging.info('Training loader has {} batches'.format(len(train_loader)))
    logging.info('Validation loader has {} batches'.format(len(val_loader)))

    # interrupts are caught and delayed for graceful termination
    def sigint_handler(signal, frame):
        logging.info('Caught keyboard interrupt signal, will terminate')
        nonlocal stop
        if stop:
            raise KeyboardInterrupt
        stop = True
    stop = False
    signal.signal(signal.SIGINT, sigint_handler)

    model = get_model(conf.model.name)(conf.model)
    if init_cp is not None:
        if list(init_cp['model'].keys())[0][:7] == 'module.':
            # Remove 'module.' added by the DataParallel training
            init_state_dict = {k[7:]: v for k, v in init_cp['model'].items()}
        else:
            init_state_dict = init_cp['model']
        model.load_state_dict(init_state_dict)
    logging.info(f'Model: \n{model}')

    loss_fn, metrics_fn = model.loss, model.metrics
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    logging.info('Using {} GPU(s)'.format(torch.cuda.device_count()))
    if torch.cuda.device_count() > 1:
        logging.warning('Using DistributedParallel is preferred.')
        model = torch.nn.DataParallel(model)  # should update later on
    torch.backends.cudnn.benchmark = True

    optimizer_fn = {'sgd': torch.optim.SGD,
                    'adam': torch.optim.Adam,
                    'rmsprop': torch.optim.RMSprop}[conf.train.optimizer]
    optimizer = optimizer_fn(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=conf.train.lr, **conf.train.optimizer_options)
    if args.restore:
        optimizer.load_state_dict(init_cp['optimizer'])

    if conf.train.scheduler is not None:
        if conf.train.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer, patience=conf.train.patience, mode='min')
        else:
            sys.exit("Unknown LR scheduler: " + conf.train.scheduler)

    logging.info(f'Starting training with configuration:\n{OmegaConf.to_yaml(conf)}')
    losses_ = None

    while epoch < conf.train.epochs and not stop:
        logging.info(f'Starting epoch {epoch}')
        set_seed(conf.train.seed + epoch)
        if epoch > 0 and conf.train.dataset_callback_fn:
            getattr(train_loader.dataset, conf.train.dataset_callback_fn)(
                conf.train.seed + epoch)

        for it, data in enumerate(train_loader):
            tot_it = len(train_loader)*epoch + it

            model.train()
            optimizer.zero_grad()
            data = batch_to_device(data, device, non_blocking=True)
            pred = model(data)
            losses = loss_fn(pred, data)
            loss = torch.mean(losses['total'])
            loss.backward()
            optimizer.step()

            if it % conf.train.log_every_iter == 0:
                losses_ = {k: torch.mean(v).item() for k, v in losses.items()}
                str_losses = [f'{k} {v:.8f}' for k, v in losses_.items()]
                logging.info('[E {} | it {}] loss {{{}}}'.format(
                    epoch, it, ', '.join(str_losses)))
                for k, v in losses_.items():
                    writer.add_scalar('training/'+k, v, tot_it)

            del pred, data, loss, losses

            if ((it % conf.train.eval_every_iter == 0) or stop
                    or it == (len(train_loader)-1)):
                results = do_evaluation(
                    model, val_loader, device, loss_fn, metrics_fn, conf.train)
                str_results = [f'{k} {v:.8f}' for k, v in results.items()]
                logging.info('[Validation] {{{}}}'.format(
                    ', '.join(str_results)))
                for k, v in results.items():
                    writer.add_scalar('val/'+k, v, tot_it)
                if conf.train.scheduler is not None:
                    scheduler.step(results['loss/total'])
                torch.cuda.empty_cache()  # should be cleared at the first iter

            if stop:
                break

            if ((conf.train.save_every_iter is not None)
                and (tot_it % conf.train.save_every_iter == 0)):
                best_eval = save_experiment(
                    model, optimizer, scheduler, conf, losses_, results,
                    best_eval, epoch, tot_it, output_dir, stop)

        best_eval = save_experiment(
            model, optimizer, scheduler, conf, losses_, results,
            best_eval, epoch, tot_it, output_dir, stop)

        epoch += 1

    logging.info('Finished training.')
    writer.close()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('--conf', type=str)
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('dotlist', nargs='*')
    args = parser.parse_intermixed_args()

    output_dir = Path(EXPER_PATH, args.experiment)
    output_dir.mkdir(exist_ok=True, parents=True)

    conf = OmegaConf.from_cli(args.dotlist)
    if args.conf:
        conf = OmegaConf.merge(OmegaConf.load(args.conf), conf)
    if not args.restore:
        if conf.train.seed is None:
            conf.train.seed = torch.initial_seed() & (2**32 - 1)
        OmegaConf.save(conf, str(output_dir / 'config.yaml'))

    with capture_outputs(output_dir / 'log.txt'):
        training(conf, output_dir, args)
