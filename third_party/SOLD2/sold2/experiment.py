"""
Main file to launch training and testing experiments.
"""

import yaml
import os
import argparse
import numpy as np
import torch

from .config.project_config import Config as cfg
from .train import train_net
from .export import export_predictions, export_homograpy_adaptation


# Pytorch configurations
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True


def load_config(config_path):
    """ Load configurations from a given yaml file. """
    # Check file exists
    if not os.path.exists(config_path):
        raise ValueError("[Error] The provided config path is not valid.")

    # Load the configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def update_config(path, model_cfg=None, dataset_cfg=None):
    """ Update configuration file from the resume path. """
    # Check we need to update or completely override.
    model_cfg = {} if model_cfg is None else model_cfg
    dataset_cfg = {} if dataset_cfg is None else dataset_cfg

    # Load saved configs
    with open(os.path.join(path, "model_cfg.yaml"), "r") as f:
        model_cfg_saved = yaml.safe_load(f)
        model_cfg.update(model_cfg_saved)
    with open(os.path.join(path, "dataset_cfg.yaml"), "r") as f:
        dataset_cfg_saved = yaml.safe_load(f)
        dataset_cfg.update(dataset_cfg_saved)

    # Update the saved yaml file
    if not model_cfg == model_cfg_saved:
        with open(os.path.join(path, "model_cfg.yaml"), "w") as f:
            yaml.dump(model_cfg, f)
    if not dataset_cfg == dataset_cfg_saved:
        with open(os.path.join(path, "dataset_cfg.yaml"), "w") as f:
            yaml.dump(dataset_cfg, f)

    return model_cfg, dataset_cfg


def record_config(model_cfg, dataset_cfg, output_path):
    """ Record dataset config to the log path. """
    # Record model config
    with open(os.path.join(output_path, "model_cfg.yaml"), "w") as f:
            yaml.safe_dump(model_cfg, f)
    
    # Record dataset config
    with open(os.path.join(output_path, "dataset_cfg.yaml"), "w") as f:
            yaml.safe_dump(dataset_cfg, f)
    

def train(args, dataset_cfg, model_cfg, output_path):
    """ Training function. """
    # Update model config from the resume path (only in resume mode)
    if args.resume:
        if os.path.realpath(output_path) != os.path.realpath(args.resume_path):
            record_config(model_cfg, dataset_cfg, output_path)
        
    # First time, then write the config file to the output path
    else:
        record_config(model_cfg, dataset_cfg, output_path)

    # Launch the training
    train_net(args, dataset_cfg, model_cfg, output_path)


def export(args, dataset_cfg, model_cfg, output_path,
           export_dataset_mode=None, device=torch.device("cuda")):
    """ Export function. """
    # Choose between normal predictions export or homography adaptation
    if dataset_cfg.get("homography_adaptation") is not None:
        print("[Info] Export predictions with homography adaptation.")
        export_homograpy_adaptation(args, dataset_cfg, model_cfg, output_path,
                                    export_dataset_mode, device)
    else:
        print("[Info] Export predictions normally.")
        export_predictions(args, dataset_cfg, model_cfg, output_path,
                           export_dataset_mode)


def main(args, dataset_cfg, model_cfg, export_dataset_mode=None,
         device=torch.device("cuda")):
    """ Main function. """
    # Make the output path
    output_path = os.path.join(cfg.EXP_PATH, args.exp_name)

    if args.mode == "train":
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("[Info] Training mode")
        print("\t Output path: %s" % output_path)
        train(args, dataset_cfg, model_cfg, output_path)
    elif args.mode == "export":
        # Different output_path in export mode
        output_path = os.path.join(cfg.export_dataroot, args.exp_name)
        print("[Info] Export mode")
        print("\t Output path: %s" % output_path)
        export(args, dataset_cfg, model_cfg, output_path, export_dataset_mode, device=device)
    else:
        raise ValueError("[Error]: Unknown mode: " + args.mode)


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train",
                        help="'train' or 'export'.")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Path to the dataset config.")
    parser.add_argument("--model_config", type=str, default=None,
                        help="Path to the model config.")
    parser.add_argument("--exp_name", type=str, default="exp",
                        help="Experiment name.")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Load a previously trained model.")
    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Start training from a pre-trained model.")
    parser.add_argument("--resume_path", default=None,
                        help="Path from which to resume training.")
    parser.add_argument("--pretrained_path", default=None,
                        help="Path to the pre-trained model.")
    parser.add_argument("--checkpoint_name", default=None,
                        help="Name of the checkpoint to use.")
    parser.add_argument("--export_dataset_mode", default=None,
                        help="'train' or 'test'.")
    parser.add_argument("--export_batch_size", default=4, type=int,
                        help="Export batch size.")

    args = parser.parse_args()

    # Check if GPU is available
    # Get the model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Check if dataset config and model config is given.
    if (((args.dataset_config is None) or (args.model_config is None))
        and (not args.resume) and (args.mode == "train")):
        raise ValueError(
            "[Error] The dataset config and model config should be given in non-resume mode")

    # If resume, check if the resume path has been given
    if args.resume and (args.resume_path is None):
        raise ValueError(
            "[Error] Missing resume path.")

    # [Training] Load the config file.
    if args.mode == "train" and (not args.resume):
        # Check the pretrained checkpoint_path exists
        if args.pretrained:
            checkpoint_folder = args.resume_path
            checkpoint_path = os.path.join(args.pretrained_path,
                                           args.checkpoint_name)
            if not os.path.exists(checkpoint_path):
                raise ValueError("[Error] Missing checkpoint: "
                                 + checkpoint_path)
        dataset_cfg = load_config(args.dataset_config)
        model_cfg = load_config(args.model_config)       

    # [resume Training, Test, Export] Load the config file.
    elif (args.mode == "train" and args.resume) or (args.mode == "export"):
        # Check checkpoint path exists
        checkpoint_folder = args.resume_path
        checkpoint_path = os.path.join(args.resume_path, args.checkpoint_name)
        if not os.path.exists(checkpoint_path):
            raise ValueError("[Error] Missing checkpoint: " + checkpoint_path)

        # Load model_cfg from checkpoint folder if not provided
        if args.model_config is None:
            print("[Info] No model config provided. Loading from checkpoint folder.")
            model_cfg_path = os.path.join(checkpoint_folder, "model_cfg.yaml")
            if not os.path.exists(model_cfg_path):
                raise ValueError(
                    "[Error] Missing model config in checkpoint path.")
            model_cfg = load_config(model_cfg_path)
        else:
            model_cfg = load_config(args.model_config)
        
        # Load dataset_cfg from checkpoint folder if not provided
        if args.dataset_config is None:
            print("[Info] No dataset config provided. Loading from checkpoint folder.")
            dataset_cfg_path = os.path.join(checkpoint_folder,
                                            "dataset_cfg.yaml")
            if not os.path.exists(dataset_cfg_path):
                raise ValueError(
                    "[Error] Missing dataset config in checkpoint path.")
            dataset_cfg = load_config(dataset_cfg_path)
        else:
            dataset_cfg = load_config(args.dataset_config)
        
        # Check the --export_dataset_mode flag
        if (args.mode == "export") and (args.export_dataset_mode is None):
            raise ValueError("[Error] Empty --export_dataset_mode flag.")
    else:
        raise ValueError("[Error] Unknown mode: " + args.mode)
    
    # Set the random seed
    seed = dataset_cfg.get("random_seed", 0)
    set_random_seed(seed)

    main(args, dataset_cfg, model_cfg,
         export_dataset_mode=args.export_dataset_mode, device=device)
