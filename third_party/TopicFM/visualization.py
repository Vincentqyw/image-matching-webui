#!/usr/bin/env python
# coding: utf-8

import os, glob, cv2
import argparse
from argparse import Namespace
import yaml
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler

from src.datasets.custom_dataloader import TestDataLoader
from src.utils.dataset import read_img_gray
from configs.data.base import cfg as data_cfg
import viz


def get_model_config(method_name, dataset_name, root_dir='viz'):
    config_file = f'{root_dir}/configs/{method_name}.yml'
    with open(config_file, 'r') as f:
        model_conf = yaml.load(f, Loader=yaml.FullLoader)[dataset_name]
    return model_conf


class DemoDataset(Dataset):
    def __init__(self, dataset_dir, img_file=None, resize=0, down_factor=16):
        self.dataset_dir = dataset_dir
        if img_file is None:
            self.list_img_files = glob.glob(os.path.join(dataset_dir, "*.*"))
            self.list_img_files.sort()
        else:
            with open(img_file) as f:
                self.list_img_files = [os.path.join(dataset_dir, img_file.strip()) for img_file in f.readlines()]
        self.resize = resize
        self.down_factor = down_factor

    def __len__(self):
        return len(self.list_img_files)

    def __getitem__(self, idx):
        img_path = self.list_img_files[idx] #os.path.join(self.dataset_dir, self.list_img_files[idx])
        img, scale = read_img_gray(img_path, resize=self.resize, down_factor=self.down_factor)
        return {"img": img, "id": idx, "img_path": img_path}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize matches')
    parser.add_argument('--gpu', '-gpu', type=str, default='0')
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--dataset_dir', type=str, default='data/aachen-day-night')
    parser.add_argument('--pair_dir', type=str, default=None)
    parser.add_argument(
        '--dataset_name', type=str, choices=['megadepth', 'scannet', 'aachen_v1.1', 'inloc'], default='megadepth'
    )
    parser.add_argument('--measure_time', action="store_true")
    parser.add_argument('--no_viz', action="store_true")
    parser.add_argument('--compute_eval_metrics', action="store_true")
    parser.add_argument('--run_demo', action="store_true")

    args = parser.parse_args()

    model_cfg = get_model_config(args.method, args.dataset_name)
    class_name = model_cfg["class"]
    model = viz.__dict__[class_name](model_cfg)
    # all_args = Namespace(**vars(args), **model_cfg)
    if not args.run_demo:
        if args.dataset_name == 'megadepth':
            from configs.data.megadepth_test_1500 import cfg

            data_cfg.merge_from_other_cfg(cfg)
        elif args.dataset_name == 'scannet':
            from configs.data.scannet_test_1500 import cfg

            data_cfg.merge_from_other_cfg(cfg)
        elif args.dataset_name == 'aachen_v1.1':
            data_cfg.merge_from_list(["DATASET.TEST_DATA_SOURCE", "aachen_v1.1",
                                      "DATASET.TEST_DATA_ROOT", os.path.join(args.dataset_dir, "images/images_upright"),
                                      "DATASET.TEST_LIST_PATH", args.pair_dir,
                                      "DATASET.TEST_IMGSIZE", model_cfg["imsize"]])
        elif args.dataset_name == 'inloc':
            data_cfg.merge_from_list(["DATASET.TEST_DATA_SOURCE", "inloc",
                                      "DATASET.TEST_DATA_ROOT", args.dataset_dir,
                                      "DATASET.TEST_LIST_PATH", args.pair_dir,
                                      "DATASET.TEST_IMGSIZE", model_cfg["imsize"]])

        has_ground_truth = str(data_cfg.DATASET.TEST_DATA_SOURCE).lower() in ["megadepth", "scannet"]
        dataloader = TestDataLoader(data_cfg)
        with torch.no_grad():
            for data_dict in tqdm(dataloader):
                for k, v in data_dict.items():
                    if isinstance(v, torch.Tensor):
                        data_dict[k] = v.cuda() if torch.cuda.is_available() else v
                img_root_dir = data_cfg.DATASET.TEST_DATA_ROOT
                model.match_and_draw(data_dict, root_dir=img_root_dir, ground_truth=has_ground_truth,
                                     measure_time=args.measure_time, viz_matches=(not args.no_viz))

        if args.measure_time:
            print("Running time for each image is {} miliseconds".format(model.measure_time()))
        if args.compute_eval_metrics and has_ground_truth:
            model.compute_eval_metrics()
    else:
        demo_dataset = DemoDataset(args.dataset_dir, img_file=args.pair_dir, resize=640)
        sampler = SequentialSampler(demo_dataset)
        dataloader = DataLoader(demo_dataset, batch_size=1, sampler=sampler)

        writer = cv2.VideoWriter('topicfm_demo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (640 * 2 + 5, 480 * 2 + 10))

        model.run_demo(iter(dataloader), writer) #, output_dir="demo", no_display=True)
