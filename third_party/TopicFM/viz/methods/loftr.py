from argparse import Namespace
import os
import torch
import cv2

from .base import Viz
from src.utils.metrics import compute_symmetrical_epipolar_errors, compute_pose_errors

from third_party.loftr.src.loftr import LoFTR, default_cfg


class VizLoFTR(Viz):
    def __init__(self, args):
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)

        self.match_threshold = args.match_threshold

        # Load model
        conf = dict(default_cfg)
        conf['match_coarse']['thr'] = self.match_threshold
        print(conf)
        self.model = LoFTR(config=conf)
        ckpt_dict = torch.load(args.ckpt)
        self.model.load_state_dict(ckpt_dict['state_dict'])
        self.model = self.model.eval().to(self.device)

        # Name the method
        # self.ckpt_name = args.ckpt.split('/')[-1].split('.')[0]
        self.name = 'LoFTR'

        print(f'Initialize {self.name}')

    def match_and_draw(self, data_dict, root_dir=None, ground_truth=False, measure_time=False, viz_matches=True):
        if measure_time:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        self.model(data_dict)
        if measure_time:
            torch.cuda.synchronize()
            end.record()
            torch.cuda.synchronize()
            self.time_stats.append(start.elapsed_time(end))

        kpts0 = data_dict['mkpts0_f'].cpu().numpy()
        kpts1 = data_dict['mkpts1_f'].cpu().numpy()

        img_name0, img_name1 = list(zip(*data_dict['pair_names']))[0]
        img0 = cv2.imread(os.path.join(root_dir, img_name0))
        img1 = cv2.imread(os.path.join(root_dir, img_name1))
        if str(data_dict["dataset_name"][0]).lower() == 'scannet':
            img0 = cv2.resize(img0, (640, 480))
            img1 = cv2.resize(img1, (640, 480))

        if viz_matches:
            saved_name = "_".join([img_name0.split('/')[-1].split('.')[0], img_name1.split('/')[-1].split('.')[0]])
            folder_matches = os.path.join(root_dir, "{}_viz_matches".format(self.name))
            if not os.path.exists(folder_matches):
                os.makedirs(folder_matches)
            path_to_save_matches = os.path.join(folder_matches, "{}.png".format(saved_name))
            if ground_truth:
                compute_symmetrical_epipolar_errors(data_dict)  # compute epi_errs for each match
                compute_pose_errors(data_dict)  # compute R_errs, t_errs, pose_errs for each pair
                epi_errors = data_dict['epi_errs'].cpu().numpy()
                R_errors, t_errors = data_dict['R_errs'][0], data_dict['t_errs'][0]

                self.draw_matches(kpts0, kpts1, img0, img1, epi_errors, path=path_to_save_matches,
                                  R_errs=R_errors, t_errs=t_errors)

                rel_pair_names = list(zip(*data_dict['pair_names']))
                bs = data_dict['image0'].size(0)
                metrics = {
                    # to filter duplicate pairs caused by DistributedSampler
                    'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                    'epi_errs': [data_dict['epi_errs'][data_dict['m_bids'] == b].cpu().numpy() for b in range(bs)],
                    'R_errs': data_dict['R_errs'],
                    't_errs': data_dict['t_errs'],
                    'inliers': data_dict['inliers']}
                self.eval_stats.append({'metrics': metrics})
            else:
                m_conf = 1 - data_dict["mconf"].cpu().numpy()
                self.draw_matches(kpts0, kpts1, img0, img1, m_conf, path=path_to_save_matches, conf_thr=0.4)
