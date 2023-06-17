from argparse import Namespace
import os, sys
import torch
import cv2
from pathlib import Path

from .base import Viz
from src.utils.metrics import compute_symmetrical_epipolar_errors, compute_pose_errors

patch2pix_path = Path(__file__).parent / '../../third_party/patch2pix'
sys.path.append(str(patch2pix_path))
from third_party.patch2pix.utils.eval.model_helper import load_model, estimate_matches


class VizPatch2Pix(Viz):
    def __init__(self, args):
        super().__init__()

        if type(args) == dict:
            args = Namespace(**args)
        self.imsize = args.imsize
        self.match_threshold = args.match_threshold
        self.ksize = args.ksize
        self.model = load_model(args.ckpt, method='patch2pix')
        self.name = 'Patch2Pix'
        print(f'Initialize {self.name} with image size {self.imsize}')

    def match_and_draw(self, data_dict, root_dir=None, ground_truth=False, measure_time=False, viz_matches=True):
        img_name0, img_name1 = list(zip(*data_dict['pair_names']))[0]
        path_img0 = os.path.join(root_dir, img_name0)
        path_img1 = os.path.join(root_dir, img_name1)
        img0, img1 = cv2.imread(path_img0), cv2.imread(path_img1)
        return_m_upscale = True
        if str(data_dict["dataset_name"][0]).lower() == 'scannet':
            # self.imsize = 640
            img0 = cv2.resize(img0, tuple(self.imsize))  # (640, 480))
            img1 = cv2.resize(img1, tuple(self.imsize))  # (640, 480))
            return_m_upscale = False
        outputs = estimate_matches(self.model, path_img0, path_img1,
                                   ksize=self.ksize, io_thres=self.match_threshold,
                                   eval_type='fine', imsize=self.imsize,
                                   return_upscale=return_m_upscale, measure_time=measure_time)
        if measure_time:
            self.time_stats.append(outputs[-1])
        matches, mconf = outputs[0], outputs[1]
        kpts0 = matches[:, :2]
        kpts1 = matches[:, 2:4]

        if viz_matches:
            saved_name = "_".join([img_name0.split('/')[-1].split('.')[0], img_name1.split('/')[-1].split('.')[0]])
            folder_matches = os.path.join(root_dir, "{}_viz_matches".format(self.name))
            if not os.path.exists(folder_matches):
                os.makedirs(folder_matches)
            path_to_save_matches = os.path.join(folder_matches, "{}.png".format(saved_name))

            if ground_truth:
                data_dict["mkpts0_f"] = torch.from_numpy(matches[:, :2]).float().to(self.device)
                data_dict["mkpts1_f"] = torch.from_numpy(matches[:, 2:4]).float().to(self.device)
                data_dict["m_bids"] = torch.zeros(matches.shape[0], device=self.device, dtype=torch.float32)
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
                m_conf = 1 - mconf
                self.draw_matches(kpts0, kpts1, img0, img1, m_conf, path=path_to_save_matches, conf_thr=0.4)
