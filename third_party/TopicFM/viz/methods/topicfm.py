from argparse import Namespace
import os
import torch
import cv2
from time import time
from pathlib import Path
import matplotlib.cm as cm
import numpy as np

from src.models.topic_fm import TopicFM
from src import get_model_cfg
from .base import Viz
from src.utils.metrics import compute_symmetrical_epipolar_errors, compute_pose_errors
from src.utils.plotting import draw_topics, draw_topicfm_demo, error_colormap


class VizTopicFM(Viz):
    def __init__(self, args):
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)

        self.match_threshold = args.match_threshold
        self.n_sampling_topics = args.n_sampling_topics
        self.show_n_topics = args.show_n_topics

        # Load model
        conf = dict(get_model_cfg())
        conf['match_coarse']['thr'] = self.match_threshold
        conf['coarse']['n_samples'] = self.n_sampling_topics
        print("model config: ", conf)
        self.model = TopicFM(config=conf)
        ckpt_dict = torch.load(args.ckpt)
        self.model.load_state_dict(ckpt_dict['state_dict'])
        self.model = self.model.eval().to(self.device)

        # Name the method
        # self.ckpt_name = args.ckpt.split('/')[-1].split('.')[0]
        self.name = 'TopicFM'

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

                # compute evaluation metrics
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
            if self.show_n_topics > 0:
                folder_topics = os.path.join(root_dir, "{}_viz_topics".format(self.name))
                if not os.path.exists(folder_topics):
                    os.makedirs(folder_topics)
                draw_topics(data_dict, img0, img1, saved_folder=folder_topics, show_n_topics=self.show_n_topics,
                            saved_name=saved_name)

    def run_demo(self, dataloader, writer=None, output_dir=None, no_display=False, skip_frames=1):
        data_dict = next(dataloader)

        frame_id = 0
        last_image_id = 0
        img0 = np.array(cv2.imread(str(data_dict["img_path"][0])), dtype=np.float32) / 255
        frame_tensor = data_dict["img"].to(self.device)
        pair_data = {'image0': frame_tensor}
        last_frame = cv2.resize(img0, (frame_tensor.shape[-1], frame_tensor.shape[-2]), cv2.INTER_LINEAR)

        if output_dir is not None:
            print('==> Will write outputs to {}'.format(output_dir))
            Path(output_dir).mkdir(exist_ok=True)

        # Create a window to display the demo.
        if not no_display:
            window_name = 'Topic-assisted Feature Matching'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, (640 * 2, 480 * 2))
        else:
            print('Skipping visualization, will not show a GUI.')

        # Print the keyboard help menu.
        print('==> Keyboard control:\n'
              '\tn: select the current frame as the reference image (left)\n'
              '\tq: quit')

        # vis_range = [kwargs["bottom_k"], kwargs["top_k"]]

        while True:
            frame_id += 1
            if frame_id == len(dataloader):
                print('Finished demo_loftr.py')
                break
            data_dict = next(dataloader)
            if frame_id % skip_frames != 0:
                # print("Skipping frame.")
                continue

            stem0, stem1 = last_image_id, data_dict["id"][0].item() - 1
            frame = np.array(cv2.imread(str(data_dict["img_path"][0])), dtype=np.float32) / 255

            frame_tensor = data_dict["img"].to(self.device)
            frame = cv2.resize(frame, (frame_tensor.shape[-1], frame_tensor.shape[-2]), interpolation=cv2.INTER_LINEAR)
            pair_data = {**pair_data, 'image1': frame_tensor}
            self.model(pair_data)

            total_n_matches = len(pair_data['mkpts0_f'])
            mkpts0 = pair_data['mkpts0_f'].cpu().numpy()  # [vis_range[0]:vis_range[1]]
            mkpts1 = pair_data['mkpts1_f'].cpu().numpy()  # [vis_range[0]:vis_range[1]]
            mconf = pair_data['mconf'].cpu().numpy()  # [vis_range[0]:vis_range[1]]

            # Normalize confidence.
            if len(mconf) > 0:
                mconf = 1 - mconf

            # alpha = 0
            # color = cm.jet(mconf, alpha=alpha)
            color = error_colormap(mconf, thr=0.4, alpha=0.1)

            text = [
                f'Topics',
                '#Matches: {}'.format(total_n_matches),
            ]

            out = draw_topicfm_demo(pair_data, last_frame, frame, mkpts0, mkpts1, color, text,
                                    show_n_topics=4, path=None)

            if not no_display:
                if writer is not None:
                    writer.write(out)
                cv2.imshow('TopicFM Matches', out)
                key = chr(cv2.waitKey(10) & 0xFF)
                if key == 'q':
                    if writer is not None:
                        writer.release()
                    print('Exiting...')
                    break
                elif key == 'n':
                    pair_data['image0'] = frame_tensor
                    last_frame = frame
                    last_image_id = (data_dict["id"][0].item() - 1)
                    frame_id_left = frame_id

            elif output_dir is not None:
                stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
                out_file = str(Path(output_dir, stem + '.png'))
                print('\nWriting image to {}'.format(out_file))
                cv2.imwrite(out_file, out)
            else:
                raise ValueError("output_dir is required when no display is given.")

        cv2.destroyAllWindows()
        if writer is not None:
            writer.release()

