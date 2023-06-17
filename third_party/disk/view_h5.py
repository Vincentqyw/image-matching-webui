import argparse, h5py, os, imageio, torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

from disk.common.vis import MultiFigure

parser = argparse.ArgumentParser(
    description='Script for viewing the keypoints.h5 and matches.h5 contents',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('h5_path', help='Path to .h5 artifacts')
parser.add_argument('image_path', help='Path to corresponding images')
parser.add_argument(
    '--image-extension', default='jpg', type=str,
    help='Extension of the images'
)
parser.add_argument(
    '--save', default=None, type=str,
    help=('If give a path, saves the visualizations rather than displaying '
          'them interactively')
)
parser.add_argument(
    'mode', choices=['keypoints', 'matches'],
    help=('Whether to dispay the keypoints (in a single image) or matches '
          '(across pairs)')
)

args = parser.parse_args()

save_i = 1
def show_or_save():
    global save_i

    if args.save is None:
        plt.show()
        return
    else:
        path = os.path.join(os.path.expanduser(args.save), f'{save_i}.png')
        plt.savefig(path)
        print(f'Saved to {path}')
        save_i += 1
        plt.close()

def view_keypoints(h5_path, image_path):
    keypoint_f = h5py.File(os.path.join(h5_path, 'keypoints.h5'), 'r')

    fname_to_id = {}
    for filename in tqdm(list(keypoint_f.keys())):
        keypoints = keypoint_f[filename][()]

        fname_with_ext = filename + '.' + args.image_extension
        path = os.path.join(image_path, fname_with_ext)
        if not os.path.isfile(path):
            raise IOError(f'Invalid image path {path}')

        image = imageio.imread(path)
        scale = 10 / max(image.shape)
        fig, ax = plt.subplots(figsize=(scale * image.shape[1], scale * image.shape[0]), constrained_layout=True)
        ax.axis('off')
        ax.imshow(image)
        ax.scatter(keypoints[:, 0], keypoints[:, 1], s=7, marker='o', color='white', edgecolors='black', linewidths=0.5)

        show_or_save()

def view_matches(h5_path, image_path):
    keypoint_f = h5py.File(os.path.join(h5_path, 'keypoints.h5'), 'r')
    match_file = h5py.File(os.path.join(h5_path, 'matches.h5'), 'r')
    
    added = set()

    for key_1 in match_file.keys():
        for key_2 in match_file[key_1].keys():
            matches = match_file[key_1][key_2][()]
            kp_1 = keypoint_f[key_1][()]
            kp_2 = keypoint_f[key_2][()]

            path_1 = os.path.join(image_path, key_1 + '.' + args.image_extension)
            path_2 = os.path.join(image_path, key_2 + '.' + args.image_extension)

            bm_1 = torch.from_numpy(imageio.imread(path_1))
            bm_2 = torch.from_numpy(imageio.imread(path_2))

            bigger_x = max(bm_1.shape[0], bm_2.shape[0])
            bigger_y = max(bm_1.shape[1], bm_2.shape[1])

            padded_1 = F.pad(bm_1, (
                0, 0,
                0, bigger_y - bm_1.shape[1],
                0, bigger_x - bm_1.shape[0]
            ))
            padded_2 = F.pad(bm_2, (
                0, 0,
                0, bigger_y - bm_2.shape[1],
                0, bigger_x - bm_2.shape[0]
            ))

            fig = MultiFigure(padded_1, padded_2)

            left  = torch.from_numpy(kp_1[matches[0]]).T
            right = torch.from_numpy(kp_2[matches[1]]).T

            fig.mark_xy(left, right)

            show_or_save()

if args.mode == 'keypoints':
    view_keypoints(args.h5_path, args.image_path)
elif args.mode == 'matches':
    view_matches(args.h5_path, args.image_path)
