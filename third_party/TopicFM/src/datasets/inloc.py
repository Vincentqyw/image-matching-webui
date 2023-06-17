import os
from torch.utils.data import Dataset

from src.utils.dataset import read_img_gray


class InLocDataset(Dataset):
    def __init__(self, img_path, match_list_path, img_resize=None, down_factor=16):
        self.img_path = img_path
        self.img_resize = img_resize
        self.down_factor = down_factor
        with open(match_list_path, 'r') as f:
            self.raw_pairs = f.readlines()
        print("number of matching pairs: ", len(self.raw_pairs))

    def __len__(self):
        return len(self.raw_pairs)

    def __getitem__(self, idx):
        raw_pair = self.raw_pairs[idx]
        image_name0, image_name1 = raw_pair.strip('\n').split(' ')
        path_img0 = os.path.join(self.img_path, image_name0)
        path_img1 = os.path.join(self.img_path, image_name1)
        img0, scale0 = read_img_gray(path_img0, resize=self.img_resize, down_factor=self.down_factor)
        img1, scale1 = read_img_gray(path_img1, resize=self.img_resize, down_factor=self.down_factor)
        return {"image0": img0, "image1": img1,
                "scale0": scale0, "scale1": scale1,
                "pair_names": (image_name0, image_name1),
                "dataset_name": "InLoc"}