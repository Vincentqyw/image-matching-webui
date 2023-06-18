import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / '../../third_party'))
from DKM.dkm import DKMv3_outdoor

dkm_path = Path(__file__).parent / '../../third_party/DKM'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# "DKMv3": {
#     "outdoor": "https://github.com/Parskatt/storage/releases/download/dkmv3/DKMv3_outdoor.pth",
#     "indoor": "https://github.com/Parskatt/storage/releases/download/dkmv3/DKMv3_indoor.pth",
# },

class DKMv3(BaseModel):
    default_conf = {
        'weights': 'outdoor',
        'match_threshold': 0.2,
        'checkpoint_dir': dkm_path / 'pretrained',
    }
    required_inputs = [
        'image0',
        'image1',
    ]
    def _init(self, conf):
        path_to_weights = conf['checkpoint_dir'] / f'DKMv3_{conf["weights"]}.pth'
        self.net = DKMv3_outdoor(path_to_weights = str(path_to_weights), device=device)
    def _forward(self, data):
        img0 = data['image0'].cpu().numpy().squeeze() * 255
        img1 = data['image1'].cpu().numpy().squeeze() * 255
        img0 = img0.transpose(1, 2, 0)
        img1 = img1.transpose(1, 2, 0)
        img0 = Image.fromarray(img0.astype('uint8'))
        img1 = Image.fromarray(img1.astype('uint8'))
        W_A, H_A = img0.size
        W_B, H_B = img1.size

        warp, certainty = self.net.match(img0, img1, device=device)
        matches, certainty = self.net.sample(warp, certainty)
        kpts1, kpts2 = self.net.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)    
        pred = {}
        pred['keypoints0'], pred['keypoints1'] = kpts1, kpts2
        return pred
