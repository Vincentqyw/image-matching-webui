import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.ASpanFormer.aspanformer import ASpanFormer 
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config
import demo_utils 

import cv2
import torch
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='../configs/aspan/outdoor/aspan_test.py',
  help='path for config file.')
parser.add_argument('--img0_path', type=str, default='../assets/phototourism_sample_images/piazza_san_marco_06795901_3725050516.jpg',
  help='path for image0.')
parser.add_argument('--img1_path', type=str, default='../assets/phototourism_sample_images/piazza_san_marco_15148634_5228701572.jpg',
  help='path for image1.')
parser.add_argument('--weights_path', type=str, default='../weights/outdoor.ckpt',
  help='path for model weights.')
parser.add_argument('--long_dim0', type=int, default=1024,
  help='resize for longest dim of image0.')
parser.add_argument('--long_dim1', type=int, default=1024,
  help='resize for longest dim of image1.')

args = parser.parse_args()


if __name__=='__main__':
    config = get_cfg_defaults()
    config.merge_from_file(args.config_path)
    _config = lower_config(config)
    matcher = ASpanFormer(config=_config['aspan'])
    state_dict = torch.load(args.weights_path, map_location='cpu')['state_dict']
    matcher.load_state_dict(state_dict,strict=False)
    matcher.cuda(),matcher.eval()

    img0,img1=cv2.imread(args.img0_path),cv2.imread(args.img1_path)
    img0_g,img1_g=cv2.imread(args.img0_path,0),cv2.imread(args.img1_path,0)
    img0,img1=demo_utils.resize(img0,args.long_dim0),demo_utils.resize(img1,args.long_dim1)
    img0_g,img1_g=demo_utils.resize(img0_g,args.long_dim0),demo_utils.resize(img1_g,args.long_dim1)
    data={'image0':torch.from_numpy(img0_g/255.)[None,None].cuda().float(),
          'image1':torch.from_numpy(img1_g/255.)[None,None].cuda().float()} 
    with torch.no_grad():   
      matcher(data,online_resize=True)
      corr0,corr1=data['mkpts0_f'].cpu().numpy(),data['mkpts1_f'].cpu().numpy()

    F_hat,mask_F=cv2.findFundamentalMat(corr0,corr1,method=cv2.FM_RANSAC,ransacReprojThreshold=1)
    if mask_F is not None:
      mask_F=mask_F[:,0].astype(bool) 
    else:
      mask_F=np.zeros_like(corr0[:,0]).astype(bool)

    #visualize match
    display=demo_utils.draw_match(img0,img1,corr0,corr1)
    display_ransac=demo_utils.draw_match(img0,img1,corr0[mask_F],corr1[mask_F])
    cv2.imwrite('match.png',display)
    cv2.imwrite('match_ransac.png',display_ransac)
    print(len(corr1),len(corr1[mask_F]))