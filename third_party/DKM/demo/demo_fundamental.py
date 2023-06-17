from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from dkm.utils.utils import tensor_to_pil
import cv2
from dkm import DKMv3_outdoor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/sacre_coeur_A.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/sacre_coeur_B.jpg", type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path

    # Create model
    dkm_model = DKMv3_outdoor(device=device)


    W_A, H_A = Image.open(im1_path).size
    W_B, H_B = Image.open(im2_path).size

    # Match
    warp, certainty = dkm_model.match(im1_path, im2_path, device=device)
    # Sample matches for estimation
    matches, certainty = dkm_model.sample(warp, certainty)
    kpts1, kpts2 = dkm_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)    
    F, mask = cv2.findFundamentalMat(
        kpts1.cpu().numpy(), kpts2.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
    )
    # TODO: some better visualization    