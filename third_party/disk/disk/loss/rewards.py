import torch
from torch_dimcheck import dimchecked

from disk import Image
from disk.geom.epi import asymmdist_from_imgs

class EpipolarReward:
    def __init__(self, th=2., lm_tp=1., lm_fp=-0.25):
        self.th   = th
        self.lm_tp = lm_tp
        self.lm_fp = lm_fp

    @dimchecked
    def __call__(
        self,
        kps1: ['N', 2],
        kps2: ['M', 2],
        img1: Image,
        img2: Image
    ) -> ['N', 'M']:
        '''
        assigns all pairs of keypoints across (kps1, kps2) a reward depending
        if the are correct or incorrect under epipolar constraints
        '''
        good = self.classify(kps1, kps2, img1, img2)
        return self.lm_tp * good + self.lm_fp * (~good)

    @dimchecked
    def classify(
        self,
        kps1: ['N', 2],
        kps2: ['M', 2],
        img1: Image,
        img2: Image,
    ) -> ['N', 'M']:
        '''
        classifies all pairs of keypoints across (kps1, kps2) as correct or
        incorrect depending on epipolar error
        '''

        epi_1_to_2 = asymmdist_from_imgs(kps1.T, kps2.T, img1, img2).abs()
        epi_2_to_1 = asymmdist_from_imgs(kps2.T, kps1.T, img2, img1).abs()

        # the distance is asymmetric, so we check if both 2_to_1 is
        # correct and 1_to_2.
        return (epi_1_to_2 < self.th) & (epi_2_to_1 < self.th).T

class DepthReward:
    def __init__(self, th=2., lm_tp=1., lm_fp=-0.25):
        self.th   = th 
        self.lm_tp = lm_tp
        self.lm_fp = lm_fp

        self._epipolar = EpipolarReward(th=th)

    @dimchecked
    def __call__(
        self,
        kps1: ['N', 2],
        kps2: ['M', 2],
        img1: Image,
        img2: Image
    ) -> ['N', 'M']:
        '''
        classifies all (kp1, kp2) pairs as either
        * correct  : within dist_α in reprojection
        * incorrect: above dist_α away in epipolar constraints
        * unknown  : no depth is available and is not incorrect

        and assigns them rewards according to DepthReward parameters
        '''

        # reproject to the other image.
        kps1_r = img2.project(img1.unproject(kps1.T)) # [2, N]
        kps2_r = img1.project(img2.unproject(kps2.T)) # [2, M]

        # compute pixel-space differences between (kp1, repr(kp2))
        # and (repr(kp1), kp2)
        diff1 = kps2_r[:, None, :] - kps1.T[:, :, None] # [2, N, M]
        diff2 = kps1_r[:, :, None] - kps2.T[:, None, :] # [2, N, M]

        # NaNs indicate we had no depth available at this location
        has_depth = (torch.isfinite(diff1) & torch.isfinite(diff2)).all(dim=0)

        # threshold the distances
        close1    = torch.norm(diff1, p=2, dim=0) < self.th
        close2    = torch.norm(diff2, p=2, dim=0) < self.th
        
        epi_bad    = ~self._epipolar.classify(kps1, kps2, img1, img2)
        good_pairs = close1 & close2

        return self.lm_tp * good_pairs + self.lm_fp * epi_bad
