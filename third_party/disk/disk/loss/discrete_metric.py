import torch, math
import numpy as np
from torch_dimcheck import dimchecked

from disk import MatchedPairs, Image, NpArray
from disk.geom.epi import p_asymmdist_from_imgs

@dimchecked
def classify_pairs(kps1: ['N', 2], kps2: ['N', 2], img1: Image, img2: Image, th):
    '''
    classifies keypoint pairs as either possible or impossible under
    epipolar constraints
    '''

    epi_1_to_2 = p_asymmdist_from_imgs(kps1.T, kps2.T, img1, img2).abs()
    epi_2_to_1 = p_asymmdist_from_imgs(kps2.T, kps1.T, img2, img1).abs()

    return (epi_1_to_2 < th) & (epi_2_to_1 < th)

class DiscreteMetric(torch.nn.Module):
    def __init__(self, th=2., lm_kp=0., lm_tp=1., lm_fp=-0.25):
        super(DiscreteMetric, self).__init__() 

        self.th   = th
        self.lm_kp = lm_kp
        self.lm_tp = lm_tp
        self.lm_fp = lm_fp

    def forward(
        self,
        images : NpArray[Image],       # [N_scenes, N_per_scene]
        matches: NpArray[MatchedPairs] # [N_scenes, N_per_scene choose 2]
    ):
        N_scenes, N_per_scene = images.shape

        assert matches.shape[0] == N_scenes
        assert matches.shape[1] == ((N_per_scene - 1) * N_per_scene) // 2

        stats = np.zeros(matches.shape, dtype=object)

        for i_scene in range(N_scenes):
            i_match = 0
            scene_matches = matches[i_scene]
            scene_images  = images[i_scene]

            for i_image1 in range(N_per_scene):
                image1 = scene_images[i_image1]

                for i_image2 in range(i_image1+1, N_per_scene):
                    image2 = scene_images[i_image2]

                    stats[i_scene, i_match] = self._loss_one_pair(
                        scene_matches[i_match],
                        image1, image2
                    )

                    i_match += 1

        return stats

    def _loss_one_pair(self, pairs: MatchedPairs, img1: Image, img2: Image):
        n_kps   = pairs.kps1.shape[0] + pairs.kps2.shape[0]

        kps1 = pairs.kps1[pairs.matches[0]]
        kps2 = pairs.kps2[pairs.matches[1]]

        n_pairs = pairs.matches.shape[1]

        good = classify_pairs(kps1, kps2, img1, img2, th=self.th)
        bad  = ~good

        n_good = good.to(torch.int64).sum().item()
        n_bad  = bad.to(torch.int64).sum().item()
        prec   = n_good / (n_pairs + 1)

        reward = self.lm_tp * n_good  + \
                 self.lm_fp * n_bad   + \
                 self.lm_kp * n_kps

        stats = {
            'n_kps'    : n_kps,
            'n_pairs'  : n_pairs,
            'tp'       : n_good,
            'fp'       : n_bad,
            'reward'   : reward,
            'precision': prec,
        }

        return stats
