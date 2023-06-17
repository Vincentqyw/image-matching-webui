import torch, typing, cv2, sys
import numpy as np
import multiprocessing as mp
from torch_dimcheck import dimchecked
from typing import Dict

from disk import MatchedPairs, Image, NpArray, EstimationFailedError
from disk.loss.ransac import Ransac
from disk.geom import Pose, PoseError

CPU = torch.device('cpu')

class PoseQualityResult:
    def __init__(self, error: PoseError, n_inliers: int, success: bool = True):
        self.error     = error
        self.n_inliers = n_inliers
        self.success   = success

    def to_dict(self):
        return {
            **self.error.to_dict(),
            'n_inliers': self.n_inliers,
            'success'  : int(self.success),
        }

    def __str__(self):
        return (f'<PoseQualityResult error={self.error}, '
                f'n_inliers={self.n_inliers}, success={self.success}>')

# the error returned when pose estimation fails
FAILED_RESULT = PoseQualityResult(
    error=PoseError(
        # less than 90 to avoid creating an extra bin in histograms
        Δ_θ=89.95,
        Δ_T=179.95,
    ),
    n_inliers=0,
    success=False
)

class Job(typing.NamedTuple):
    matches: MatchedPairs
    K1     : torch.Tensor
    K2     : torch.Tensor
    pose1  : Pose
    pose2  : Pose
    ransac : Ransac

    def __call__(self):
        m = self.matches

        left  = m.kps1[m.matches[0]]
        right = m.kps2[m.matches[1]]

        try:
            pose_estimate, mask = self.ransac(left, right, self.K1, self.K2)
        except EstimationFailedError:
            return FAILED_RESULT

        gt_pose = Pose.relative(self.pose1, self.pose2, normed=True)
        error   = Pose.error(gt_pose, pose_estimate)

        n_inliers = mask.to(torch.int64).sum().item()

        return PoseQualityResult(
            error=error,
            n_inliers=n_inliers,
            success=True,
        )

    __repr__ = object.__repr__

    @staticmethod
    def execute(job):
        return job().to_dict()

class DummyPool:
    '''
    acts like multiprocessing.Pool and can be used to debug stuff
    '''
    def map(self, f, args):
        return [f(arg) for arg in args]

class PoseQuality:
    def __init__(self, ransac=Ransac(), dummy_pool=False, n_proc=6):
        self.ransac = ransac
        if dummy_pool:
            self.pool = DummyPool()
        else:
            self.pool = mp.Pool(processes=n_proc)

    def __call__(
        self,
        images: NpArray[Image],
        decisions: NpArray[MatchedPairs]
    )-> NpArray[Dict[str, float]]:

        N_scenes, N_per_scene = images.shape

        assert decisions.shape[0] == N_scenes
        assert decisions.shape[1] == ((N_per_scene - 1) * N_per_scene) // 2

        jobs = np.zeros(decisions.shape, dtype=object)

        for i_scene in range(N_scenes):
            i_decision = 0
            scene_decisions = decisions[i_scene]
            scene_images    = images[i_scene]

            for i_image1 in range(N_per_scene):
                image1 = scene_images[i_image1]
                pose1  = Pose.from_poselike(image1).to(CPU)
                K1     = image1.K.cpu()

                for i_image2 in range(i_image1+1, N_per_scene):
                    image2 = scene_images[i_image2]
                    pose2  = Pose.from_poselike(image2).to(CPU)
                    K2     = image2.K.cpu()

                    jobs[i_scene, i_decision] = Job(
                        scene_decisions[i_decision].to(CPU),
                        K1, K2,
                        pose1, pose2,
                        self.ransac
                    )

                    i_decision += 1

        return np.array(self.pool.map(Job.execute, jobs.flat)).reshape(*jobs.shape)
