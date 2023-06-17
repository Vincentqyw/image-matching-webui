import torch, pydegensac, cv2, typing

from torch_dimcheck import dimchecked

from disk import EstimationFailedError
from disk.geom import Pose

@dimchecked
def _recover_pose(E: [3, 3], i_coords: ['N', 2], j_coords: ['N', 2]):
    E_ = E.to(torch.float64).numpy()
    i_coords_ = i_coords.to(torch.float64).numpy()
    j_coords_ = j_coords.to(torch.float64).numpy()

    # this function seems to only take f64 arguments, that's why
    # all the casting above
    n_inliers, R, T, inlier_mask = cv2.recoverPose(
        E_,
        i_coords_,
        j_coords_,
    )

    R = torch.from_numpy(R).to(torch.float32)
    # T is returned as a 3x1 array for some reason
    T = torch.from_numpy(T).to(torch.float32).squeeze(1)

    return Pose(R, T)

@dimchecked
def _normalize_coords(coords: ['N', 2], K: [3, 3]) -> ['N', 2]:
    coords = coords.to(torch.float32)

    f = torch.tensor([[K[0, 0], K[1, 1]]])
    c = torch.tensor([[K[0, 2], K[1, 2]]])

    return (coords - c) / f

class Ransac(typing.NamedTuple):
    reprojection_threshold: float = 1.
    confidence            : float = 0.9999
    max_iters             : int   = 10_000
    candidate_threshold   : int   = 10

    @dimchecked
    def __call__(
        self,
        left: ['N', 2],
        right: ['N', 2],
        K1: [3, 3],
        K2: [3, 3]
    ):
        left  = left.cpu()
        right = right.cpu()
        K1 = K1.cpu()
        K2 = K2.cpu()

        if left.shape[0] < self.candidate_threshold:
            raise EstimationFailedError()

        F, mask = pydegensac.findFundamentalMatrix(
            left.numpy(),
            right.numpy(),
            px_th=self.reprojection_threshold,
            conf=self.confidence,
            max_iters=self.max_iters
        )

        # FIXME: how does pydegensac handle failure?
        if mask is None:
            raise EstimationFailedError()

        mask = torch.from_numpy(mask)
        F    = torch.from_numpy(F).to(torch.float32)

        E = K2.T @ F @ K1

        try:
            pose = _recover_pose(
                E,
                _normalize_coords( left[mask], K1),
                _normalize_coords(right[mask], K2),
            )
        except cv2.error:
            raise EstimationFailedError()

        return pose, mask
