import sys
sys.path.append('../')
import homography_est as hest
import numpy as np

#from misc.geometry_utils import warp_points

def warp_points(points, H):
    points = np.concatenate([points, np.ones_like(points, shape=points.shape[:-1] + (1,))], -1)
    warped_points = points @ H.T
    return warped_points[:,0:2] / warped_points[:,[2,2]]

pure_rotation = True
if pure_rotation:
    H_gt = np.random.rand(3, 3)
    H_gt, _ = np.linalg.qr(H_gt)
else:
    H_gt = np.random.rand(3, 3)

n_lines = 100
n_pts = 100
sigma_line = 0.1
sigma_pts = 0.1
n_outliers = 95

# Generate random lines
endpoints2 = np.random.rand(n_lines * 2, 2)
endpoints1 = warp_points(endpoints2, H_gt) + sigma_line * np.random.randn(n_lines * 2, 2)
if n_lines > 0:
    endpoints2[np.random.choice(n_lines, n_outliers, replace=False)] = np.random.rand(n_outliers, 2)

line_seg1 = [hest.LineSegment(l[0], l[1])
             for l in endpoints1.reshape(-1, 2, 2)]
line_seg2 = [hest.LineSegment(l[0], l[1])
             for l in endpoints2.reshape(-1, 2, 2)]

points2 = np.random.rand(n_pts, 2)
points1 = warp_points(points2, H_gt) + sigma_pts * np.random.randn(n_pts, 2)
if n_pts > 0:
    points2[np.random.choice(n_pts, n_outliers, replace=False)] = np.random.rand(n_outliers, 2)



tol_px = 1.0
H = hest.ransac_point_line_homography(points1, points2, line_seg1, line_seg2, tol_px, pure_rotation, [], [])

# The predicted H is 2 -> 1
if pure_rotation:
    print('R', H, '\nR_gt', H_gt)
else:
    print('H', H/H[2,2], '\nH_gt', H_gt/H_gt[2,2])
