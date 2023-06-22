"""
Implementation of the line matching methods.
"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from ..misc.geometry_utils import keypoints_to_grid


class WunschLineMatcher(object):
    """ Class matching two sets of line segments
        with the Needleman-Wunsch algorithm. """
    def __init__(self, cross_check=True, num_samples=10, min_dist_pts=8,
                 top_k_candidates=10, grid_size=8, sampling="regular",
                 line_score=False):
        self.cross_check = cross_check
        self.num_samples = num_samples
        self.min_dist_pts = min_dist_pts
        self.top_k_candidates = top_k_candidates
        self.grid_size = grid_size
        self.line_score = line_score  # True to compute saliency on a line
        self.sampling_mode = sampling
        if sampling not in ["regular", "d2_net", "asl_feat"]:
            raise ValueError("Wrong sampling mode: " + sampling)

    def forward(self, line_seg1, line_seg2, desc1, desc2):
        """
            Find the best matches between two sets of line segments
            and their corresponding descriptors.
        """
        img_size1 = (desc1.shape[2] * self.grid_size,
                     desc1.shape[3] * self.grid_size)
        img_size2 = (desc2.shape[2] * self.grid_size,
                     desc2.shape[3] * self.grid_size)
        device = desc1.device

        # Default case when an image has no lines
        if len(line_seg1) == 0:
            return np.empty((0), dtype=int)
        if len(line_seg2) == 0:
            return -np.ones(len(line_seg1), dtype=int)

        # Sample points regularly along each line
        if self.sampling_mode == "regular":
            line_points1, valid_points1 = self.sample_line_points(line_seg1)
            line_points2, valid_points2 = self.sample_line_points(line_seg2)
        else:
            line_points1, valid_points1 = self.sample_salient_points(
                line_seg1, desc1, img_size1, self.sampling_mode)
            line_points2, valid_points2 = self.sample_salient_points(
                line_seg2, desc2, img_size2, self.sampling_mode)
        line_points1 = torch.tensor(line_points1.reshape(-1, 2),
                                    dtype=torch.float, device=device)
        line_points2 = torch.tensor(line_points2.reshape(-1, 2),
                                    dtype=torch.float, device=device)

        # Extract the descriptors for each point
        grid1 = keypoints_to_grid(line_points1, img_size1)
        grid2 = keypoints_to_grid(line_points2, img_size2)
        desc1 = F.normalize(F.grid_sample(desc1, grid1)[0, :, :, 0], dim=0)
        desc2 = F.normalize(F.grid_sample(desc2, grid2)[0, :, :, 0], dim=0)

        # Precompute the distance between line points for every pair of lines
        # Assign a score of -1 for unvalid points
        scores = desc1.t() @ desc2
        scores[~valid_points1.flatten()] = -1
        scores[:, ~valid_points2.flatten()] = -1
        scores = scores.reshape(len(line_seg1), self.num_samples,
                                len(line_seg2), self.num_samples)
        scores = scores.permute(0, 2, 1, 3)
        # scores.shape = (n_lines1, n_lines2, num_samples, num_samples)

        # Pre-filter the line candidates and find the best match for each line
        matches = self.filter_and_match_lines(scores)

        # [Optionally] filter matches with mutual nearest neighbor filtering
        if self.cross_check:
            matches2 = self.filter_and_match_lines(
                scores.permute(1, 0, 3, 2))
            mutual = matches2[matches] == np.arange(len(line_seg1))
            matches[~mutual] = -1

        return matches

    def d2_net_saliency_score(self, desc):
        """ Compute the D2-Net saliency score
            on a 3D or 4D descriptor. """
        is_3d = len(desc.shape) == 3
        b_size = len(desc)
        feat = F.relu(desc)

        # Compute the soft local max
        exp = torch.exp(feat)
        if is_3d:
            sum_exp = 3 * F.avg_pool1d(exp, kernel_size=3, stride=1,
                                       padding=1)
        else:
            sum_exp = 9 * F.avg_pool2d(exp, kernel_size=3, stride=1,
                                       padding=1)
        soft_local_max = exp / sum_exp

        # Compute the depth-wise maximum
        depth_wise_max = torch.max(feat, dim=1)[0]
        depth_wise_max = feat / depth_wise_max.unsqueeze(1)

        # Total saliency score
        score = torch.max(soft_local_max * depth_wise_max, dim=1)[0]
        normalization = torch.sum(score.reshape(b_size, -1), dim=1)
        if is_3d:
            normalization = normalization.reshape(b_size, 1)
        else:
            normalization = normalization.reshape(b_size, 1, 1)
        score = score / normalization
        return score

    def asl_feat_saliency_score(self, desc):
        """ Compute the ASLFeat saliency score on a 3D or 4D descriptor. """
        is_3d = len(desc.shape) == 3
        b_size = len(desc)

        # Compute the soft local peakiness
        if is_3d:
            local_avg = F.avg_pool1d(desc, kernel_size=3, stride=1, padding=1)
        else:
            local_avg = F.avg_pool2d(desc, kernel_size=3, stride=1, padding=1)
        soft_local_score = F.softplus(desc - local_avg)

        # Compute the depth-wise peakiness
        depth_wise_mean = torch.mean(desc, dim=1).unsqueeze(1)
        depth_wise_score = F.softplus(desc - depth_wise_mean)

        # Total saliency score
        score = torch.max(soft_local_score * depth_wise_score, dim=1)[0]
        normalization = torch.sum(score.reshape(b_size, -1), dim=1)
        if is_3d:
            normalization = normalization.reshape(b_size, 1)
        else:
            normalization = normalization.reshape(b_size, 1, 1)
        score = score / normalization
        return score

    def sample_salient_points(self, line_seg, desc, img_size,
                              saliency_type='d2_net'):
        """
        Sample the most salient points along each line segments, with a
        minimal distance between each point. Pad the remaining points.
        Inputs:
            line_seg: an Nx2x2 torch.Tensor.
            desc: a NxDxHxW torch.Tensor.
            image_size: the original image size.
            saliency_type: 'd2_net' or 'asl_feat'.
        Outputs:
            line_points: an Nxnum_samplesx2 np.array.
            valid_points: a boolean Nxnum_samples np.array.
        """
        device = desc.device
        if not self.line_score:
            # Compute the score map
            if saliency_type == "d2_net":
                score = self.d2_net_saliency_score(desc)
            else:
                score = self.asl_feat_saliency_score(desc)

        num_lines = len(line_seg)
        line_lengths = np.linalg.norm(line_seg[:, 0] - line_seg[:, 1], axis=1)

        # The number of samples depends on the length of the line
        num_samples_lst = np.clip(line_lengths // self.min_dist_pts,
                                  2, self.num_samples)
        line_points = np.empty((num_lines, self.num_samples, 2), dtype=float)
        valid_points = np.empty((num_lines, self.num_samples), dtype=bool)

        # Sample the score on a fixed number of points of each line
        n_samples_per_region = 4
        for n in np.arange(2, self.num_samples + 1):
            sample_rate = n * n_samples_per_region
            # Consider all lines where we can fit up to n points
            cur_mask = num_samples_lst == n
            cur_line_seg = line_seg[cur_mask]
            cur_num_lines = len(cur_line_seg)
            if cur_num_lines == 0:
                continue
            line_points_x = np.linspace(cur_line_seg[:, 0, 0],
                                        cur_line_seg[:, 1, 0],
                                        sample_rate, axis=-1)
            line_points_y = np.linspace(cur_line_seg[:, 0, 1],
                                        cur_line_seg[:, 1, 1],
                                        sample_rate, axis=-1)
            cur_line_points = np.stack([line_points_x, line_points_y],
                                       axis=-1).reshape(-1, 2)
            # cur_line_points is of shape (n_cur_lines * sample_rate, 2)
            cur_line_points = torch.tensor(cur_line_points, dtype=torch.float,
                                           device=device)
            grid_points = keypoints_to_grid(cur_line_points, img_size)

            if self.line_score:
                # The saliency score is high when the activation are locally
                # maximal along the line (and not in a square neigborhood)
                line_desc = F.grid_sample(desc, grid_points).squeeze()
                line_desc = line_desc.reshape(-1, cur_num_lines, sample_rate)
                line_desc = line_desc.permute(1, 0, 2)
                if saliency_type == "d2_net":
                    scores = self.d2_net_saliency_score(line_desc)
                else:
                    scores = self.asl_feat_saliency_score(line_desc)
            else:
                scores = F.grid_sample(score.unsqueeze(1),
                                       grid_points).squeeze()

            # Take the most salient point in n distinct regions
            scores = scores.reshape(-1, n, n_samples_per_region)
            best = torch.max(scores, dim=2, keepdim=True)[1].cpu().numpy()
            cur_line_points = cur_line_points.reshape(-1, n,
                                                      n_samples_per_region, 2)
            cur_line_points = np.take_along_axis(
                cur_line_points, best[..., None], axis=2)[:, :, 0]

            # Pad
            cur_valid_points = np.ones((cur_num_lines, self.num_samples),
                                       dtype=bool)
            cur_valid_points[:, n:] = False
            cur_line_points = np.concatenate([
                cur_line_points,
                np.zeros((cur_num_lines, self.num_samples - n, 2), dtype=float)],
                axis=1)

            line_points[cur_mask] = cur_line_points
            valid_points[cur_mask] = cur_valid_points

        return line_points, valid_points

    def sample_line_points(self, line_seg):
        """
        Regularly sample points along each line segments, with a minimal
        distance between each point. Pad the remaining points.
        Inputs:
            line_seg: an Nx2x2 torch.Tensor.
        Outputs:
            line_points: an Nxnum_samplesx2 np.array.
            valid_points: a boolean Nxnum_samples np.array.
        """
        num_lines = len(line_seg)
        line_lengths = np.linalg.norm(line_seg[:, 0] - line_seg[:, 1], axis=1)

        # Sample the points separated by at least min_dist_pts along each line
        # The number of samples depends on the length of the line
        num_samples_lst = np.clip(line_lengths // self.min_dist_pts,
                                  2, self.num_samples)
        line_points = np.empty((num_lines, self.num_samples, 2), dtype=float)
        valid_points = np.empty((num_lines, self.num_samples), dtype=bool)
        for n in np.arange(2, self.num_samples + 1):
            # Consider all lines where we can fit up to n points
            cur_mask = num_samples_lst == n
            cur_line_seg = line_seg[cur_mask]
            line_points_x = np.linspace(cur_line_seg[:, 0, 0],
                                        cur_line_seg[:, 1, 0],
                                        n, axis=-1)
            line_points_y = np.linspace(cur_line_seg[:, 0, 1],
                                        cur_line_seg[:, 1, 1],
                                        n, axis=-1)
            cur_line_points = np.stack([line_points_x, line_points_y], axis=-1)

            # Pad
            cur_num_lines = len(cur_line_seg)
            cur_valid_points = np.ones((cur_num_lines, self.num_samples),
                                       dtype=bool)
            cur_valid_points[:, n:] = False
            cur_line_points = np.concatenate([
                cur_line_points,
                np.zeros((cur_num_lines, self.num_samples - n, 2), dtype=float)],
                axis=1)

            line_points[cur_mask] = cur_line_points
            valid_points[cur_mask] = cur_valid_points

        return line_points, valid_points

    def filter_and_match_lines(self, scores):
        """
        Use the scores to keep the top k best lines, compute the Needleman-
        Wunsch algorithm on each candidate pairs, and keep the highest score.
        Inputs:
            scores: a (N, M, n, n) torch.Tensor containing the pairwise scores
                    of the elements to match.
        Outputs:
            matches: a (N) np.array containing the indices of the best match
        """
        # Pre-filter the pairs and keep the top k best candidate lines
        line_scores1 = scores.max(3)[0]
        valid_scores1 = line_scores1 != -1
        line_scores1 = ((line_scores1 * valid_scores1).sum(2)
                        / valid_scores1.sum(2))
        line_scores2 = scores.max(2)[0]
        valid_scores2 = line_scores2 != -1
        line_scores2 = ((line_scores2 * valid_scores2).sum(2)
                        / valid_scores2.sum(2))
        line_scores = (line_scores1 + line_scores2) / 2
        topk_lines = torch.argsort(line_scores,
                                dim=1)[:, -self.top_k_candidates:]
        scores, topk_lines = scores.cpu().numpy(), topk_lines.cpu().numpy()
        # topk_lines.shape = (n_lines1, top_k_candidates)
        top_scores = np.take_along_axis(scores, topk_lines[:, :, None, None],
                                        axis=1)

        # Consider the reversed line segments as well
        top_scores = np.concatenate([top_scores, top_scores[..., ::-1]],
                                    axis=1)

        # Compute the line distance matrix with Needleman-Wunsch algo and
        # retrieve the closest line neighbor
        n_lines1, top2k, n, m = top_scores.shape
        top_scores = top_scores.reshape(n_lines1 * top2k, n, m)
        nw_scores = self.needleman_wunsch(top_scores)
        nw_scores = nw_scores.reshape(n_lines1, top2k)
        matches = np.mod(np.argmax(nw_scores, axis=1), top2k // 2)
        matches = topk_lines[np.arange(n_lines1), matches]
        return matches

    def needleman_wunsch(self, scores):
        """
        Batched implementation of the Needleman-Wunsch algorithm.
        The cost of the InDel operation is set to 0 by subtracting the gap
        penalty to the scores.
        Inputs:
            scores: a (B, N, M) np.array containing the pairwise scores
                    of the elements to match.
        """
        b, n, m = scores.shape

        # Recalibrate the scores to get a gap score of 0
        gap = 0.1
        nw_scores = scores - gap

        # Run the dynamic programming algorithm
        nw_grid = np.zeros((b, n + 1, m + 1), dtype=float)
        for i in range(n):
            for j in range(m):
                nw_grid[:, i + 1, j + 1] = np.maximum(
                    np.maximum(nw_grid[:, i + 1, j], nw_grid[:, i, j + 1]),
                    nw_grid[:, i, j] + nw_scores[:, i, j])

        return nw_grid[:, -1, -1]

    def get_pairwise_distance(self, line_seg1, line_seg2, desc1, desc2):
        """
            Compute the OPPOSITE of the NW score for pairs of line segments
            and their corresponding descriptors.
        """
        num_lines = len(line_seg1)
        assert num_lines == len(line_seg2), "The same number of lines is required in pairwise score."
        img_size1 = (desc1.shape[2] * self.grid_size,
                     desc1.shape[3] * self.grid_size)
        img_size2 = (desc2.shape[2] * self.grid_size,
                     desc2.shape[3] * self.grid_size)
        device = desc1.device

        # Sample points regularly along each line
        line_points1, valid_points1 = self.sample_line_points(line_seg1)
        line_points2, valid_points2 = self.sample_line_points(line_seg2)
        line_points1 = torch.tensor(line_points1.reshape(-1, 2),
                                    dtype=torch.float, device=device)
        line_points2 = torch.tensor(line_points2.reshape(-1, 2),
                                    dtype=torch.float, device=device)

        # Extract the descriptors for each point
        grid1 = keypoints_to_grid(line_points1, img_size1)
        grid2 = keypoints_to_grid(line_points2, img_size2)
        desc1 = F.normalize(F.grid_sample(desc1, grid1)[0, :, :, 0], dim=0)
        desc1 = desc1.reshape(-1, num_lines, self.num_samples)
        desc2 = F.normalize(F.grid_sample(desc2, grid2)[0, :, :, 0], dim=0)
        desc2 = desc2.reshape(-1, num_lines, self.num_samples)

        # Compute the distance between line points for every pair of lines
        # Assign a score of -1 for unvalid points
        scores = torch.einsum('dns,dnt->nst', desc1, desc2).cpu().numpy()
        scores = scores.reshape(num_lines * self.num_samples,
                                self.num_samples)
        scores[~valid_points1.flatten()] = -1
        scores = scores.reshape(num_lines, self.num_samples, self.num_samples)
        scores = scores.transpose(1, 0, 2).reshape(self.num_samples, -1)
        scores[:, ~valid_points2.flatten()] = -1
        scores = scores.reshape(self.num_samples, num_lines, self.num_samples)
        scores = scores.transpose(1, 0, 2)
        # scores.shape = (num_lines, num_samples, num_samples)

        # Compute the NW score for each pair of lines
        pairwise_scores = np.array([self.needleman_wunsch(s) for s in scores])
        return -pairwise_scores
