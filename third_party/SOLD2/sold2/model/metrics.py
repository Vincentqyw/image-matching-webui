"""
This file implements the evaluation metrics.
"""
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.ops.boxes import batched_nms

from ..misc.geometry_utils import keypoints_to_grid


class Metrics(object):
    """ Metric evaluation calculator. """
    def __init__(self, detection_thresh, prob_thresh, grid_size,
                 junc_metric_lst=None, heatmap_metric_lst=None,
                 pr_metric_lst=None, desc_metric_lst=None):
        # List supported metrics
        self.supported_junc_metrics = ["junc_precision", "junc_precision_nms",
                                       "junc_recall", "junc_recall_nms"]
        self.supported_heatmap_metrics = ["heatmap_precision",
                                          "heatmap_recall"]
        self.supported_pr_metrics = ["junc_pr", "junc_nms_pr"]
        self.supported_desc_metrics = ["matching_score"]

        # If metric_lst is None, default to use all metrics
        if junc_metric_lst is None:
            self.junc_metric_lst = self.supported_junc_metrics
        else:
            self.junc_metric_lst = junc_metric_lst
        if heatmap_metric_lst is None:
            self.heatmap_metric_lst = self.supported_heatmap_metrics
        else:
            self.heatmap_metric_lst = heatmap_metric_lst
        if pr_metric_lst is None:
            self.pr_metric_lst = self.supported_pr_metrics
        else:
            self.pr_metric_lst = pr_metric_lst
        # For the descriptors, the default None assumes no desc metric at all
        if desc_metric_lst is None:
            self.desc_metric_lst = []
        elif desc_metric_lst == 'all':
            self.desc_metric_lst = self.supported_desc_metrics
        else:
            self.desc_metric_lst = desc_metric_lst

        if not self._check_metrics():
            raise ValueError(
                "[Error] Some elements in the metric_lst are invalid.")

        # Metric mapping table
        self.metric_table = {
            "junc_precision": junction_precision(detection_thresh),
            "junc_precision_nms": junction_precision(detection_thresh),
            "junc_recall": junction_recall(detection_thresh),
            "junc_recall_nms": junction_recall(detection_thresh),
            "heatmap_precision": heatmap_precision(prob_thresh),
            "heatmap_recall": heatmap_recall(prob_thresh),
            "junc_pr": junction_pr(),
            "junc_nms_pr": junction_pr(),
            "matching_score": matching_score(grid_size)
        }

        # Initialize the results
        self.metric_results = {}
        for key in self.metric_table.keys():
            self.metric_results[key] = 0.

    def evaluate(self, junc_pred, junc_pred_nms, junc_gt, heatmap_pred,
                 heatmap_gt, valid_mask, line_points1=None, line_points2=None,
                 desc_pred1=None, desc_pred2=None, valid_points=None):
        """ Perform evaluation. """
        for metric in self.junc_metric_lst:
            # If nms metrics then use nms to compute it.
            if "nms" in metric:
                junc_pred_input = junc_pred_nms
            # Use normal inputs instead.
            else:
                junc_pred_input = junc_pred
            self.metric_results[metric] = self.metric_table[metric](
                junc_pred_input, junc_gt, valid_mask)

        for metric in self.heatmap_metric_lst:
            self.metric_results[metric] = self.metric_table[metric](
                heatmap_pred, heatmap_gt, valid_mask)

        for metric in self.pr_metric_lst:
            if "nms" in metric:
                self.metric_results[metric] = self.metric_table[metric](
                    junc_pred_nms, junc_gt, valid_mask)
            else:
                self.metric_results[metric] = self.metric_table[metric](
                    junc_pred, junc_gt, valid_mask)

        for metric in self.desc_metric_lst:
            self.metric_results[metric] = self.metric_table[metric](
                line_points1, line_points2, desc_pred1,
                desc_pred2, valid_points)

    def _check_metrics(self):
        """ Check if all input metrics are valid. """
        flag = True
        for metric in self.junc_metric_lst:
            if not metric in self.supported_junc_metrics:
                flag = False
                break
        for metric in self.heatmap_metric_lst:
            if not metric in self.supported_heatmap_metrics:
                flag = False
                break
        for metric in self.desc_metric_lst:
            if not metric in self.supported_desc_metrics:
                flag = False
                break

        return flag


class AverageMeter(object):
    def __init__(self, junc_metric_lst=None, heatmap_metric_lst=None,
                 is_training=True, desc_metric_lst=None):
        # List supported metrics
        self.supported_junc_metrics = ["junc_precision", "junc_precision_nms",
                                       "junc_recall", "junc_recall_nms"]
        self.supported_heatmap_metrics = ["heatmap_precision",
                                          "heatmap_recall"]
        self.supported_pr_metrics = ["junc_pr", "junc_nms_pr"]
        self.supported_desc_metrics = ["matching_score"]
        # Record loss in training mode
        # if is_training:
        self.supported_loss = [
            "junc_loss", "heatmap_loss", "descriptor_loss", "total_loss"]

        self.is_training = is_training

        # If metric_lst is None, default to use all metrics
        if junc_metric_lst is None:
            self.junc_metric_lst = self.supported_junc_metrics
        else:
            self.junc_metric_lst = junc_metric_lst
        if heatmap_metric_lst is None:
            self.heatmap_metric_lst = self.supported_heatmap_metrics
        else:
            self.heatmap_metric_lst = heatmap_metric_lst
        # For the descriptors, the default None assumes no desc metric at all
        if desc_metric_lst is None:
            self.desc_metric_lst = []
        elif desc_metric_lst == 'all':
            self.desc_metric_lst = self.supported_desc_metrics
        else:
            self.desc_metric_lst = desc_metric_lst

        if not self._check_metrics():
            raise ValueError(
                "[Error] Some elements in the metric_lst are invalid.")

        # Initialize the results
        self.metric_results = {}
        for key in (self.supported_junc_metrics
                    + self.supported_heatmap_metrics
                    + self.supported_loss + self.supported_desc_metrics):
            self.metric_results[key] = 0.
        for key in self.supported_pr_metrics:
            zero_lst = [0 for _ in range(50)]
            self.metric_results[key] = {
                "tp": zero_lst,
                "tn": zero_lst,
                "fp": zero_lst,
                "fn": zero_lst,
                "precision": zero_lst,
                "recall": zero_lst
            }

        # Initialize total count
        self.count = 0

    def update(self, metrics, loss_dict=None, num_samples=1):
        # loss should be given in the training mode
        if self.is_training and (loss_dict is None):
            raise ValueError(
                "[Error] loss info should be given in the training mode.")

        # update total counts
        self.count += num_samples

        # update all the metrics
        for met in (self.supported_junc_metrics
                    + self.supported_heatmap_metrics
                    + self.supported_desc_metrics):
            self.metric_results[met] += (num_samples
                                         * metrics.metric_results[met])

        # Update all the losses
        for loss in loss_dict.keys():
            self.metric_results[loss] += num_samples * loss_dict[loss]

        # Update all pr counts
        for pr_met in self.supported_pr_metrics:
            # Update all tp, tn, fp, fn, precision, and recall.
            for key in metrics.metric_results[pr_met].keys():
                # Update each interval
                for idx in range(len(self.metric_results[pr_met][key])):
                    self.metric_results[pr_met][key][idx] += (
                        num_samples
                        * metrics.metric_results[pr_met][key][idx])

    def average(self):
        results = {}
        for met in self.metric_results.keys():
            # Skip pr curve metrics
            if not met in self.supported_pr_metrics:
                results[met] = self.metric_results[met] / self.count
            # Only update precision and recall in pr metrics
            else:
                met_results = {
                    "tp": self.metric_results[met]["tp"],
                    "tn": self.metric_results[met]["tn"],
                    "fp": self.metric_results[met]["fp"],
                    "fn": self.metric_results[met]["fn"],
                    "precision": [],
                    "recall": []
                }
                for idx in range(len(self.metric_results[met]["precision"])):
                    met_results["precision"].append(
                        self.metric_results[met]["precision"][idx]
                        / self.count)
                    met_results["recall"].append(
                        self.metric_results[met]["recall"][idx] / self.count)

                results[met] = met_results

        return results

    def _check_metrics(self):
        """ Check if all input metrics are valid. """
        flag = True
        for metric in self.junc_metric_lst:
            if not metric in self.supported_junc_metrics:
                flag = False
                break
        for metric in self.heatmap_metric_lst:
            if not metric in self.supported_heatmap_metrics:
                flag = False
                break
        for metric in self.desc_metric_lst:
            if not metric in self.supported_desc_metrics:
                flag = False
                break

        return flag


class junction_precision(object):
    """ Junction precision. """
    def __init__(self, detection_thresh):
        self.detection_thresh = detection_thresh

    # Compute the evaluation result
    def __call__(self, junc_pred, junc_gt, valid_mask):
        # Convert prediction to discrete detection
        junc_pred = (junc_pred >= self.detection_thresh).astype(np.int)
        junc_pred = junc_pred * valid_mask.squeeze()

        # Deal with the corner case of the prediction
        if np.sum(junc_pred) > 0:
            precision = (np.sum(junc_pred * junc_gt.squeeze())
                         / np.sum(junc_pred))
        else:
            precision = 0

        return float(precision)


class junction_recall(object):
    """ Junction recall. """
    def __init__(self, detection_thresh):
        self.detection_thresh = detection_thresh

    # Compute the evaluation result
    def __call__(self, junc_pred, junc_gt, valid_mask):
        # Convert prediction to discrete detection
        junc_pred = (junc_pred >= self.detection_thresh).astype(np.int)
        junc_pred = junc_pred * valid_mask.squeeze()

        # Deal with the corner case of the recall.
        if np.sum(junc_gt):
            recall = np.sum(junc_pred * junc_gt.squeeze()) / np.sum(junc_gt)
        else:
            recall = 0

        return float(recall)


class junction_pr(object):
    """ Junction precision-recall info. """
    def __init__(self, num_threshold=50):
        self.max = 0.4
        step = self.max / num_threshold
        self.min = step
        self.intervals = np.flip(np.arange(self.min, self.max + step, step))

    def __call__(self, junc_pred_raw, junc_gt, valid_mask):
        tp_lst = []
        fp_lst = []
        tn_lst = []
        fn_lst = []
        precision_lst = []
        recall_lst = []

        valid_mask = valid_mask.squeeze()
        # Iterate through all the thresholds
        for thresh in list(self.intervals):
            # Convert prediction to discrete detection
            junc_pred = (junc_pred_raw >= thresh).astype(np.int)
            junc_pred = junc_pred * valid_mask

            # Compute tp, fp, tn, fn
            junc_gt = junc_gt.squeeze()
            tp = np.sum(junc_pred * junc_gt)
            tn = np.sum((junc_pred == 0).astype(np.float)
                        * (junc_gt == 0).astype(np.float) * valid_mask)
            fp = np.sum((junc_pred == 1).astype(np.float)
                        * (junc_gt == 0).astype(np.float) * valid_mask)
            fn = np.sum((junc_pred == 0).astype(np.float)
                        * (junc_gt == 1).astype(np.float) * valid_mask)

            tp_lst.append(tp)
            tn_lst.append(tn)
            fp_lst.append(fp)
            fn_lst.append(fn)
            precision_lst.append(tp / (tp + fp))
            recall_lst.append(tp / (tp + fn))

        return {
            "tp": np.array(tp_lst),
            "tn": np.array(tn_lst),
            "fp": np.array(fp_lst),
            "fn": np.array(fn_lst),
            "precision": np.array(precision_lst),
            "recall": np.array(recall_lst)
        }


class heatmap_precision(object):
    """ Heatmap precision. """
    def __init__(self, prob_thresh):
        self.prob_thresh = prob_thresh

    def __call__(self, heatmap_pred, heatmap_gt, valid_mask):
        # Assume NHWC (Handle L1 and L2 cases) NxHxWx1
        heatmap_pred = np.squeeze(heatmap_pred > self.prob_thresh)
        heatmap_pred = heatmap_pred * valid_mask.squeeze()

        # Deal with the corner case of the prediction
        if np.sum(heatmap_pred) > 0:
            precision = (np.sum(heatmap_pred * heatmap_gt.squeeze())
                         / np.sum(heatmap_pred))
        else:
            precision = 0.

        return precision


class heatmap_recall(object):
    """ Heatmap recall. """
    def __init__(self, prob_thresh):
        self.prob_thresh = prob_thresh

    def __call__(self, heatmap_pred, heatmap_gt, valid_mask):
        # Assume NHWC (Handle L1 and L2 cases) NxHxWx1
        heatmap_pred = np.squeeze(heatmap_pred > self.prob_thresh)
        heatmap_pred = heatmap_pred * valid_mask.squeeze()

        # Deal with the corner case of the ground truth
        if np.sum(heatmap_gt) > 0:
            recall = (np.sum(heatmap_pred * heatmap_gt.squeeze())
                      / np.sum(heatmap_gt))
        else:
            recall = 0.

        return recall


class matching_score(object):
    """ Descriptors matching score. """
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def __call__(self, points1, points2, desc_pred1,
                 desc_pred2, line_indices):
        b_size, _, Hc, Wc = desc_pred1.size()
        img_size = (Hc * self.grid_size, Wc * self.grid_size)
        device = desc_pred1.device

        # Extract valid keypoints
        n_points = line_indices.size()[1]
        valid_points = line_indices.bool().flatten()
        n_correct_points = torch.sum(valid_points).item()
        if n_correct_points == 0:
            return torch.tensor(0., dtype=torch.float, device=device)

        # Convert the keypoints to a grid suitable for interpolation
        grid1 = keypoints_to_grid(points1, img_size)
        grid2 = keypoints_to_grid(points2, img_size)

        # Extract the descriptors
        desc1 = F.grid_sample(desc_pred1, grid1).permute(
            0, 2, 3, 1).reshape(b_size * n_points, -1)[valid_points]
        desc1 = F.normalize(desc1, dim=1)
        desc2 = F.grid_sample(desc_pred2, grid2).permute(
            0, 2, 3, 1).reshape(b_size * n_points, -1)[valid_points]
        desc2 = F.normalize(desc2, dim=1)
        desc_dists = 2 - 2 * (desc1 @ desc2.t())

        # Compute percentage of correct matches
        matches0 = torch.min(desc_dists, dim=1)[1]
        matches1 = torch.min(desc_dists, dim=0)[1]
        matching_score = (matches1[matches0]
                          == torch.arange(len(matches0)).to(device))
        matching_score = matching_score.float().mean()
        return matching_score


def super_nms(prob_predictions, dist_thresh, prob_thresh=0.01, top_k=0):
    """ Non-maximum suppression adapted from SuperPoint. """
    # Iterate through batch dimension
    im_h = prob_predictions.shape[1]
    im_w = prob_predictions.shape[2]
    output_lst = []
    for i in range(prob_predictions.shape[0]):
        # print(i)
        prob_pred = prob_predictions[i, ...]
        # Filter the points using prob_thresh
        coord = np.where(prob_pred >= prob_thresh) # HW format
        points = np.concatenate((coord[0][..., None], coord[1][..., None]),
                                axis=1) # HW format

        # Get the probability score
        prob_score = prob_pred[points[:, 0], points[:, 1]]

        # Perform super nms
        # Modify the in_points to xy format (instead of HW format)
        in_points = np.concatenate((coord[1][..., None], coord[0][..., None],
                                    prob_score), axis=1).T
        keep_points_, keep_inds = nms_fast(in_points, im_h, im_w, dist_thresh)
        # Remember to flip outputs back to HW format
        keep_points = np.round(np.flip(keep_points_[:2, :], axis=0).T)
        keep_score = keep_points_[-1, :].T

        # Whether we only keep the topk value
        if (top_k > 0) or (top_k is None):
            k = min([keep_points.shape[0], top_k])
            keep_points = keep_points[:k, :]
            keep_score = keep_score[:k]

        # Re-compose the probability map
        output_map = np.zeros([im_h, im_w])
        output_map[keep_points[:, 0].astype(np.int),
                   keep_points[:, 1].astype(np.int)] = keep_score.squeeze()

        output_lst.append(output_map[None, ...])

    return np.concatenate(output_lst, axis=0)


def nms_fast(in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T

    Algo summary: Create a grid sized HxW. Assign each corner location a 1,
    rest are zeros. Iterate through all the 1's and convert them to -1 or 0.
    Suppress points by setting nearby values to 0.

    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).

    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundary.

    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinite distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds
