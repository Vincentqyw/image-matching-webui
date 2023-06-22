import numpy as np
import cv2
import pytlbd


ETH_EPS = 1e-10

class PyTLBD(object):
    def __init__(self):
        pass

    @staticmethod
    def to_multiscale_lines(lines):
        ms_lines = []
        for l in lines.reshape(-1, 4):
            ll = np.append(l, [0, np.linalg.norm(l[:2] - l[2:4])])
            ms_lines.append([(0, ll)] + [(i, ll / (i * np.sqrt(2))) for i in range(1, 5)])
        return ms_lines

    @staticmethod
    def get_lbg_descrs(img, lines):
        ##########################################################################
        ms_lines = PyTLBD.to_multiscale_lines(lines)
        pyramid = get_img_pyramid(img)
        descriptors = pytlbd.lbd_multiscale_pyr(pyramid, ms_lines, 9, 7)
        return descriptors

    @staticmethod
    def match_lbd_hellinger(descriptors0, descriptors1, ratio_th=0.85):
        # Get the distance matrix between descriptors
        D = multiscale_helinger_dist(descriptors0, descriptors1)
        # Compute the ratio-distance matches
        _, ratio_dists = matlab_like_desc_distance(D)
        # Apply the ratio test
        ratio_mask = ratio_dists > (1 / ratio_th)
        ratio_sort_indices = np.argsort(-ratio_dists[ratio_mask])
        matches = np.argwhere(ratio_mask)[ratio_sort_indices]

        # Check that the matches are Mutual Nearest Neighbours
        argmin_rows = D.argmin(0)
        argmin_cols = D.argmin(1)
        mnn_cols = argmin_cols[matches[:, 0]] == matches[:, 1]
        mnn_rows = argmin_rows[matches[:, 1]] == matches[:, 0]
        mnn = np.logical_and(mnn_cols, mnn_rows)
        my_matches = matches[mnn]

        pred_matches = np.full(len(descriptors0), -1, dtype=int)
        pred_matches[my_matches[:, 0]] = my_matches[:, 1]
        return pred_matches

    def compute_descriptors(self, img, lines):
        # Compute multi-scale descriptors
        desc = self.get_lbg_descrs(img, lines.reshape(-1, 4))
        return np.array(desc)
    
    def match_lines(self, lines0, lines1, desc0, desc1):
        # Find matches using the heuristic approach defined in the paper
        multiscale_lines0 = PyTLBD.to_multiscale_lines(lines0)
        multiscale_lines1 = PyTLBD.to_multiscale_lines(lines1)
        try:
            my_matches = np.array(pytlbd.lbd_matching_multiscale(
                multiscale_lines0, multiscale_lines1,
                list(desc0), list(desc1)))
            pred_matches = -np.ones((len(desc0)), dtype=int)
            if len(my_matches) > 0:
                pred_matches[my_matches[:, 0].astype(np.int32)] = my_matches[:, 1].astype(np.int32)
            return pred_matches
        except RuntimeError:
            return -np.ones((len(desc0)), dtype=int)


### Util functions for LBD heuristic matcher

def get_img_pyramid(img, n_levels=5, level_scale=np.sqrt(2)):
    octave_img = img.copy()
    pre_sigma2 = 0
    cur_sigma2 = 1.0
    pyramid = []
    for i in range(n_levels):
        increase_sigma = np.sqrt(cur_sigma2 - pre_sigma2)
        blurred = cv2.GaussianBlur(octave_img, (5, 5), increase_sigma,
                                   borderType=cv2.BORDER_REPLICATE)
        pyramid.append(blurred)

        # down sample the current octave image to get the next octave image
        new_size = (int(octave_img.shape[1] / level_scale),
                    int(octave_img.shape[0] / level_scale))
        octave_img = cv2.resize(blurred, new_size, 0, 0,
                                interpolation=cv2.INTER_NEAREST)
        pre_sigma2 = cur_sigma2
        cur_sigma2 = cur_sigma2 * 2

    return pyramid


def matlab_like_desc_distance(distances_mat):
    """
    Computes the distance between teo set of descriptors.
    :return: A pair of numpy array:
    - A matrix where all the elements are HIGH_VALUE, but the one matched. The non HIGH_VALUE elements will
      have numbers from 1 to the number of matches where the lower numbers indicates most probable match.
    - A matrix containing the ratios between the first and the second match.
    :rtype (numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    HIGH_VALUE = 1000000
    # The shape of the output matrices
    out_shape = distances_mat.shape

    sorted_nn_mat = np.full(out_shape, HIGH_VALUE, dtype=float)
    ratio_dists = np.zeros(out_shape, dtype=float)
    tdesc_out = distances_mat.copy()

    # For each distance form the smallest to the higher, se in sorted_nn_mat a index [0, 1, 2, ...]
    # that indicates if is the smaller, the second smaller, thrid smaller, ...
    dnbr = 0
    while True:
        minj, mini = np.unravel_index(np.argmin(tdesc_out), tdesc_out.shape)
        min_dist = tdesc_out[minj, mini]

        if min_dist >= HIGH_VALUE:
            break

        # Set the row and the column with smaller distance to a HIGH_VALUE
        tdesc_out[:, mini] = HIGH_VALUE
        tdesc_out[minj] = HIGH_VALUE
        # Set the position where is the smaller distance to be the smallest one
        sorted_nn_mat[minj, mini] = dnbr
        dnbr += 1
        minratio = 1000000.0

        # Find the second element with smaller distance and the ratio between the first and second one
        for j in range(distances_mat.shape[0]):
            ratio = distances_mat[j, mini] / min_dist
            if 1 < ratio < minratio and j != minj:
                minratio = ratio

        ratio_dists[minj, mini] = minratio

    return sorted_nn_mat, ratio_dists


# TODO This distance is always between 0 and 1 :-(
def hellinger_dist(mean1, std1, mean2, std2):
    h2 = 1 - np.sqrt((2 * std1 * std2) / (ETH_EPS + std1 ** 2 + std2 ** 2)) * np.exp(
        -0.25 * (mean1 - mean2) ** 2 / (ETH_EPS + std1 ** 2 + std2 ** 2))
    return np.sqrt(h2)


def descriptors_hellinger_dist(a, b):
    a = a.reshape(a.shape[:-1] + (-1, 8))
    b = b.reshape(b.shape[:-1] + (-1, 8))
    means1, means2 = a[..., :4], b[..., :4]
    stds1, stds2 = a[..., 4:], b[..., 4:]
    total = hellinger_dist(means1, stds1, means2, stds2)
    # Sum along the last two axes, the number of bands and the gradient directions
    return total.sum(axis=(-2, -1))


def multiscale_helinger_dist(descriptorsL, descriptorsR):
    if len(descriptorsL) == 0 or len(descriptorsR) == 0:
        return np.array([])

    descriptorsL = list(map(lambda d: np.array(d), descriptorsL))
    ndims = descriptorsL[0].shape[1]

    maxR = np.max(list(map(lambda d: len(d), descriptorsR)))
    descriptorsR = np.array(list(map(
        lambda d: np.vstack([np.array(d), np.full((maxR - len(d), ndims), 0, np.float32)]), descriptorsR)))

    # Compute the L2 distance matrix and use it to find the matches
    D = np.zeros((len(descriptorsL), len(descriptorsR)), dtype=np.float32)
    # for r in tqdm(range(len(descriptorsL))):
    for r in range(len(descriptorsL)):
        D[r] = descriptors_hellinger_dist(descriptorsL[r], descriptorsR[:, :, np.newaxis]).min(axis=(1, 2))

    return D


def multiscale_descr_dist(descriptors_l, descriptors_r):
    # maxL = np.max(list(map(lambda d: len(d), descriptorsL)))
    max_r = np.max(list(map(lambda d: len(d), descriptors_r)))

    descriptors_l = list(map(lambda d: np.array(d), descriptors_l))
    descriptors_r = np.array(list(map(
        lambda d: np.vstack([np.array(d), np.full((max_r - len(d), 72), np.inf, np.float32)]), descriptors_r)))

    # Compute the L2 distance matrix and use it to find the matches
    D = np.zeros((len(descriptors_l), len(descriptors_r)), dtype=np.float32)
    for r in range(len(descriptors_l)):
        D[r] = np.linalg.norm(descriptors_l[r] - descriptors_r[:, :, np.newaxis], axis=-1).min(axis=(1, 2))

    return D
