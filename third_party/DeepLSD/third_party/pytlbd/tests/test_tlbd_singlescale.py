import numpy as np
import cv2
import pytlbd
from scipy.spatial.distance import cdist


def draw_multiscale_matches(img_left, img_right, segs_left, segs_right, matches):
    assert img_left.ndim == 2
    h, w = img_left.shape

    # store the matching results of the first and second images into a single image
    left_color_img = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
    right_color_img = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)
    r1, g1, b1 = [], [], []  # the line colors

    # Draw the matches lines
    for pair in range(len(matches)):
        r1.append(int(255 * np.random.rand()))
        g1.append(int(255 * np.random.rand()))
        b1.append(255 - r1[-1])
        line_id_l, line_id_r = matches[pair]

        l = segs_left[line_id_l]
        cv2.line(left_color_img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (r1[pair], g1[pair], b1[pair]), 3)

        l = segs_right[line_id_r]
        cv2.line(right_color_img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (r1[pair], g1[pair], b1[pair]), 3)

    # Connect the matched lines with a semi-transparent line
    result_img = np.hstack([left_color_img, right_color_img])
    result_img_tmp = result_img.copy()
    for pair in range(27, len(matches)):
        line_id_l, line_id_r = matches[pair]
        seg_left = segs_left[line_id_l]
        seg_right = segs_right[line_id_r]
        start_ptn = (int(seg_left[0]), int(seg_left[1]))
        end_ptn = (int(seg_right[0] + w), int(seg_right[1]))
        cv2.line(result_img_tmp, start_ptn, end_ptn, (r1[pair], g1[pair], b1[pair]), 2, cv2.LINE_AA)

    result_img = cv2.addWeighted(result_img, 0.5, result_img_tmp, 0.5, 0.0)
    return result_img


# read both images
imgLeft = cv2.imread('resources/leuven1.jpg', cv2.IMREAD_GRAYSCALE)
imgRight = cv2.imread('resources/leuven2.jpg', cv2.IMREAD_GRAYSCALE)

# Detect segments
segmentsLeft = pytlbd.edlines_single_scale(imgLeft)
segmentsRight = pytlbd.edlines_single_scale(imgRight)

# Compute single scale descriptors
descriptorsL = pytlbd.lbd_single_scale(imgLeft, segmentsLeft, 9, 7)
descriptorsR = pytlbd.lbd_single_scale(imgRight, segmentsRight, 9, 7)

# Compute the L2 distance matrix and use it to find the matches
D = cdist(descriptorsL, descriptorsR)
matches = np.array([np.arange(D.shape[0]), D.argmin(1)])
matches = matches.T[np.argsort(D.min(1))]
print(f"--> Number of total matches = {len(matches)}")

# Draw the obtained matches
matchesImage = draw_multiscale_matches(imgLeft, imgRight, segmentsLeft, segmentsRight, matches)
cv2.imshow("Matches found", matchesImage)
cv2.waitKey()
