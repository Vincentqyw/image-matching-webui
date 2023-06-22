# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytlsd
from skimage.transform import pyramid_reduce

NOTDEF = -1024.0


def get_thresholded_grad(resized_img):
    modgrad = np.full(resized_img.shape, NOTDEF, np.float64)
    anglegrad = np.full(resized_img.shape, NOTDEF, np.float64)

    # A B
    # C D
    A, B, C, D = resized_img[:-1, :-1], resized_img[:-1, 1:], resized_img[1:, :-1], resized_img[1:, 1:]
    gx = B + D - (A + C)  # horizontal difference
    gy = C + D - (A + B)  # vertical difference

    threshold = 5.2262518595055063
    modgrad[:-1, :-1] = 0.5 * np.sqrt(gx ** 2 + gy ** 2)
    anglegrad[:-1, :-1] = np.arctan2(gx, -gy)
    anglegrad[modgrad <= threshold] = NOTDEF
    return gx, gy, modgrad, anglegrad


gray = cv2.imread('resources/ai_001_001.frame.0000.color.jpg', cv2.IMREAD_GRAYSCALE)
flt_img = gray.astype(np.float64)

scale_down = 0.8
resized_img = pyramid_reduce(flt_img, 1 / scale_down, 0.6)

# Get image gradients
gx, gy, gradnorm, gradangle = get_thresholded_grad(resized_img)

segments = pytlsd.lsd(resized_img, 1.0, gradnorm=gradnorm, gradangle=gradangle)
segments /= scale_down

plt.title("Gradient norm")
plt.imshow(gradnorm[:-1, :-1])
plt.colorbar()
plt.figure()
gradangle[gradangle == NOTDEF] = -5
plt.title("Thresholded gradient angle")
plt.imshow(gradangle[:-1, :-1])
plt.colorbar()


img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
for segment in segments:
    cv2.line(img_color, (int(segment[0]), int(segment[1])), (int(segment[2]), int(segment[3])), (0, 255, 0))

plt.figure()
plt.title(f"Detected segments N {len(segments)}")
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.show()
