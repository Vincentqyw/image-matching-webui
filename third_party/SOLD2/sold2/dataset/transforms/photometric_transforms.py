"""
Common photometric transforms for data augmentation.
"""
import numpy as np
from PIL import Image
from torchvision import transforms as transforms
import cv2


# List all the available augmentations
available_augmentations = [
    'additive_gaussian_noise',
    'additive_speckle_noise',
    'random_brightness',
    'random_contrast',
    'additive_shade',
    'motion_blur'
]


class additive_gaussian_noise(object):
    """ Additive gaussian noise. """
    def __init__(self, stddev_range=None):
        # If std is not given, use the default setting
        if stddev_range is None:
            self.stddev_range = [5, 95]
        else:
            self.stddev_range = stddev_range

    def __call__(self, input_image):
        # Get the noise stddev
        stddev = np.random.uniform(self.stddev_range[0], self.stddev_range[1])
        noise = np.random.normal(0., stddev, size=input_image.shape)
        noisy_image = (input_image + noise).clip(0., 255.)

        return noisy_image


class additive_speckle_noise(object):
    """ Additive speckle noise. """
    def __init__(self, prob_range=None):
        # If prob range is not given, use the default setting
        if prob_range is None:
            self.prob_range = [0.0, 0.005]
        else:
            self.prob_range = prob_range

    def __call__(self, input_image):
        # Sample
        prob = np.random.uniform(self.prob_range[0], self.prob_range[1])
        sample = np.random.uniform(0., 1., size=input_image.shape)

        # Get the mask
        mask0 = sample <= prob
        mask1 = sample >= (1 - prob)

        # Mask the image (here we assume the image ranges from 0~255
        noisy = input_image.copy()
        noisy[mask0] = 0.
        noisy[mask1] = 255.

        return noisy


class random_brightness(object):
    """ Brightness change. """
    def __init__(self, brightness=None):
        # If the brightness is not given, use the default setting
        if brightness is None:
            self.brightness = 0.5
        else:
            self.brightness = brightness

        # Initialize the transformer
        self.transform = transforms.ColorJitter(brightness=self.brightness)

    def __call__(self, input_image):
        # Convert to PIL image
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image.astype(np.uint8))

        return np.array(self.transform(input_image))


class random_contrast(object):
    """ Additive contrast. """
    def __init__(self, contrast=None):
        # If the brightness is not given, use the default setting
        if contrast is None:
            self.contrast = 0.5
        else:
            self.contrast = contrast

        # Initialize the transformer
        self.transform = transforms.ColorJitter(contrast=self.contrast)

    def __call__(self, input_image):
        # Convert to PIL image
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image.astype(np.uint8))

        return np.array(self.transform(input_image))


class additive_shade(object):
    """ Additive shade. """
    def __init__(self, nb_ellipses=20, transparency_range=None,
                   kernel_size_range=None):
        self.nb_ellipses = nb_ellipses
        if transparency_range is None:
            self.transparency_range = [-0.5, 0.8]
        else:
            self.transparency_range = transparency_range

        if kernel_size_range is None:
            self.kernel_size_range = [250, 350]
        else:
            self.kernel_size_range = kernel_size_range

    def __call__(self, input_image):
        # ToDo: if we should convert to numpy array first.
        min_dim = min(input_image.shape[:2]) / 4
        mask = np.zeros(input_image.shape[:2], np.uint8)
        for i in range(self.nb_ellipses):
            ax = int(max(np.random.rand() * min_dim, min_dim / 5))
            ay = int(max(np.random.rand() * min_dim, min_dim / 5))
            max_rad = max(ax, ay)
            x = np.random.randint(max_rad, input_image.shape[1] - max_rad)
            y = np.random.randint(max_rad, input_image.shape[0] - max_rad)
            angle = np.random.rand() * 90
            cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

        transparency = np.random.uniform(*self.transparency_range)
        kernel_size = np.random.randint(*self.kernel_size_range)

        # kernel_size has to be odd
        if (kernel_size % 2) == 0:
            kernel_size += 1
        mask = cv2.GaussianBlur(mask.astype(np.float32),
                                (kernel_size, kernel_size), 0)
        shaded = (input_image[..., None]
                  * (1 - transparency * mask[..., np.newaxis]/255.))
        shaded = np.clip(shaded, 0, 255)

        return np.reshape(shaded, input_image.shape)


class motion_blur(object):
    """ Motion blur. """
    def __init__(self, max_kernel_size=10):
        self.max_kernel_size = max_kernel_size

    def __call__(self, input_image):
        # Either vertical, horizontal or diagonal blur
        mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
        ksize = np.random.randint(
            0, int(round((self.max_kernel_size + 1) / 2))) * 2 + 1
        center = int((ksize - 1) / 2)
        kernel = np.zeros((ksize, ksize))
        if mode == 'h':
            kernel[center, :] = 1.
        elif mode == 'v':
            kernel[:, center] = 1.
        elif mode == 'diag_down':
            kernel = np.eye(ksize)
        elif mode == 'diag_up':
            kernel = np.flip(np.eye(ksize), 0)
        var = ksize * ksize / 16.
        grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
        gaussian = np.exp(-(np.square(grid - center)
                            + np.square(grid.T - center)) / (2. * var))
        kernel *= gaussian
        kernel /= np.sum(kernel)
        blurred = cv2.filter2D(input_image, -1, kernel)

        return np.reshape(blurred, input_image.shape)


class normalize_image(object):
    """ Image normalization to the range [0, 1]. """
    def __init__(self):
        self.normalize_value = 255

    def __call__(self, input_image):
        return (input_image / self.normalize_value).astype(np.float32)
