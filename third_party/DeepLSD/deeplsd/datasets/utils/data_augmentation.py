"""
Common photometric transforms for data augmentation.
"""
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms as transforms


available_augmentations = [
    'additive_gaussian_noise',
    'additive_speckle_noise',
    'random_brightness',
    'random_contrast',
    'additive_shade',
    'motion_blur'
]


class additive_gaussian_noise(object):
    def __init__(self, stddev_range=[5, 95]):
        self.stddev_range = stddev_range

    def __call__(self, input_image):
        # Get the noise stddev
        stddev = np.random.uniform(self.stddev_range[0], self.stddev_range[1])
        noise = np.random.normal(0., stddev, size=input_image.shape)
        noisy_image = (input_image + noise).clip(0., 255.)

        return noisy_image


class additive_speckle_noise(object):
    def __init__(self, prob_range=[0.0, 0.01]):
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
    def __init__(self, brightness=0.5):
        self.brightness = brightness

        # Initialize the transformer
        self.transform = transforms.ColorJitter(brightness=self.brightness)

    def __call__(self, input_image):
        # Convert to PIL image
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image.astype(np.uint8))

        return np.array(self.transform(input_image))


class random_contrast(object):
    def __init__(self, strength_range=[0.5, 1.5]):
        self.strength_range = strength_range

    def __call__(self, input_image):
        strength = np.random.uniform(self.strength_range[0],
                                     self.strength_range[1])
        contrasted_img = input_image.copy()
        mean = np.mean(contrasted_img)
        contrasted_img = (contrasted_img - mean) * strength + mean

        return contrasted_img.clip(0, 255)


class additive_shade(object):
    def __init__(self, nb_ellipses=20, transparency_range=[-0.8, 0.8],
                   kernel_size_range=[100, 150]):
        self.nb_ellipses = nb_ellipses
        self.transparency_range = transparency_range
        self.kernel_size_range = kernel_size_range

    def __call__(self, input_image):
        min_dim = min(input_image.shape[:2]) / 4
        mask = np.zeros(input_image.shape[:2], np.uint8)
        for _ in range(self.nb_ellipses):
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
        if len(input_image.shape) == 2:
            shaded = input_image[:, :, None] * (1 - transparency
                                                * mask[..., np.newaxis] / 255.)
        else:
            shaded = input_image * (1 - transparency
                                    * mask[..., np.newaxis] / 255.)
        shaded = np.clip(shaded, 0, 255)

        return np.reshape(shaded, input_image.shape)


class motion_blur(object):
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
        gaussian = np.exp(-(np.square(grid - center) + np.square(grid.T - center)) / (2. * var))
        kernel *= gaussian
        kernel /= np.sum(kernel)
        blurred = cv2.filter2D(input_image, -1, kernel)

        return np.reshape(blurred, input_image.shape)


def photometric_augmentation(input_img, config):
    """ Process the input image through multiple transforms. """
    if 'primitives' in config:
        transforms = config['primitives']
    else:
        transforms = available_augmentations
    
    # Take a random subset of transforms
    n_transforms = len(transforms)
    n_used = np.random.randint(n_transforms + 1)
    transforms = np.random.choice(transforms, n_used, replace=False)

    # Apply the transforms
    transformed_img = input_img.copy()
    for primitive in transforms:
        transform = globals()[primitive](**config['params'][primitive])
        transformed_img = transform(transformed_img)
    
    return transformed_img