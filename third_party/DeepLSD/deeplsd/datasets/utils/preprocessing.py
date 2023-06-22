import cv2
import numpy as np
import csv


def numpy_image_to_torch(image):
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f'Not an image: {image.shape}')
    return (image / 255.).astype(np.float32, copy=False)


def resize(image, size, fn=None, nearest=False):
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h*scale)), int(round(w*scale))
        scale = (scale, scale)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f'Incorrect new size: {size}')
    mode = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def resize_and_crop(image, size, interp_mode=None):
    """ Apply a central crop to an image to resize it to a fixed size. """
    source_size = np.array(image.shape[:2], dtype=float)
    target_size = np.array(size, dtype=float)

    # Scale
    scale = np.amax(target_size / source_size)
    inter_size = np.round(source_size * scale).astype(int)
    if interp_mode is None:
        interp_mode = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    image = cv2.resize(image, (inter_size[1], inter_size[0]),
                       interpolation=interp_mode)

    # Central crop
    pad = np.round((source_size * scale - target_size) / 2.).astype(int)
    image = image[pad[0]:(pad[0] + int(target_size[0])),
                  pad[1]:(pad[1] + int(target_size[1]))]
    
    return image


def crop(image, size, random=True, other=None, K=None):
    """Random or deterministic crop of an image, adjust depth and intrinsics.
    """
    h, w = image.shape[:2]
    h_new, w_new = (size, size) if isinstance(size, int) else size
    top = np.random.randint(0, h - h_new + 1) if random else 0
    left = np.random.randint(0, w - w_new + 1) if random else 0
    image = image[top:top+h_new, left:left+w_new]
    ret = [image]
    if other is not None:
        ret += [other[top:top+h_new, left:left+w_new]]
    if K is not None:
        K[0, 2] -= left
        K[1, 2] -= top
        ret += [K]
    return ret


def read_timestamps(text_file):
    """
    Read a text file containing the timestamps of images
    and return a dictionary matching the name of the image
    to its timestamp.
    """
    timestamps = {'name': [], 'date': [], 'hour': [],
                  'minute': [], 'time': []}
    with open(text_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            timestamps['name'].append(row[0])
            timestamps['date'].append(row[1])
            hour = int(row[2])
            timestamps['hour'].append(hour)
            minute = int(row[3])
            timestamps['minute'].append(minute)
            timestamps['time'].append(hour + minute / 60.)
    return timestamps
