import os, argparse, h5py, warnings
import numpy as np
from tqdm import tqdm
from PIL import Image, ExifTags

from colmap.database import COLMAPDatabase, image_ids_to_pair_id

def get_focal(image_path, err_on_default=False):
    image         = Image.open(image_path)
    max_size      = max(image.size)

    exif = image.getexif()
    focal = None
    if exif is not None:
        focal_35mm = None
        # https://github.com/colmap/colmap/blob/d3a29e203ab69e91eda938d6e56e1c7339d62a99/src/util/bitmap.cc#L299
        for tag, value in exif.items():
            focal_35mm = None
            if ExifTags.TAGS.get(tag, None) == 'FocalLengthIn35mmFilm':
                focal_35mm = float(value)
                break

        if focal_35mm is not None:
            focal = focal_35mm / 35. * max_size
    
    if focal is None:
        if err_on_default:
            raise RuntimeError("Failed to find focal length")

        # failed to find it in exif, use prior
        FOCAL_PRIOR = 1.2
        focal = FOCAL_PRIOR * max_size

    return focal

def create_camera(db, image_path):
    image         = Image.open(image_path)
    width, height = image.size

    focal = get_focal(image_path)

    if args.camera_model == 'simple-pinhole':
        model = 0 # simple pinhole
        param_arr = np.array([focal, width / 2, height / 2])
    if args.camera_model == 'pinhole':
        model = 1 # pinhole
        param_arr = np.array([focal, focal, width / 2, height / 2])
    elif args.camera_model == 'simple-radial':
        model = 2 # simple radial
        param_arr = np.array([focal, width / 2, height / 2, 0.1])
    elif args.camera_model == 'opencv':
        model = 4 # opencv
        param_arr = np.array([focal, focal, width / 2, height / 2, 0., 0., 0., 0.])
         
    return db.add_camera(model, width, height, param_arr)


def add_keypoints(db, h5_path, image_path):
    keypoint_f = h5py.File(os.path.join(h5_path, 'keypoints.h5'), 'r')

    camera_id = None
    fname_to_id = {}
    for filename in tqdm(list(keypoint_f.keys())):
        keypoints = keypoint_f[filename][()]

        fname_with_ext = filename + args.image_extension
        path = os.path.join(image_path, fname_with_ext)
        if not os.path.isfile(path):
            raise IOError(f'Invalid image path {path}')

        if camera_id is None or not args.single_camera:
            camera_id = create_camera(db, path)
        image_id = db.add_image(fname_with_ext, camera_id)
        fname_to_id[filename] = image_id

        db.add_keypoints(image_id, keypoints)

    return fname_to_id

def add_matches(db, h5_path, fname_to_id):
    match_file = h5py.File(os.path.join(h5_path, 'matches.h5'), 'r')
    
    added = set()
    n_keys = len(match_file.keys())
    n_total = (n_keys * (n_keys - 1)) // 2

    with tqdm(total=n_total) as pbar:
        for key_1 in match_file.keys():
            group = match_file[key_1]
            for key_2 in group.keys():
                id_1 = fname_to_id[key_1]
                id_2 = fname_to_id[key_2]

                pair_id = image_ids_to_pair_id(id_1, id_2)
                if pair_id in added:
                    warnings.warn(f'Pair {pair_id} ({id_1}, {id_2}) already added!')
                    continue
            
                matches = group[key_2][()]
                db.add_matches(id_1, id_2, matches.T)

                added.add(pair_id)

                pbar.update(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('h5_path', help=('Path to the directory with '
                                         'keypoints.h5 and matches.h5'))
    parser.add_argument('image_path', help='Path to source images')
    parser.add_argument(
        '--image-extension', default='.jpg', type=str,
        help='Extension of files in image_path'
    )
    parser.add_argument('--database-path', default='database.db',
        help='Location where the COLMAP .db file will be created'
    )
    parser.add_argument(
        '--single-camera', action='store_true',
        help=('Consider all photos to be made with a single camera (COLMAP '
              'will reduce the number of degrees of freedom'),
    )
    parser.add_argument(
        '--camera-model',
        choices=['simple-pinhole', 'pinhole', 'simple-radial', 'opencv'],
        default='simple-radial',
        help=('Camera model to use in COLMAP. '
              'See https://github.com/colmap/colmap/blob/master/src/base/camera_models.h'
              ' for explanations')
    )

    args = parser.parse_args()

    if args.camera_model == 'opencv' and not args.single_camera:
        raise RuntimeError("Cannot use --camera-model=opencv camera without "
                           "--single-camera (the COLMAP optimisation will "
                           "likely fail to converge)")

    if os.path.exists(args.database_path):
        raise RuntimeError("database path already exists - will not modify it.")

    db = COLMAPDatabase.connect(args.database_path)
    db.create_tables()

    fname_to_id = add_keypoints(db, args.h5_path, args.image_path)
    add_matches(
        db,
        args.h5_path,
        fname_to_id,
    )

    db.commit()
