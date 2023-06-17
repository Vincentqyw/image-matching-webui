import torch, imageio, h5py, json, typing
import numpy as np
import os.path as P
from torch_dimcheck import dimchecked

from disk import NpArray, Image, DataError
from disk.data.tuple_dataset import TupleDataset
from disk.data.limited_dataset import LimitedConcatDataset

'''
The datasets are all loaded based on a json file which specifies which tuples
(pairs, triplets, ...) of images are covisible. The structure of the dataset
is as follows:

{
    scene_name_1: {
        image_path: path_to_directory_with_images_for_scene_1,
        depth_path: path_to_directory_with_depths_for_scene_1,
        calib_path: path_to_directory_with_calibs_for_scene_1,
        images: [img_name_1, img_name_2, ...],
        tuples: [[id1_1, id1_2, id1_3], [id2_1, id2_2, id2_3], ...]
    }
}

where 
    * `path_to_directory_with_*_for_scene_*` can be absolute or relative to
      the location of the json file itself.

    * `depth_path` may be missing if you always use DISKDataset with
      no_depth=True

    * `images` lists the file names *with* their extension

    * `tuples` specifies co-visible tuples by their IDs in `images`, that is
      the first tuple above consists of
      [images[id1_1], images[id1_2], images[id1_3]]
      and all tuples are of equal length.
'''

def _base_image_name(img_name):
    return img_name.split('.')[0]

def _crop(image, crop_size):
    return image.scale(crop_size).pad(crop_size)
 
@dimchecked
def _read_bitmap(bitmap_path) -> [3, 'h', 'w']:
    bitmap = imageio.imread(bitmap_path)
    bitmap = bitmap.astype(np.float32) / 255

    return torch.from_numpy(bitmap).permute(2, 0, 1)

@dimchecked
def _read_depth(depth_path) -> [1, 'h', 'w']:
    h5 = h5py.File(depth_path, 'r')
    depth = h5['depth'][:].astype(np.float32)
    depth[depth == 0.] = float('NaN')

    return torch.from_numpy(depth).unsqueeze(0)

class ImageSet:
    '''
    This class represents a directory (single scene) of images with known poses
    and (potentially) depth maps. It DOES NOT know anything about
    covisible-pairs and is meant to be used as a "resource" by DISKDataset
    below.
    '''
    def __init__(self, json_data, crop_size, root, no_depth=False):
        def maybe_add_root(path):
            '''
            We try if the path pointed to by the dataset json file exists and
            and if not, we prepend the dataset location (thus making it
            relative). If this fails, we error out
            '''
            if P.exists(path) and P.isdir(path):
                return path
            rooted_path = P.join(root, path)
            if (not P.exists(rooted_path)) or (not P.isdir(rooted_path)):
                raise DataError(f"Couldn't find a directory at {path} nor "
                                f"{rooted_path}")

            return rooted_path
                
        self.image_path = maybe_add_root(json_data['image_path'])
        self.calib_path = maybe_add_root(json_data['calib_path'])
        if no_depth:
            self.depth_path = None
        else:
            self.depth_path = maybe_add_root(json_data['depth_path'])

        self.id2name    = json_data['images']
        self.crop_size  = crop_size

    def _get_depth(self, image_name):
        if self.depth_path is None:
            return None

        h5_name = _base_image_name(image_name) + '.h5'
        depth_path = P.join(self.depth_path, h5_name)
        return _read_depth(depth_path)

    def _get_bitmap(self, image_name):
        return _read_bitmap(self._get_bitmap_path(image_name))

    def _get_bitmap_path(self, image_name):
        base_name = _base_image_name(image_name)
        return P.join(self.image_path, base_name + '.jpg')

    def _get_KRT(self, image_name):
        calibration_path = P.join(self.calib_path,
            f'calibration_{image_name}.h5'
        )

        values = []
        with h5py.File(calibration_path, 'r') as calib_file:
            for f in ['K', 'R', 'T']:
                v = torch.from_numpy(calib_file[f][()]).to(torch.float32)
                values.append(v)

        return values
 
    def __getitem__(self, id):
        image_name = self.id2name[id]

        image = Image(
            *self._get_KRT(image_name),
            self._get_bitmap(image_name),
            self._get_depth(image_name),
            self._get_bitmap_path(image_name)
        )

        return _crop(image, self.crop_size)

class SceneTuples(TupleDataset):
    '''
    This class knows about the tuples (pairs, triplets, etc) of covisible
    images in the scene. It holds an ImageSet and subclasses TupleDataset
    to implement the tuple selection logic.
    '''
    def __init__(self, json_data, crop_size, root, no_depth=False):
        items  = ImageSet(json_data, crop_size, root, no_depth=no_depth)
        tuples = json_data['tuples']

        super(SceneTuples, self).__init__(items, tuples)

class DISKDataset(LimitedConcatDataset):
    '''
    This class holds a number of SceneTuples instances (one per scene in the
    dataset) and acts as their concatenation.
    '''
    def __init__(
        self, json_path, crop_size=(768, 768),
        no_depth=False, limit=None, shuffle=False, warn=True,
    ):
        self.crop_size = crop_size

        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)

        root_path, _ = P.split(json_path)
        scene_datasets = []        
        for scene in json_data:
            scene_datasets.append(SceneTuples(
                json_data[scene],
                crop_size,
                root_path,
                no_depth=no_depth
            ))

        super(DISKDataset, self).__init__(
            scene_datasets,
            limit=limit,
            shuffle=shuffle,
            warn=warn,
        )
         
    @staticmethod
    def collate_fn(batch):
        bitmaps= []
        images = [] 

        for tuple_ in batch:
            images.append(np.array(tuple_))
            bitmaps.append(torch.stack([image.bitmap for image in tuple_]))

        return PinnableTupleBatch(torch.stack(bitmaps), np.stack(images))

class PinnableTupleBatch(typing.NamedTuple):
    '''
    This class allows for easier manipulation of batches coming from using
    DISKDataset along with torch.utils.data.DataLoader and, in particular,
    using the `batch.to(device='cuda', non_blocking=True)` method
    '''
    bitmaps: torch.Tensor
    images : NpArray[Image]

    def pin_memory(self):
        bitmaps = self.bitmaps.pin_memory()
        images  = [im.pin_memory() for im in self.images]

        return PinnableTupleBatch(bitmaps, images)

    def to(self, *args, **kwargs):
        bitmaps = self.bitmaps.to(*args, **kwargs)
        images  = self.images.copy()
        for i in range(images.size):
            images.flat[i] = images.flat[i].to(*args, **kwargs)

        return PinnableTupleBatch(bitmaps, images)
