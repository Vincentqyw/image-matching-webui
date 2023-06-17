from tqdm import tqdm
from os import path as osp
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from src.datasets.megadepth import MegaDepthDataset
from src.datasets.scannet import ScanNetDataset
from src.datasets.aachen import AachenDataset
from src.datasets.inloc import InLocDataset


class TestDataLoader(DataLoader):
    """
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """

    def __init__(self, config):

        # 1. data config
        self.test_data_source = config.DATASET.TEST_DATA_SOURCE
        dataset_name = str(self.test_data_source).lower()
        # testing
        self.test_data_root = config.DATASET.TEST_DATA_ROOT
        self.test_pose_root = config.DATASET.TEST_POSE_ROOT  # (optional)
        self.test_npz_root = config.DATASET.TEST_NPZ_ROOT
        self.test_list_path = config.DATASET.TEST_LIST_PATH
        self.test_intrinsic_path = config.DATASET.TEST_INTRINSIC_PATH

        # 2. dataset config
        # general options
        self.min_overlap_score_test = config.DATASET.MIN_OVERLAP_SCORE_TEST  # 0.4, omit data with overlap_score < min_overlap_score

        # MegaDepth options
        if dataset_name == 'megadepth':
            self.mgdpt_img_resize = config.DATASET.MGDPT_IMG_RESIZE  # 800
            self.mgdpt_img_pad = True
            self.mgdpt_depth_pad = True
            self.mgdpt_df = 8
            self.coarse_scale = 0.125
        if dataset_name == 'scannet':
            self.img_resize = config.DATASET.TEST_IMGSIZE

        if (dataset_name == 'megadepth') or (dataset_name == 'scannet'):
            test_dataset = self._setup_dataset(
                self.test_data_root,
                self.test_npz_root,
                self.test_list_path,
                self.test_intrinsic_path,
                mode='test',
                min_overlap_score=self.min_overlap_score_test,
                pose_dir=self.test_pose_root)
        elif dataset_name == 'aachen_v1.1':
            test_dataset = AachenDataset(self.test_data_root, self.test_list_path,
                                         img_resize=config.DATASET.TEST_IMGSIZE)
        elif dataset_name == 'inloc':
            test_dataset = InLocDataset(self.test_data_root, self.test_list_path,
                                        img_resize=config.DATASET.TEST_IMGSIZE)
        else:
            raise "unknown dataset"

        self.test_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 4,
            'pin_memory': True
        }

        # sampler = Seq(self.test_dataset, shuffle=False)
        super(TestDataLoader, self).__init__(test_dataset, **self.test_loader_params)

    def _setup_dataset(self,
                       data_root,
                       split_npz_root,
                       scene_list_path,
                       intri_path,
                       mode='train',
                       min_overlap_score=0.,
                       pose_dir=None):
        """ Setup train / val / test set"""
        with open(scene_list_path, 'r') as f:
            npz_names = [name.split()[0] for name in f.readlines()]
        local_npz_names = npz_names

        return self._build_concat_dataset(data_root, local_npz_names, split_npz_root, intri_path,
                                          mode=mode, min_overlap_score=min_overlap_score, pose_dir=pose_dir)

    def _build_concat_dataset(
            self,
            data_root,
            npz_names,
            npz_dir,
            intrinsic_path,
            mode,
            min_overlap_score=0.,
            pose_dir=None
    ):
        datasets = []
        # augment_fn = self.augment_fn if mode == 'train' else None
        data_source = self.test_data_source
        if str(data_source).lower() == 'megadepth':
            npz_names = [f'{n}.npz' for n in npz_names]
        for npz_name in tqdm(npz_names):
            # `ScanNetDataset`/`MegaDepthDataset` load all data from npz_path when initialized, which might take time.
            npz_path = osp.join(npz_dir, npz_name)
            if data_source == 'ScanNet':
                datasets.append(
                    ScanNetDataset(data_root,
                                   npz_path,
                                   intrinsic_path,
                                   mode=mode, img_resize=self.img_resize,
                                   min_overlap_score=min_overlap_score,
                                   pose_dir=pose_dir))
            elif data_source == 'MegaDepth':
                datasets.append(
                    MegaDepthDataset(data_root,
                                     npz_path,
                                     mode=mode,
                                     min_overlap_score=min_overlap_score,
                                     img_resize=self.mgdpt_img_resize,
                                     df=self.mgdpt_df,
                                     img_padding=self.mgdpt_img_pad,
                                     depth_padding=self.mgdpt_depth_pad,
                                     coarse_scale=self.coarse_scale))
            else:
                raise NotImplementedError()
        return ConcatDataset(datasets)
