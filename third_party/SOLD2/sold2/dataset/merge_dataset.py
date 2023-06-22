""" Compose multiple datasets in a single loader. """

import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset

from .wireframe_dataset import WireframeDataset
from .holicity_dataset import HolicityDataset


class MergeDataset(Dataset):
    def __init__(self, mode, config=None):
        super(MergeDataset, self).__init__()
        # Initialize the datasets
        self._datasets = []
        spec_config = deepcopy(config)
        for i, d in enumerate(config['datasets']):
            spec_config['dataset_name'] = d
            spec_config['gt_source_train'] = config['gt_source_train'][i]
            spec_config['gt_source_test'] = config['gt_source_test'][i]
            if d == "wireframe":
                self._datasets.append(WireframeDataset(mode, spec_config))
            elif d == "holicity":
                spec_config['train_split'] = config['train_splits'][i]
                self._datasets.append(HolicityDataset(mode, spec_config))
            else:
                raise ValueError("Unknown dataset: " + d)            

        self._weights = config['weights']
    
    def __getitem__(self, item):
        dataset = self._datasets[np.random.choice(
            range(len(self._datasets)), p=self._weights)]
        return dataset[np.random.randint(len(dataset))]

    def __len__(self):
        return np.sum([len(d) for d in self._datasets])
