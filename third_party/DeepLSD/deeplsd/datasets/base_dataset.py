"""
Base class for dataset.
See mnist.py for an example of dataset.
"""

from abc import ABCMeta, abstractmethod
from omegaconf import OmegaConf
import omegaconf
from torch.utils.data import DataLoader, Sampler, get_worker_info
import logging

from ..utils.tools import set_num_threads


class LoopSampler(Sampler):
    def __init__(self, loop_size, total_size=None):
        self.loop_size = loop_size
        self.total_size = total_size - (total_size % loop_size)

    def __iter__(self):
        return (i % self.loop_size for i in range(self.total_size))

    def __len__(self):
        return self.total_size


def worker_init_fn(i):
    info = get_worker_info()
    if hasattr(info.dataset, 'conf'):
        set_num_threads(info.dataset.conf.num_threads)
    else:
        set_num_threads(1)


class BaseDataset(metaclass=ABCMeta):
    """
    What the dataset model is expect to declare:
        default_conf: dictionary of the default configuration of the dataset.
        It overwrites base_default_conf in BaseModel, and it is overwritten by
        the user-provided configuration passed to __init__.
        Configurations can be nested.

        _init(self, conf): initialization method, where conf is the final
        configuration object (also accessible with `self.conf`). Accessing
        unkown configuration entries will raise an error.

        get_dataset(self, split): method that returns an instance of
        torch.utils.data.Dataset corresponding to the requested split string,
        which can be `'train'`, `'val'`, `'test'`, or `'export'`.
    """
    base_default_conf = {
        'name': '???',
        'num_workers': 1,
        'train_batch_size': '???',
        'val_batch_size': '???',
        'test_batch_size': '???',
        'export_batch_size': '???',
        'batch_size': 1,
        'num_threads': 1,
    }
    default_conf = {}

    def __init__(self, conf):
        """Perform some logic and call the _init method of the child model."""
        default_conf = OmegaConf.merge(
                OmegaConf.create(self.base_default_conf),
                OmegaConf.create(self.default_conf))
        OmegaConf.set_struct(default_conf, True)
        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = OmegaConf.merge(default_conf, conf)
        
        # Update the batch sizes if necessary
        for split in ['train', 'val', 'test', 'export']:
            if OmegaConf.is_missing(self.conf, split + '_batch_size'):
                OmegaConf.update(self.conf, split + '_batch_size',
                                 self.conf.batch_size, merge=False)
        
        OmegaConf.set_readonly(self.conf, True)
        logging.info(f'Creating dataset {self.__class__.__name__}')
        self._init(self.conf)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def get_dataset(self, split):
        """To be implemented by the child class."""
        raise NotImplementedError

    def get_data_loader(self, split, shuffle=False):
        """Return a data loader for a given split."""
        assert split in ['train', 'val', 'test', 'export']
        batch_size = self.conf.get(split+'_batch_size')
        num_workers = self.conf.get('num_workers', batch_size)
        return DataLoader(self.get_dataset(split), batch_size=batch_size,
                          shuffle=shuffle or split == 'train',
                          pin_memory=True, num_workers=num_workers,
                          worker_init_fn=worker_init_fn)

    def get_overfit_loader(self, split):
        """Return an overfit data loader.
        The training set is composed of a single duplicated batch, while
        the validation and test sets contain a single copy of this same batch.
        This is useful to debug a model and make sure that losses and metrics
        correlate well.
        """
        assert split in ['train', 'val', 'test', 'export']
        dataset = self.get_dataset('train')
        sampler = LoopSampler(
            self.conf.batch_size,
            len(dataset) if split == 'train' else self.conf.batch_size)
        num_workers = self.conf.get('num_workers', self.conf.batch_size)
        return DataLoader(dataset, batch_size=self.conf.batch_size,
                          pin_memory=True, num_workers=num_workers,
                          sampler=sampler, worker_init_fn=worker_init_fn)
