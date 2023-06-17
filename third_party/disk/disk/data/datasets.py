import itertools, os
from torch.utils.data import DataLoader

from disk.data import DISKDataset

def get_datasets(
    root,
    no_depth=None,
    batch_size=2,
    crop_size=(768, 768),
    substep=1,
    n_epochs=50,
    chunk_size=5000,
    train_limit=1000,
    test_limit=250,
):
    if no_depth is None:
        raise ValueError("Unspecified no_depth")

    train_dataset = DISKDataset(
        os.path.join(root, 'train/dataset.json'),
        crop_size=crop_size,
        limit=train_limit,
        shuffle=True,
        no_depth=no_depth,
    )
    dataloader_kwargs = {
        'collate_fn': train_dataset.collate_fn,
        'pin_memory': True,
        'num_workers': 4,
    }
    train_dataloader = DataLoader(
        train_dataset, shuffle=True,
        batch_size=batch_size, **dataloader_kwargs
    )

    test_dataset = DISKDataset(
        os.path.join(root, 'test/dataset.json'),
        crop_size=crop_size,
        limit=test_limit,
        shuffle=True,
        no_depth=no_depth,
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False,
        batch_size=batch_size, **dataloader_kwargs
    )


    if len(train_dataloader) < chunk_size:
        raise ValueError(f'Your training dataset has {len(train_dataloader)} '
                         f'items, which is less than your --chunk-size setting '
                         f'({chunk_size}) therefore no chunks could be '
                          'created. Please reduce --chunk-size or use a bigger'
                          ' dataset.')

    train_chunk_iter = itertools.islice(DividedIter(
        train_dataloader,
        n_repeats=n_epochs*10,
        chunk_size=chunk_size,
        reinit=lambda dataloader: dataloader.dataset.shuffle(),
    ), n_epochs)

    return train_chunk_iter, test_dataloader

class DividedIter:
    def __init__(self, iterable, n_repeats=1, n_chunks=None,
                 chunk_size=None, reinit=None):

        if (n_chunks is None) == (chunk_size is None):
            raise ValueError(
                'Exactly one of `n_chunks` and `chunk_size` has to be None'
            )

        self._iterable    = iterable
        self._base_length = len(iterable)

        if chunk_size is None:
            chunk_size = self._base_length // n_chunks
        if n_chunks is None:
            n_chunks   = self._base_length // chunk_size

        self.n_repeats  = n_repeats
        self.n_chunks   = n_chunks
        self.chunk_size = chunk_size
        self.reinit     = reinit

        self.total_chunks = self.n_chunks * self.n_repeats

    def __len__(self):
        return self.total_chunks

    def __iter__(self):
        for _ in range(self.n_repeats):
            if self.reinit is not None:
                self.reinit(self._iterable)

            base_iter = iter(self._iterable)

            for _ in range(self.n_chunks):
                yield itertools.islice(base_iter, self.chunk_size)
