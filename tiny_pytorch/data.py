import itertools
import math
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Callable, Iterable, Iterator

import numpy as np

from .tensor import Tensor


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: list | None = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError(
            "Subclasses of Dataset should implement __getitem__"
        )

    def apply_transforms(self, x):
        if self.transforms is not None:
            for tform in self.transforms:
                x = tform(x)
        return x


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])


class Sampler:
    def __init__(self, ds: Dataset, shuffle: bool = False):
        self.n = len(ds)
        self.shuffle = shuffle

    def __iter__(self):
        res = list(range(self.n))
        if self.shuffle:
            random.shuffle(res)
        return iter(res)


class BatchSampler:
    def __init__(
        self,
        sampler: Sampler | Iterable[int],
        batch_size: int,
        drop_last: bool = False,
    ):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        yield from BatchSampler.chunked(
            iter(self.sampler), self.batch_size, drop_last=self.drop_last
        )

    @staticmethod
    def chunked(it, chunk_sz=None, drop_last=False):
        if not isinstance(it, Iterator):
            it = iter(it)
        while True:
            res = list(itertools.islice(it, chunk_sz))
            if res and (len(res) == chunk_sz or not drop_last):
                yield res
            if len(res) < chunk_sz:
                return


# FIXME: Need to figure out a way to stack xb and yb
# Currently it returns a tuple of tensors for xb and yb
def collate(idxs, ds):
    xb, yb = zip(*[ds[i] for i in idxs])
    return xb, yb


class DataLoader:
    """
    A data loader that combines a dataset and a sampler to provide iterable batches of data.

    The DataLoader provides an efficient way to iterate over a dataset in batches,
    with options for shuffling, parallel data loading, and custom collation.

    Args:
        dataset (Dataset): The dataset to load data from
        batch_size (int, optional): Number of samples per batch. Default: 1
        n_workers (int, optional): Number of worker processes for parallel loading. Default: 1
        shuffle (bool, optional): Whether to shuffle the data at every epoch. Default: False
        drop_last (bool, optional): Whether to drop the last incomplete batch. Default: False
        collate_fn (Callable, optional): Function to collate samples into batches. Default: collate

    Example:
        >>> dataset = MyDataset(...)
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch_x, batch_y in loader:
        ...     # Process batch
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        n_workers: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        collate_fn: Callable[
            [Iterable[int], Dataset], tuple[Tensor, Tensor]
        ] = collate,
    ):

        self.dataset = dataset
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.shuffle = shuffle
        self.batch_sampler = BatchSampler(
            Sampler(dataset, shuffle), batch_size, drop_last
        )
        self.collate_fn = collate_fn

    def __len__(self):
        return (len(self.dataset) - 1) // self.batch_size + 1

    def __iter__(self):
        with ProcessPoolExecutor(self.n_workers) as ex:
            yield from ex.map(
                partial(self.collate_fn, ds=self.dataset), self.batch_sampler
            )


class Transform:
    def __call__(self, x):
        raise NotImplementedError()


class RandomFlipHorizontal(Transform):
    """Horizonally flip an image, specified as n H x W x C NDArray."""

    def __init__(self, p: int = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            return np.flip(img, 1)
        return img


class RandomCrop(Transform):
    """Zero pad and then randomly crop an image."""

    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        if len(img.shape) < 3:
            size = int(math.sqrt(img.shape[0]))
            img = img.reshape(size, size, 1)
        h, w, _ = img.shape
        top_left = self.padding + shift_x
        bottom_left = self.padding + shift_y
        padded_img = np.pad(
            img,
            pad_width=(
                (self.padding, self.padding),
                (self.padding, self.padding),
                (0, 0),
            ),
        )
        return padded_img[
            top_left : top_left + h, bottom_left : bottom_left + w, :
        ]
