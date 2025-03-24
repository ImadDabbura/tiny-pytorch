"""Data loading and processing utilities.

This module provides utilities for loading and processing data in the tiny-pytorch
framework. It includes dataset abstractions, data loading functionality, and
transform operations similar to PyTorch's data utilities.

The module provides base classes for datasets and transforms, as well as concrete
implementations for specific data types and transformations.

Classes
-------
Dataset
    Base class that provides common dataset functionality.
NDArrayDataset
    Dataset implementation for numpy arrays.
DataLoader
    Iterates over a dataset in batches with optional multiprocessing.
Transform
    Base class for all data transformations.
RandomCrop
    Randomly crops data to specified size.
RandomFlipHorizontal
    Randomly flips data horizontally with given probability.
"""

import itertools
import math
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Callable, Iterable, Iterator

import numpy as np

from .tensor import Tensor


class Dataset:
    """Base class for all datasets.

    This class defines the basic interface and functionality that all dataset
    implementations should follow. It provides common methods like __getitem__
    for accessing data samples and apply_transforms for data augmentation.

    Notes
    -----
    All datasets should inherit from this base class and implement the __getitem__
    method according to their specific data loading requirements.
    """

    def __init__(self, transforms: list | None = None):
        """
        Parameters
        ----------
        transforms : list or None, optional
            List of transform functions to be applied to data samples. Each transform
            should be a callable that takes a sample and returns the transformed sample.
            Default is None.

        Notes
        -----
        The transforms will be applied sequentially in the order they appear in the list
        when apply_transforms() is called on a sample.
        """
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
    """Dataset for working with NDArrays.

    A dataset class that wraps NDArrays for use in machine learning tasks.
    Supports multiple arrays that will be returned as tuples when indexed.
    Commonly used for features and labels in supervised learning.

    Notes
    -----
    All arrays must have the same first dimension (length).
    Arrays will be returned in the same order they were passed to __init__.
    """

    def __init__(self, *arrays):
        """
        Parameters
        ----------
        *arrays : array_like
            One or more arrays to include in the dataset. All arrays must have the
            same first dimension (length).

        Raises
        ------
        ValueError
            If no arrays are provided or if arrays have different lengths.

        Notes
        -----
        Arrays will be returned in the same order they were passed when indexing
        the dataset.

        Examples
        --------
        >>> import numpy as np
        >>> X = np.random.randn(100, 10)  # 100 samples, 10 features
        >>> y = np.random.randint(0, 2, 100)  # Binary labels
        >>> dataset = NDArrayDataset(X, y)
        >>> x, y = dataset[0]  # Get first sample and label
        """
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])


class Sampler:
    """Base class for sampling elements from a dataset.

    A Sampler provides an iterable over indices of a dataset, defining the order
    in which elements are visited. This base class supports sequential or shuffled
    sampling.

    Notes
    -----
    Samplers are used by DataLoader to determine the order and grouping of
    samples during iteration.

    The shuffle parameter determines whether indices are returned in sequential
    or randomized order.

    See Also
    --------
    BatchSampler : Wraps a sampler to yield batches of indices
    DataLoader : Uses samplers to iterate over dataset elements
    """

    def __init__(self, ds: Dataset, shuffle: bool = False):
        """
        Parameters
        ----------
        ds : Dataset
            Dataset to sample from.
        shuffle : bool, optional
            If True, samples are returned in random order. Default is False.
        """
        self.n = len(ds)
        self.shuffle = shuffle

    def __iter__(self):
        res = list(range(self.n))
        if self.shuffle:
            random.shuffle(res)
        return iter(res)


class BatchSampler:
    """Wraps a sampler to yield batches of indices.

    A BatchSampler takes a sampler that yields individual indices and wraps it to
    yield batches of indices instead. This is useful for mini-batch training where
    we want to process multiple samples at once.

    Notes
    -----
    The batch size determines how many indices are yielded in each batch. If
    drop_last is True, the last batch will be dropped if it's smaller than the
    batch size.

    The sampler can be any iterable that yields indices, but is typically an
    instance of Sampler.

    See Also
    --------
    Sampler : Base class for sampling individual indices
    """

    def __init__(
        self,
        sampler: Sampler | Iterable[int],
        batch_size: int,
        drop_last: bool = False,
    ):
        """
        Parameters
        ----------
        sampler : Sampler or Iterable[int]
            Sampler instance or iterable that yields indices.
        batch_size : int
            Number of indices to include in each batch.
        drop_last : bool, optional
            If True, drop the last batch if it's smaller than batch_size.
            Default is False.
        """
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
    """Iterator over a dataset that supports batching and parallel data loading.

    DataLoader combines a dataset and a sampler, and provides an iterable over
    the given dataset. It supports automatic batching, parallel data loading,
    and customizable data loading order.

    Notes
    -----
    The DataLoader provides an efficient way to load data in batches for training
    and evaluation. It handles the complexities of:

    - Batching individual data points into batches
    - Shuffling the data if requested
    - Parallel data loading using multiple worker processes
    - Custom collation of data samples into batches
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
        """
        Parameters
        ----------
        dataset : Dataset
            Dataset from which to load the data.
        batch_size : int, optional
            How many samples per batch to load. Default: 1.
        n_workers : int, optional
            How many subprocesses to use for data loading. Default: 1.
        shuffle : bool, optional
            Whether to shuffle the data at every epoch. Default: False.
        drop_last : bool, optional
            Whether to drop the last incomplete batch if dataset size is not
            divisible by batch_size. Default: False.
        collate_fn : callable, optional
            Merges a list of samples to form a mini-batch. Default: collate.
        """

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
    """Base class for all transforms.

    This class defines the interface for transformations that can be applied to data.
    Each transform should implement the __call__ method to specify how the data
    should be transformed.

    Notes
    -----
    Transforms are commonly used in computer vision tasks to augment training data
    and improve model generalization. They can include operations like flipping,
    rotating, cropping, or normalizing images.

    See Also
    --------
    RandomFlipHorizontal : Transform that randomly flips images horizontally
    RandomCrop : Transform that randomly crops images
    """

    def __call__(self, x):
        raise NotImplementedError()


class RandomFlipHorizontal(Transform):
    """Transform that randomly flips images (specified as H x W x C NDArray) horizontally.

    This transform applies horizontal flipping to images with a specified
    probability. Horizontal flipping is a common data augmentation technique
    that helps models become invariant to the horizontal orientation of objects
    in images.

    Notes
    -----
    The flip is applied with probability p (default 0.5). When applied, the image
    is flipped along its horizontal axis, meaning the left side becomes the right
    side and vice versa.

    See Also
    --------
    RandomCrop : Transform that randomly crops images
    """

    def __init__(self, p: int = 0.5):
        """
        Parameters
        ----------
        p : float, optional
            Probability of flipping the image horizontally. Default is 0.5.
        """
        self.p = p

    def __call__(self, img):
        """
        Parameters
        ----------
        img : ndarray
            H x W x C array representing an image

        Returns
        -------
        ndarray
            H x W x C array of flipped or original image
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            return np.flip(img, 1)
        return img


class RandomCrop(Transform):
    """Transform that randomly crops images after zero padding.

    This transform first applies zero padding around the image borders, then
    randomly crops the padded image back to its original size. This creates
    slight translations of the image content, which helps models become more
    robust to object position variations.

    Notes
    -----
    The padding size determines the maximum possible shift in any direction.
    For example, with padding=3, the image content can be shifted by up to
    3 pixels in any direction.

    The cropped region maintains the original image dimensions, effectively
    creating a translated version of the original image with zero padding
    filling in any gaps.

    See Also
    --------
    RandomFlipHorizontal : Transform that randomly flips images horizontally
    """

    def __init__(self, padding=3):
        """
        Parameters
        ----------
        padding : int, optional
            Number of pixels to pad around image borders. Default is 3.
        """
        self.padding = padding

    def __call__(self, img):
        """
        Parameters
        ----------
        img : ndarray
            H x W x C array representing an image

        Returns
        -------
        ndarray
            H x W x C array of randomly cropped image after padding
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
