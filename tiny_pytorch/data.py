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
