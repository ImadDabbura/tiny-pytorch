import numpy as np
import pytest

import tiny_pytorch.backend_ndarray.ndarray as nd


@pytest.mark.parametrize(
    "params",
    [
        {
            "shape": (4, 3),
            "np_fn": lambda X: X.reshape(2, 2, 3),
            "nd_fn": lambda X: X.reshape((2, 2, 3)),
        },
        {
            "shape": (16, 16),  # testing for compaction of large ndims array
            "np_fn": lambda X: X.reshape(2, 4, 2, 2, 2, 2, 2),
            "nd_fn": lambda X: X.reshape((2, 4, 2, 2, 2, 2, 2)),
        },
        {
            "shape": (
                2,
                4,
                2,
                2,
                2,
                2,
                2,
            ),  # testing for compaction of large ndims array
            "np_fn": lambda X: X.reshape(16, 16),
            "nd_fn": lambda X: X.reshape((16, 16)),
        },
        {
            "shape": (4, 4),
            "np_fn": lambda X: X.transpose(),
            "nd_fn": lambda X: X.permute((1, 0)),
        },
    ],
    ids=["reshape1", "reshape2", "reshape3", "permute"],
)
def test_compact(params):
    shape, np_fn, nd_fn = params["shape"], params["np_fn"], params["nd_fn"]
    _A = np.random.randint(low=0, high=10, size=shape)
    A = nd.NDArray(_A)

    lhs = nd_fn(A).compact()
    assert lhs.is_compact(), "array is not compact"

    rhs = np_fn(_A)
    np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5, rtol=1e-5)
