import numpy as np
import pytest
import torch

import tiny_pytorch.backend_ndarray.ndarray as nd
from tiny_pytorch import Tensor, ops
from tiny_pytorch.tensor import TensorTuple

_DEVICES = [
    nd.cpu(),
    pytest.param(
        nd.cuda(),
        marks=pytest.mark.skipif(not nd.cuda().enabled(), reason="No GPU"),
    ),
]


def compare_strides(a_np, a_nd):
    size = a_np.itemsize
    assert tuple([x // size for x in a_np.strides]) == a_nd.strides


def check_same_memory(original, view):
    assert original._handle.ptr() == view._handle.ptr()


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
        {
            "shape": (4, 1, 4),
            "np_fn": lambda X: np.broadcast_to(X, shape=(4, 5, 4)),
            "nd_fn": lambda X: X.broadcast_to((4, 5, 4)),
        },
        {
            "shape": (8, 8),
            "np_fn": lambda X: X[4:, 4:],
            "nd_fn": lambda X: X[4:, 4:],
        },
        {
            "shape": (8, 8, 2, 2, 2, 2),
            "np_fn": lambda X: X[1:3, 5:8, 1:2, 0:1, 0:1, 1:2],
            "nd_fn": lambda X: X[1:3, 5:8, 1:2, 0:1, 0:1, 1:2],
        },
        {
            "shape": (7, 8),
            "np_fn": lambda X: X.transpose()[3:7, 2:5],
            "nd_fn": lambda X: X.permute((1, 0))[3:7, 2:5],
        },
    ],
    ids=[
        "reshape1",
        "reshape2",
        "reshape3",
        "permute",
        "broadcast_to",
        "getitem1",
        "getitem2",
        "transposegetitem",
    ],
)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_compact(params, device):
    shape, np_fn, nd_fn = params["shape"], params["np_fn"], params["nd_fn"]
    _A = np.random.randint(low=0, high=10, size=shape)
    A = nd.array(_A, device=device)

    lhs = nd_fn(A).compact()
    assert lhs.is_compact(), "array is not compact"

    rhs = np_fn(_A)
    np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5, rtol=1e-5)


reduce_params = [
    {"dims": (10,), "axis": 0},
    {"dims": (4, 5, 6), "axis": 0},
    {"dims": (4, 5, 6), "axis": 1},
    {"dims": (4, 5, 6), "axis": 2},
]


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("params", reduce_params)
def test_reduce_sum(params, device):
    dims, axis = params["dims"], params["axis"]
    _A = np.random.randn(*dims)
    A = nd.array(_A, device=device)
    np.testing.assert_allclose(
        _A.sum(axis=axis, keepdims=True),
        A.sum(axis=axis, keepdims=True).numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("params", reduce_params)
def test_reduce_max(params, device):
    dims, axis = params["dims"], params["axis"]
    _A = np.random.randn(*dims)
    A = nd.array(_A, device=device)
    np.testing.assert_allclose(
        _A.max(axis=axis, keepdims=True),
        A.max(axis=axis, keepdims=True).numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


class _ShapeAndSlices(nd.NDArray):
    def __getitem__(self, idxs):
        idxs = tuple(
            [
                (
                    self._process_idx(s, i)
                    if isinstance(s, slice)
                    else slice(s, s + 1, 1)
                )
                for i, s in enumerate(idxs)
            ]
        )
        return self.shape, idxs


ShapeAndSlices = lambda *shape: _ShapeAndSlices(np.ones(shape))  # noqa: E731


@pytest.mark.parametrize(
    "params",
    [
        {
            "lhs": ShapeAndSlices(4, 5, 6)[1:2, 0, 0],
            "rhs": ShapeAndSlices(7, 7, 7)[1:2, 0, 0],
        },
        {
            "lhs": ShapeAndSlices(4, 5, 6)[1:4:2, 0, 0],
            "rhs": ShapeAndSlices(7, 7, 7)[1:3, 0, 0],
        },
        {
            "lhs": ShapeAndSlices(4, 5, 6)[1:3, 2:5, 2:6],
            "rhs": ShapeAndSlices(7, 7, 7)[:2, :3, :4],
        },
    ],
)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_setitem_ewise(params, device):
    lhs_shape, lhs_slices = params["lhs"]
    rhs_shape, rhs_slices = params["rhs"]
    _A = np.random.randn(*lhs_shape)
    _B = np.random.randn(*rhs_shape)
    A = nd.array(_A, device=device)
    B = nd.array(_B, device=device)
    start_ptr = A._handle.ptr()
    A[lhs_slices] = B[rhs_slices]
    _A[lhs_slices] = _B[rhs_slices]
    end_ptr = A._handle.ptr()
    assert start_ptr == end_ptr, "you should modify in-place"
    compare_strides(_A, A)
    np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)


# Ex: We want arrays of size (4, 5, 6) setting element(s) [1:4, 2, 3] to a scalar
@pytest.mark.parametrize(
    "params",
    [
        ShapeAndSlices(4, 5, 6)[1, 2, 3],
        ShapeAndSlices(4, 5, 6)[1:4, 2, 3],
        ShapeAndSlices(4, 5, 6)[:4, 2:5, 3],
        ShapeAndSlices(4, 5, 6)[1::2, 2:5, ::2],
    ],
)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_setitem_scalar(params, device):
    shape, slices = params
    _A = np.random.randn(*shape)
    A = nd.array(_A, device=device)
    # probably tear these out using lambdas
    start_ptr = A._handle.ptr()
    _A[slices] = 4.0
    A[slices] = 4.0
    end_ptr = A._handle.ptr()
    assert start_ptr == end_ptr, "you should modify in-place"
    np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)
    compare_strides(_A, A)


matmul_tiled_shapes = [(1, 1, 1), (2, 2, 3), (1, 2, 1), (3, 3, 3)]


@pytest.mark.parametrize("m,n,p", matmul_tiled_shapes)
def test_matmul_tiled(m, n, p):
    device = nd.cpu()
    assert hasattr(device, "matmul_tiled")
    t = device.__tile_size__
    A = nd.array(np.random.randn(m, n, t, t), device=nd.cpu())
    B = nd.array(np.random.randn(n, p, t, t), device=nd.cpu())
    C = nd.NDArray.make((m, p, t, t), device=nd.cpu())
    device.matmul_tiled(A._handle, B._handle, C._handle, m * t, n * t, p * t)

    lhs = A.numpy().transpose(0, 2, 1, 3).flatten().reshape(
        m * t, n * t
    ) @ B.numpy().transpose(0, 2, 1, 3).flatten().reshape(n * t, p * t)
    rhs = C.numpy().transpose(0, 2, 1, 3).flatten().reshape(m * t, p * t)

    np.testing.assert_allclose(lhs, rhs, atol=1e-5, rtol=1e-5)


OPS = {
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "equal": lambda a, b: a == b,
    "greater_than": lambda a, b: a >= b,
}
OP_FNS = [OPS[k] for k in OPS]
OP_NAMES = [k for k in OPS]

ewise_shapes = [(1, 1, 1), (4, 5, 6)]


@pytest.mark.parametrize("fn", OP_FNS, ids=OP_NAMES)
@pytest.mark.parametrize("shape", ewise_shapes)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ewise_fn(fn, shape, device):
    _A = np.random.randn(*shape)
    _B = np.random.randn(*shape)
    A = nd.array(_A, device=device)
    B = nd.array(_B, device=device)
    np.testing.assert_allclose(
        fn(_A, _B), fn(A, B).numpy(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape", ewise_shapes)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ewise_max(shape, device):
    _A = np.random.randn(*shape)
    _B = np.random.randn(*shape)
    A = nd.array(_A, device=device)
    B = nd.array(_B, device=device)
    np.testing.assert_allclose(
        np.maximum(_A, _B), A.maximum(B).numpy(), atol=1e-5, rtol=1e-5
    )


permute_params = [
    {"dims": (4, 5, 6), "axes": (0, 1, 2)},
    {"dims": (4, 5, 6), "axes": (1, 0, 2)},
    {"dims": (4, 5, 6), "axes": (2, 1, 0)},
]


@pytest.mark.parametrize("params", permute_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_permute(device, params):
    dims = params["dims"]
    axes = params["axes"]
    _A = np.random.randn(*dims)
    A = nd.array(_A, device=device)
    lhs = np.transpose(_A, axes=axes)
    rhs = A.permute(axes)
    np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
    compare_strides(lhs, rhs)
    check_same_memory(A, rhs)


reshape_params = [
    {"shape": (8, 16), "new_shape": (2, 4, 16)},
    {"shape": (8, 16), "new_shape": (8, 4, 2, 2)},
]


@pytest.mark.parametrize("params", reshape_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_reshape(device, params):
    shape = params["shape"]
    new_shape = params["new_shape"]
    _A = np.random.randn(*shape)
    A = nd.array(_A, device=device)
    lhs = _A.reshape(*new_shape)
    rhs = A.reshape(new_shape)
    np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
    compare_strides(lhs, rhs)
    check_same_memory(A, rhs)


getitem_params = [
    {"shape": (8, 16), "fn": lambda X: X[3:4, 3:4]},
    {"shape": (8, 16), "fn": lambda X: X[1:2, 1:3]},
    {"shape": (8, 16), "fn": lambda X: X[3:4, 1:4]},
    {"shape": (8, 16), "fn": lambda X: X[1:4, 3:4]},
]


@pytest.mark.parametrize("params", getitem_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_getitem(device, params):
    fn = params["fn"]
    _A = np.random.randn(5, 5)
    A = nd.array(_A, device=device)
    lhs = fn(_A)
    rhs = fn(A)
    np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
    compare_strides(lhs, rhs)
    check_same_memory(A, rhs)


broadcast_params = [
    {"from_shape": (1, 3, 4), "to_shape": (6, 3, 4)},
]


@pytest.mark.parametrize("params", broadcast_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_broadcast_to(device, params):
    from_shape, to_shape = params["from_shape"], params["to_shape"]
    _A = np.random.randn(*from_shape)
    A = nd.array(_A, device=device)
    lhs = np.broadcast_to(_A, shape=to_shape)
    rhs = A.broadcast_to(to_shape)
    np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
    compare_strides(lhs, rhs)
    check_same_memory(A, rhs)


matmul_dims = [
    (16, 16, 16),
    (8, 8, 8),
    (1, 2, 3),
    (3, 4, 5),
    (5, 4, 3),
    (64, 64, 64),
    (72, 72, 72),
    (72, 73, 74),
    (74, 73, 72),
    (128, 128, 128),
]


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("m,n,p", matmul_dims)
def test_matmul(m, n, p, device):
    _A = np.random.randn(m, n)
    _B = np.random.randn(n, p)
    A = nd.array(_A, device=device)
    B = nd.array(_B, device=device)
    np.testing.assert_allclose((A @ B).numpy(), _A @ _B, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_mul(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    np.testing.assert_allclose(
        A * 5.0, (B * 5.0).numpy(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_div(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    np.testing.assert_allclose(
        A / 5.0, (B / 5.0).numpy(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_power(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    np.testing.assert_allclose(
        np.power(A, 5.0), (B**5.0).numpy(), atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        np.power(A, 0.5), (B**0.5).numpy(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_maximum(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    C = (np.max(A) + 1.0).item()
    np.testing.assert_allclose(
        np.maximum(A, C), (B.maximum(C)).numpy(), atol=1e-5, rtol=1e-5
    )
    C = (np.max(A) - 1.0).item()
    np.testing.assert_allclose(
        np.maximum(A, C), (B.maximum(C)).numpy(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_eq(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    C = A[0, 1].item()
    np.testing.assert_allclose(A == C, (B == C).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_ge(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    C = A[0, 1].item()
    np.testing.assert_allclose(A >= C, (B >= C).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ewise_log(device):
    A = np.abs(np.random.randn(5, 5))
    B = nd.array(A, device=device)
    np.testing.assert_allclose(
        np.log(A), (B.log()).numpy(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ewise_exp(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    np.testing.assert_allclose(
        np.exp(A), (B.exp()).numpy(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ewise_tanh(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    np.testing.assert_allclose(
        np.tanh(A), (B.tanh()).numpy(), atol=1e-5, rtol=1e-5
    )


STACK_PARAMETERS = [((5, 5), 0, 1), ((5, 5), 0, 2), ((1, 5, 7), 2, 5)]


@pytest.mark.parametrize("shape, axis, length", STACK_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_stack(shape, axis, length, device):
    _A = [np.random.randn(*shape).astype(np.float32) for i in range(length)]
    A = [Tensor(nd.array(_A[i]), device=device) for i in range(length)]
    A_t = [torch.Tensor(_A[i]) for i in range(length)]
    out = ops.stack(A, axis=axis)
    out_t = torch.stack(A_t, dim=axis)
    np.testing.assert_allclose(
        out_t.numpy(), out.numpy(), atol=1e-5, rtol=1e-5
    )


pad_params = [
    {"shape": (10, 32, 32, 8), "padding": ((0, 0), (2, 2), (2, 2), (0, 0))},
    {"shape": (10, 32, 32, 8), "padding": ((0, 0), (0, 0), (0, 0), (0, 0))},
]


@pytest.mark.parametrize("device", [nd.cpu()])
@pytest.mark.parametrize("params", pad_params)
def test_pad_forward(params, device):
    np.random.seed(0)
    shape, padding = params["shape"], params["padding"]
    _A = np.random.randn(*shape)
    _B = np.pad(_A, padding)
    A = nd.NDArray(_A, device=device)
    B = A.pad(padding)
    assert np.linalg.norm(B.numpy() - _B) < 1e-4


flip_forward_params = [
    {"shape": (10, 5), "axes": (0,)},
    {"shape": (10, 5), "axes": (1,)},
    {"shape": (10, 5), "axes": (0, 1)},
    {"shape": (10, 32, 32, 8), "axes": (0, 1)},
    {"shape": (3, 3, 6, 8), "axes": (0, 1)},
    {"shape": (10, 32, 32, 8), "axes": (1, 2)},
    {"shape": (3, 3, 6, 8), "axes": (1, 2)},
    {"shape": (10, 32, 32, 8), "axes": (2, 3)},
    {"shape": (3, 3, 6, 8), "axes": (2, 3)},
    {"shape": (10, 32, 32, 8), "axes": (0, 1, 2, 3)},
]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", flip_forward_params)
def test_flip_forward(params, device):
    np.random.seed(0)
    shape, axes = params["shape"], params["axes"]
    _A = np.random.randn(*shape)
    _B = np.flip(_A, axes)
    A = Tensor(_A, device=device)
    B = ops.flip(A, axes=axes)

    assert np.linalg.norm(B.numpy() - _B) < 1e-4


@pytest.mark.parametrize("device", _DEVICES)
def test_dilate_forward(device):
    np.random.seed(0)
    # device = ndl.cpu()

    _A = np.random.randint(1, 10, size=(2, 5))
    A = Tensor(_A, device=device)
    assert (
        np.linalg.norm(
            ops.dilate(A, dilation=0, axes=(0,)).numpy()
            - np.array([[6.0, 1.0, 4.0, 4.0, 8.0], [4.0, 6.0, 3.0, 5.0, 8.0]])
        )
        < 1e-5
    )

    _A = np.random.randint(1, 10, size=(2, 5))
    A = Tensor(_A, device=device)
    assert (
        np.linalg.norm(
            ops.dilate(A, dilation=1, axes=(0,)).numpy()
            - np.array(
                [
                    [7.0, 9.0, 9.0, 2.0, 7.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [8.0, 8.0, 9.0, 2.0, 6.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        )
        < 1e-5
    )

    _A = np.random.randint(1, 10, size=(2, 5))
    A = Tensor(_A, device=device)
    assert (
        np.linalg.norm(
            ops.dilate(A, dilation=1, axes=(1,)).numpy()
            - np.array(
                [
                    [9.0, 0.0, 5.0, 0.0, 4.0, 0.0, 1.0, 0.0, 4.0, 0.0],
                    [6.0, 0.0, 1.0, 0.0, 3.0, 0.0, 4.0, 0.0, 9.0, 0.0],
                ]
            )
        )
        < 1e-5
    )

    _A = np.random.randint(1, 10, size=(2, 5))
    A = Tensor(_A, device=device)
    assert (
        np.linalg.norm(
            ops.dilate(A, dilation=1, axes=(0, 1)).numpy()
            - np.array(
                [
                    [2.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0, 8.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 5.0, 0.0, 8.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        )
        < 1e-5
    )

    _A = np.random.randint(1, 10, size=(2, 2))
    A = Tensor(_A, device=device)
    assert (
        np.linalg.norm(
            ops.dilate(A, dilation=2, axes=(0, 1)).numpy()
            - np.array(
                [
                    [4.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [8.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        )
        < 1e-5
    )

    _A = np.random.randint(1, 10, size=(2, 2, 2, 2))
    A = Tensor(_A, device=device)
    assert (
        np.linalg.norm(
            ops.dilate(A, dilation=1, axes=(1, 2)).numpy()
            - np.array(
                [
                    [
                        [[1.0, 1.0], [0.0, 0.0], [5.0, 6.0], [0.0, 0.0]],
                        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        [[6.0, 7.0], [0.0, 0.0], [9.0, 5.0], [0.0, 0.0]],
                        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    ],
                    [
                        [[2.0, 5.0], [0.0, 0.0], [9.0, 2.0], [0.0, 0.0]],
                        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        [[2.0, 8.0], [0.0, 0.0], [4.0, 7.0], [0.0, 0.0]],
                        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    ],
                ]
            )
        )
        < 1e-5
    )


def backward_check(f, *args, **kwargs):
    eps = 1e-3
    out = f(*args, **kwargs)
    c = np.random.randn(*out.shape)
    is_stacked = False
    if isinstance(args[0], list):
        args = args[0]
        is_stacked = True
    numerical_grad = [np.zeros(a.shape) for a in args]
    num_args = len(args)
    for i in range(num_args):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            if is_stacked:
                f1 = (f(args, **kwargs).numpy() * c).sum()
            else:
                f1 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] -= 2 * eps
            if is_stacked:
                f2 = (f(args, **kwargs).numpy() * c).sum()
            else:
                f2 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] += eps
            numerical_grad[i].flat[j] = (f1 - f2) / (2 * eps)
    backward_grad = out.op.gradient_as_tuple(
        Tensor(c, device=args[0].device), out
    )
    if isinstance(backward_grad[0], TensorTuple):  # TODO keep this?
        backward_grad = backward_grad[0].tuple()
    error = sum(
        np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i])
        for i in range(len(args))
    )
    assert error < 1e-2
    return [g.numpy() for g in backward_grad]


dilate_backward_params = [
    {"shape": (2, 5), "d": 1, "axes": (0,)},
    {"shape": (2, 5), "d": 2, "axes": (1,)},
    {"shape": (2, 5), "d": 1, "axes": (0, 1)},
    {"shape": (2, 5), "d": 0, "axes": (0, 1)},
    {"shape": (2, 3, 3, 4), "d": 2, "axes": (0, 1)},
    {"shape": (3, 3, 6, 4), "d": 3, "axes": (0, 1)},
    {"shape": (2, 3, 3, 4), "d": 0, "axes": (1, 2)},
    {"shape": (2, 3, 3, 4), "d": 1, "axes": (1, 2)},
    {"shape": (3, 3, 6, 4), "d": 1, "axes": (1, 2)},
    {"shape": (2, 3, 3, 4), "d": 1, "axes": (2, 3)},
    {"shape": (3, 3, 6, 4), "d": 1, "axes": (2, 3)},
    {"shape": (2, 3, 3, 4), "d": 1, "axes": (0, 1, 2, 3)},
]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", dilate_backward_params)
def test_dilate_backward(params, device):
    np.random.seed(0)
    shape, d, axes = params["shape"], params["d"], params["axes"]
    backward_check(
        ops.dilate,
        Tensor(np.random.randn(*shape), device=device),
        dilation=d,
        axes=axes,
    )
