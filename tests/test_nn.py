import numpy as np

import tiny_pytorch.nn as nn
import tiny_pytorch.ops as ops
from tiny_pytorch.tensor import Tensor


def get_tensor(*shape, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")


def nn_linear_weight_init():
    np.random.seed(1337)
    f = nn.Linear(7, 4)
    f.weight.cached_data
    return f.weight.cached_data


def nn_linear_bias_init():
    np.random.seed(1337)
    f = nn.Linear(7, 4)
    return f.bias.cached_data


def linear_forward(lhs_shape, rhs_shape):
    np.random.seed(199)
    f = nn.Linear(*lhs_shape)
    f.bias.data = get_tensor(lhs_shape[-1])
    x = get_tensor(*rhs_shape)
    return f(x).cached_data


def linear_backward(lhs_shape, rhs_shape):
    np.random.seed(199)
    f = nn.Linear(*lhs_shape)
    f.bias.data = get_tensor(lhs_shape[-1])
    x = get_tensor(*rhs_shape)
    (f(x) ** 2).sum().backward()
    return x.grad.cached_data


def test_nn_linear_weight_init_1():
    np.testing.assert_allclose(
        nn_linear_weight_init(),
        np.array(
            [
                [
                    -4.4064468e-01,
                    -6.3199449e-01,
                    -4.1082984e-01,
                    -7.5330488e-02,
                ],
                [-3.3144259e-01, 3.4056887e-02, -4.4079605e-01, 8.8153863e-01],
                [4.3108878e-01, -7.1237373e-01, -2.1057765e-01, 2.3793796e-01],
                [-6.9425780e-01, 8.9535803e-01, -1.0512712e-01, 5.3615785e-01],
                [5.4460180e-01, -2.5689366e-01, -1.5534532e-01, 1.5601574e-01],
                [4.8174453e-01, -5.7806653e-01, -3.9223823e-01, 3.1518409e-01],
                [
                    -6.5129338e-04,
                    -5.9517515e-01,
                    -1.6083106e-01,
                    -5.5698222e-01,
                ],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_linear_bias_init_1():
    np.testing.assert_allclose(
        nn_linear_bias_init(),
        np.array(
            [[0.077647, 0.814139, -0.770975, 1.120297]], dtype=np.float32
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_linear_forward_1():
    np.testing.assert_allclose(
        linear_forward((10, 5), (1, 10)),
        np.array(
            [[3.849948, 9.50499, 2.38029, 5.572587, 5.668391]],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_linear_forward_2():
    np.testing.assert_allclose(
        linear_forward((10, 5), (3, 10)),
        np.array(
            [
                [7.763089, 10.086785, 0.380316, 6.242502, 6.944664],
                [2.548275, 7.747925, 5.343155, 2.065694, 9.871243],
                [2.871696, 7.466332, 4.236925, 2.461897, 8.209476],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_linear_forward_3():
    np.testing.assert_allclose(
        linear_forward((10, 5), (1, 3, 10)),
        np.array(
            [
                [
                    [4.351459, 8.782808, 3.935711, 3.03171, 8.014219],
                    [5.214458, 8.728788, 2.376814, 5.672185, 4.974319],
                    [1.343204, 8.639378, 2.604359, -0.282955, 9.864498],
                ]
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_linear_backward_1():
    np.testing.assert_allclose(
        linear_backward((10, 5), (1, 10)),
        np.array(
            [
                [
                    20.61148,
                    6.920893,
                    -1.625556,
                    -13.497676,
                    -6.672813,
                    18.762121,
                    7.286628,
                    8.18535,
                    2.741301,
                    5.723689,
                ]
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_linear_backward_2():
    print(linear_backward((10, 5), (3, 10)))
    np.testing.assert_allclose(
        linear_backward((10, 5), (3, 10)),
        np.array(
            [
                [
                    24.548800,
                    8.775347,
                    4.387898,
                    -21.248514,
                    -3.9669373,
                    24.256767,
                    6.3171115,
                    6.029777,
                    0.8809935,
                    3.5995162,
                ],
                [
                    12.233745,
                    -3.792646,
                    -4.1903896,
                    -5.106719,
                    -12.004269,
                    11.967942,
                    11.939469,
                    19.314493,
                    10.631226,
                    14.510731,
                ],
                [
                    12.920014,
                    -1.4545978,
                    -3.0892954,
                    -6.762379,
                    -9.713004,
                    12.523148,
                    9.904757,
                    15.442993,
                    8.044141,
                    11.4106865,
                ],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_linear_backward_3():
    print(linear_backward((10, 5), (1, 3, 10)))
    np.testing.assert_allclose(
        linear_backward((10, 5), (1, 3, 10)),
        np.array(
            [
                [
                    [
                        16.318823,
                        0.3890714,
                        -2.3196607,
                        -10.607947,
                        -8.891977,
                        16.04581,
                        9.475689,
                        14.571134,
                        6.581477,
                        10.204643,
                    ],
                    [
                        20.291656,
                        7.48733,
                        1.2581345,
                        -14.285493,
                        -6.0252004,
                        19.621624,
                        4.343303,
                        6.973201,
                        -0.8103489,
                        4.037069,
                    ],
                    [
                        11.332953,
                        -5.698288,
                        -8.815561,
                        -7.673438,
                        -7.6161675,
                        9.361553,
                        17.341637,
                        17.269142,
                        18.1076,
                        14.261493,
                    ],
                ]
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def relu_forward(*shape):
    f = nn.ReLU()
    x = get_tensor(*shape)
    return f(x).cached_data


def relu_backward(*shape):
    f = nn.ReLU()
    x = get_tensor(*shape)
    (f(x) ** 2).sum().backward()
    return x.grad.cached_data


def test_nn_relu_forward_1():
    np.testing.assert_allclose(
        relu_forward(2, 2),
        np.array([[3.35, 4.2], [0.25, 4.5]], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_relu_backward_1():
    np.testing.assert_allclose(
        relu_backward(3, 2),
        np.array([[7.5, 2.7], [0.6, 0.2], [0.3, 6.7]], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def sequential_forward(batches=3):
    np.random.seed(42)
    f = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 5))
    x = get_tensor(batches, 5)
    return f(x).cached_data


def sequential_backward(batches=3):
    np.random.seed(42)
    f = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 5))
    x = get_tensor(batches, 5)
    f(x).sum().backward()
    return x.grad.cached_data


def test_nn_sequential_forward_1():
    print(sequential_forward(batches=3))
    np.testing.assert_allclose(
        sequential_forward(batches=3),
        np.array(
            [
                [3.296263, 0.057031, 2.97568, -4.618432, -0.902491],
                [2.465332, -0.228394, 2.069803, -3.772378, -0.238334],
                [3.04427, -0.25623, 3.848721, -6.586399, -0.576819],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_sequential_backward_1():
    np.testing.assert_allclose(
        sequential_backward(batches=3),
        np.array(
            [
                [0.802697, -1.0971, 0.120842, 0.033051, 0.241105],
                [-0.364489, 0.651385, 0.482428, 0.925252, -1.233545],
                [0.802697, -1.0971, 0.120842, 0.033051, 0.241105],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def logsumexp_forward(shape, axes):
    x = get_tensor(*shape)
    return (ops.LogSumExp(axes=axes)(x)).cached_data


def logsumexp_backward(shape, axes):
    x = get_tensor(*shape)
    y = (ops.LogSumExp(axes=axes)(x) ** 2).sum()
    y.backward()
    return x.grad.cached_data


def test_op_logsumexp_forward_1():
    np.testing.assert_allclose(
        logsumexp_forward((3, 3, 3), (1, 2)),
        np.array([5.366029, 4.9753823, 6.208126], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_op_logsumexp_forward_2():
    np.testing.assert_allclose(
        logsumexp_forward((3, 3, 3), None),
        np.array([6.7517853], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_op_logsumexp_forward_3():
    np.testing.assert_allclose(
        logsumexp_forward((1, 2, 3, 4), (0, 2)),
        np.array(
            [
                [5.276974, 5.047317, 3.778802, 5.0103745],
                [5.087831, 4.391712, 5.025037, 2.0214698],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_op_logsumexp_forward_4():
    np.testing.assert_allclose(
        logsumexp_forward((3, 10), (1,)),
        np.array([5.705309, 5.976375, 5.696459], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_op_logsumexp_forward_5():
    test_data = ops.LogSumExp((0,))(
        Tensor(np.array([[1e10, 1e9, 1e8, -10], [1e-10, 1e9, 1e8, -10]])),
    ).numpy()
    np.testing.assert_allclose(
        test_data,
        np.array(
            [1.00000000e10, 1.00000000e09, 1.00000001e08, -9.30685282e00]
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_op_logsumexp_backward_1():
    np.testing.assert_allclose(
        logsumexp_backward((3, 1), (1,)),
        np.array([[1.0], [7.3], [9.9]], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_op_logsumexp_backward_2():
    np.testing.assert_allclose(
        logsumexp_backward((3, 3, 3), (1, 2)),
        np.array(
            [
                [
                    [1.4293308, 1.2933122, 0.82465225],
                    [0.50017685, 2.1323113, 2.1323113],
                    [1.4293308, 0.58112264, 0.40951014],
                ],
                [
                    [0.3578173, 0.07983983, 4.359107],
                    [1.1300558, 0.561169, 0.1132981],
                    [0.9252113, 0.65198547, 1.7722803],
                ],
                [
                    [0.2755132, 2.365242, 2.888913],
                    [0.05291228, 1.1745441, 0.02627547],
                    [2.748018, 0.13681579, 2.748018],
                ],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_op_logsumexp_backward_3():
    np.testing.assert_allclose(
        logsumexp_backward((3, 3, 3), (0, 2)),
        np.array(
            [
                [
                    [0.92824626, 0.839912, 0.5355515],
                    [0.59857905, 2.551811, 2.551811],
                    [1.0213376, 0.41524494, 0.29261813],
                ],
                [
                    [0.16957533, 0.03783737, 2.0658503],
                    [0.98689, 0.49007502, 0.09894446],
                    [0.48244575, 0.3399738, 0.9241446],
                ],
                [
                    [0.358991, 3.081887, 3.764224],
                    [0.12704718, 2.820187, 0.06308978],
                    [3.9397335, 0.19614778, 3.9397335],
                ],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_op_logsumexp_backward_5():
    grad_compare = Tensor(
        np.array([[1e10, 1e9, 1e8, -10], [1e-10, 1e9, 1e8, -10]])
    )
    _ = (ops.LogSumExp((0,))(grad_compare) ** 2).sum().backward()
    np.testing.assert_allclose(
        grad_compare.grad.cached_data,
        np.array(
            [
                [2.00000000e10, 9.99999999e08, 1.00000001e08, -9.30685282e00],
                [0.00000000e00, 9.99999999e08, 1.00000001e08, -9.30685282e00],
            ]
        ),
        rtol=1e-5,
        atol=1e-5,
    )
