import numpy as np
import pytest
import torch

import tiny_pytorch.backend_ndarray.ndarray as nd
import tiny_pytorch.nn as nn
import tiny_pytorch.ops as ops
from tiny_pytorch.tensor import Tensor

np.random.seed(3)


_DEVICES = [
    nd.cpu(),
    pytest.param(
        nd.cuda(),
        marks=pytest.mark.skipif(not nd.cuda().enabled(), reason="No GPU"),
    ),
]


BATCH_SIZES = [1, 15]
INPUT_SIZES = [1, 11]
HIDDEN_SIZES = [1, 12]
BIAS = [True, False]
INIT_HIDDEN = [True, False]
NONLINEARITIES = ["tanh", "relu"]


def get_tensor(*shape, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")


def get_int_tensor(*shape, low=0, high=10, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return Tensor(np.random.randint(low, high, size=shape))


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


class UselessModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.stuff = {
            "layer1": nn.Linear(4, 4),
            "layer2": [nn.Dropout(0.1), nn.Sequential(nn.Linear(4, 4))],
        }

    def forward(self, x):
        raise NotImplementedError()


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


def softmax_loss_forward(rows, classes):
    x = get_tensor(rows, classes)
    y = get_int_tensor(rows, low=0, high=classes)
    f = nn.SoftmaxLoss()
    return np.array(f(x, y).cached_data)


def softmax_loss_backward(rows, classes):
    x = get_tensor(rows, classes)
    y = get_int_tensor(rows, low=0, high=classes)
    f = nn.SoftmaxLoss()
    loss = f(x, y)
    loss.backward()
    return x.grad.cached_data


def test_nn_softmax_loss_forward_1():
    np.testing.assert_allclose(
        softmax_loss_forward(5, 10),
        np.array(4.041218, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_softmax_loss_forward_2():
    np.testing.assert_allclose(
        softmax_loss_forward(3, 11),
        np.array(3.3196716, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_softmax_loss_backward_1():
    np.testing.assert_allclose(
        softmax_loss_backward(5, 10),
        np.array(
            [
                [
                    0.00068890385,
                    0.0015331834,
                    0.013162163,
                    -0.16422154,
                    0.023983022,
                    0.0050903494,
                    0.00076135644,
                    0.050772052,
                    0.0062173656,
                    0.062013146,
                ],
                [
                    0.012363418,
                    0.02368262,
                    0.11730081,
                    0.001758993,
                    0.004781439,
                    0.0029000894,
                    -0.19815083,
                    0.017544521,
                    0.015874943,
                    0.0019439887,
                ],
                [
                    0.001219767,
                    0.08134181,
                    0.057320606,
                    0.0008595553,
                    0.0030001428,
                    0.0009499555,
                    -0.19633561,
                    0.0008176346,
                    0.0014898272,
                    0.0493363,
                ],
                [
                    -0.19886842,
                    0.08767337,
                    0.017700946,
                    0.026406704,
                    0.0013147127,
                    0.0107361665,
                    0.009714483,
                    0.023893777,
                    0.019562569,
                    0.0018656658,
                ],
                [
                    0.007933789,
                    0.017656967,
                    0.027691642,
                    0.0005605318,
                    0.05576411,
                    0.0013114461,
                    0.06811045,
                    0.011835824,
                    0.0071787895,
                    -0.19804356,
                ],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_softmax_loss_backward_2():
    np.testing.assert_allclose(
        softmax_loss_backward(3, 11),
        np.array(
            [
                [
                    0.0027466794,
                    0.020295369,
                    0.012940894,
                    0.04748398,
                    0.052477922,
                    0.090957515,
                    0.0028875037,
                    0.012940894,
                    0.040869843,
                    0.04748398,
                    -0.33108455,
                ],
                [
                    0.0063174255,
                    0.001721699,
                    0.09400159,
                    0.0034670753,
                    0.038218185,
                    0.009424488,
                    0.0042346967,
                    0.08090791,
                    -0.29697907,
                    0.0044518122,
                    0.054234188,
                ],
                [
                    0.14326698,
                    0.002624026,
                    0.0032049934,
                    0.01176007,
                    0.045363605,
                    0.0043262867,
                    0.039044812,
                    0.017543964,
                    0.0037236712,
                    -0.3119051,
                    0.04104668,
                ],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def layernorm_forward(shape, dim):
    f = nn.LayerNorm1d(dim)
    x = get_tensor(*shape)
    return f(x).cached_data


def layernorm_backward(shape, dims):
    f = nn.LayerNorm1d(dims)
    x = get_tensor(*shape)
    (f(x) ** 4).sum().backward()
    return x.grad.cached_data


def test_nn_layernorm_forward_1():
    np.testing.assert_allclose(
        layernorm_forward((3, 3), 3),
        np.array(
            [
                [-0.06525002, -1.1908097, 1.2560595],
                [1.3919864, -0.47999576, -0.911992],
                [1.3628436, -1.0085043, -0.3543393],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_layernorm_forward_2():
    np.testing.assert_allclose(
        layernorm_forward((2, 10), 10),
        np.array(
            [
                [
                    0.8297899,
                    1.6147263,
                    -1.525019,
                    -0.4036814,
                    0.306499,
                    0.08223152,
                    0.6429003,
                    -1.3381294,
                    0.8671678,
                    -1.0764838,
                ],
                [
                    -1.8211555,
                    0.39098236,
                    -0.5864739,
                    0.853988,
                    -0.3806936,
                    1.2655486,
                    0.33953735,
                    1.522774,
                    -0.8951442,
                    -0.68936396,
                ],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_layernorm_forward_3():
    np.testing.assert_allclose(
        layernorm_forward((1, 5), 5),
        np.array(
            [[-1.0435007, -0.8478443, 0.7500162, -0.42392215, 1.565251]],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_layernorm_backward_1():
    np.testing.assert_allclose(
        layernorm_backward((3, 3), 3),
        np.array(
            [
                [-2.8312206e-06, -6.6757202e-05, 6.9618225e-05],
                [1.9950867e-03, -6.8092346e-04, -1.3141632e-03],
                [4.4703484e-05, -3.2544136e-05, -1.1801720e-05],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_layernorm_backward_2():
    np.testing.assert_allclose(
        layernorm_backward((2, 10), 10),
        np.array(
            [
                [
                    -2.301574,
                    4.353944,
                    -1.9396116,
                    2.4330146,
                    -1.1070801,
                    0.01571643,
                    -2.209449,
                    0.49513134,
                    -2.261348,
                    2.5212562,
                ],
                [
                    -9.042961,
                    -2.6184766,
                    4.5592957,
                    -4.2109876,
                    3.4247458,
                    -1.9075732,
                    -2.2689414,
                    2.110825,
                    5.044025,
                    4.910048,
                ],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_layernorm_backward_3():
    np.testing.assert_allclose(
        layernorm_backward((1, 5), 5),
        np.array(
            [[0.150192, 0.702322, -3.321343, 0.31219, 2.156639]],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_layernorm_backward_4():
    np.testing.assert_allclose(
        layernorm_backward((5, 1), 1),
        np.array([[0], [0], [0], [0], [0]], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def flatten_forward(*shape):
    x = get_tensor(*shape)
    tform = nn.Flatten()
    return tform(x).cached_data


def flatten_backward(*shape):
    x = get_tensor(*shape)
    tform = nn.Flatten()
    (tform(x) ** 2).sum().backward()
    return x.grad.cached_data


def test_nn_flatten_forward_1():
    np.testing.assert_allclose(
        flatten_forward(3, 3),
        np.array(
            [[2.1, 0.95, 3.45], [3.1, 2.45, 2.3], [3.3, 0.4, 1.2]],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_flatten_forward_2():
    np.testing.assert_allclose(
        flatten_forward(3, 3, 3),
        np.array(
            [
                [3.35, 3.25, 2.8, 2.3, 3.75, 3.75, 3.35, 2.45, 2.1],
                [1.65, 0.15, 4.15, 2.8, 2.1, 0.5, 2.6, 2.25, 3.25],
                [2.4, 4.55, 4.75, 0.75, 3.85, 0.05, 4.7, 1.7, 4.7],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_flatten_forward_3():
    np.testing.assert_allclose(
        flatten_forward(1, 2, 3, 4),
        np.array(
            [
                [
                    4.2,
                    4.5,
                    1.9,
                    4.85,
                    4.85,
                    3.3,
                    2.7,
                    3.05,
                    0.3,
                    3.65,
                    3.1,
                    0.1,
                    4.5,
                    4.05,
                    3.05,
                    0.15,
                    3.0,
                    1.65,
                    4.85,
                    1.3,
                    3.95,
                    2.9,
                    1.2,
                    1.0,
                ]
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_flatten_forward_4():
    np.testing.assert_allclose(
        flatten_forward(3, 3, 4, 4),
        np.array(
            [
                [
                    0.95,
                    1.1,
                    1.0,
                    1.0,
                    4.9,
                    0.25,
                    1.6,
                    0.35,
                    1.5,
                    3.4,
                    1.75,
                    3.4,
                    4.8,
                    1.4,
                    2.35,
                    3.2,
                    1.65,
                    1.9,
                    3.05,
                    0.35,
                    3.15,
                    4.05,
                    3.3,
                    2.2,
                    2.5,
                    1.5,
                    3.25,
                    0.65,
                    3.05,
                    0.75,
                    3.25,
                    2.55,
                    0.55,
                    0.25,
                    3.65,
                    3.4,
                    0.05,
                    1.4,
                    0.75,
                    1.55,
                    4.45,
                    0.2,
                    3.35,
                    2.45,
                    3.45,
                    4.75,
                    2.45,
                    4.3,
                ],
                [
                    1.0,
                    0.2,
                    0.4,
                    0.7,
                    4.9,
                    4.2,
                    2.55,
                    3.15,
                    1.2,
                    3.8,
                    1.35,
                    1.85,
                    3.15,
                    2.7,
                    1.5,
                    1.35,
                    4.85,
                    4.2,
                    1.5,
                    1.75,
                    0.8,
                    4.3,
                    4.2,
                    4.85,
                    0.0,
                    3.75,
                    0.9,
                    0.0,
                    3.35,
                    1.05,
                    2.2,
                    0.75,
                    3.6,
                    2.0,
                    1.2,
                    1.9,
                    3.45,
                    1.6,
                    3.95,
                    4.45,
                    4.55,
                    4.75,
                    3.7,
                    0.3,
                    2.45,
                    3.75,
                    0.9,
                    2.2,
                ],
                [
                    4.95,
                    1.05,
                    2.4,
                    4.05,
                    3.75,
                    1.95,
                    0.65,
                    4.9,
                    4.3,
                    2.5,
                    1.9,
                    1.75,
                    2.05,
                    3.95,
                    0.8,
                    0.0,
                    0.8,
                    3.45,
                    1.55,
                    0.3,
                    1.5,
                    2.9,
                    2.15,
                    2.15,
                    3.3,
                    3.2,
                    4.3,
                    3.7,
                    0.4,
                    1.7,
                    0.35,
                    1.9,
                    1.8,
                    4.3,
                    4.7,
                    4.05,
                    3.65,
                    1.1,
                    1.0,
                    2.7,
                    3.95,
                    2.3,
                    2.6,
                    3.5,
                    0.75,
                    4.3,
                    3.0,
                    3.85,
                ],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_flatten_backward_1():
    np.testing.assert_allclose(
        flatten_backward(3, 3),
        np.array(
            [[4.2, 1.9, 6.9], [6.2, 4.9, 4.6], [6.6, 0.8, 2.4]],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_flatten_backward_2():
    np.testing.assert_allclose(
        flatten_backward(3, 3, 3),
        np.array(
            [
                [[6.7, 6.5, 5.6], [4.6, 7.5, 7.5], [6.7, 4.9, 4.2]],
                [[3.3, 0.3, 8.3], [5.6, 4.2, 1.0], [5.2, 4.5, 6.5]],
                [[4.8, 9.1, 9.5], [1.5, 7.7, 0.1], [9.4, 3.4, 9.4]],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_flatten_backward_3():
    np.testing.assert_allclose(
        flatten_backward(2, 2, 2, 2),
        np.array(
            [
                [[[6.8, 3.8], [5.4, 5.1]], [[8.5, 4.8], [3.1, 1.0]]],
                [[[9.3, 0.8], [3.4, 1.6]], [[9.4, 3.6], [6.6, 7.0]]],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_flatten_backward_4():
    np.testing.assert_allclose(
        flatten_backward(1, 2, 3, 4),
        np.array(
            [
                [
                    [
                        [8.4, 9.0, 3.8, 9.7],
                        [9.7, 6.6, 5.4, 6.1],
                        [0.6, 7.3, 6.2, 0.2],
                    ],
                    [
                        [9.0, 8.1, 6.1, 0.3],
                        [6.0, 3.3, 9.7, 2.6],
                        [7.9, 5.8, 2.4, 2.0],
                    ],
                ]
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_flatten_backward_5():
    np.testing.assert_allclose(
        flatten_backward(2, 2, 4, 3),
        np.array(
            [
                [
                    [
                        [9.8, 7.1, 5.4],
                        [4.0, 6.2, 5.7],
                        [7.2, 2.0, 2.4],
                        [8.9, 4.9, 3.3],
                    ],
                    [
                        [9.0, 9.8, 5.9],
                        [7.1, 2.7, 9.6],
                        [8.5, 9.3, 5.8],
                        [3.1, 9.0, 6.7],
                    ],
                ],
                [
                    [
                        [7.4, 8.6, 6.9],
                        [8.2, 5.3, 8.7],
                        [8.8, 8.7, 4.0],
                        [3.9, 1.8, 2.7],
                    ],
                    [
                        [5.7, 6.2, 0.0],
                        [6.0, 0.0, 0.3],
                        [2.0, 0.1, 2.7],
                        [2.1, 0.1, 6.7],
                    ],
                ],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def dropout_forward(shape, prob=0.5):
    np.random.seed(3)
    x = get_tensor(*shape)
    f = nn.Dropout(prob)
    return f(x).cached_data


def dropout_backward(shape, prob=0.5):
    np.random.seed(3)
    x = get_tensor(*shape)
    f = nn.Dropout(prob)
    y = f(x).sum()
    y.backward()
    return x.grad.cached_data


def test_nn_dropout_forward_1():
    np.testing.assert_allclose(
        dropout_forward((2, 3), prob=0.45),
        np.array(
            [[6.818182, 0.0, 0.0], [0.18181819, 0.0, 6.090909]],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_dropout_backward_1():
    np.testing.assert_allclose(
        dropout_backward((2, 3), prob=0.26),
        np.array(
            [[1.3513514, 0.0, 0.0], [1.3513514, 0.0, 1.3513514]],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def residual_forward(shape=(5, 5)):
    np.random.seed(42)
    f = nn.Residual(
        nn.Sequential(nn.Linear(*shape), nn.ReLU(), nn.Linear(*shape[::-1]))
    )
    x = get_tensor(*shape[::-1])
    return f(x).cached_data


def residual_backward(shape=(5, 5)):
    np.random.seed(42)
    f = nn.Residual(
        nn.Sequential(nn.Linear(*shape), nn.ReLU(), nn.Linear(*shape[::-1]))
    )
    x = get_tensor(*shape[::-1])
    f(x).sum().backward()
    return x.grad.cached_data


def test_nn_residual_forward_1():
    np.testing.assert_allclose(
        residual_forward(),
        np.array(
            [
                [0.4660964, 3.8619597, -3.637068, 3.7489638, 2.4931884],
                [-3.3769124, 2.5409935, -2.7110925, 4.9782896, -3.005401],
                [-3.0222898, 3.796795, -2.101042, 6.785948, 0.9347453],
                [-2.2496533, 3.635599, -2.1818666, 5.6361046, 0.9748006],
                [-0.03458184, 0.0823682, -0.06686163, 1.9169499, 1.2638961],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_residual_backward_1():
    np.testing.assert_allclose(
        residual_backward(),
        np.array(
            [
                [0.24244219, -0.19571924, -0.08556509, 0.9191598, 1.6787351],
                [0.24244219, -0.19571924, -0.08556509, 0.9191598, 1.6787351],
                [0.24244219, -0.19571924, -0.08556509, 0.9191598, 1.6787351],
                [0.24244219, -0.19571924, -0.08556509, 0.9191598, 1.6787351],
                [0.24244219, -0.19571924, -0.08556509, 0.9191598, 1.6787351],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def batchnorm_forward(*shape, affine=False):
    x = get_tensor(*shape)
    bn = nn.BatchNorm1d(shape[1])
    if affine:
        bn.weight.data = get_tensor(shape[1], entropy=42)
        bn.bias.data = get_tensor(shape[1], entropy=1337)
    return bn(x).cached_data


def batchnorm_backward(*shape, affine=False):
    x = get_tensor(*shape)
    bn = nn.BatchNorm1d(shape[1])
    if affine:
        bn.weight.data = get_tensor(shape[1], entropy=42)
        bn.bias.data = get_tensor(shape[1], entropy=1337)
    y = (bn(x) ** 2).sum()
    y.backward()
    return x.grad.cached_data


def batchnorm_running_mean(*shape, iters=10):
    bn = nn.BatchNorm1d(shape[1])
    for i in range(iters):
        x = get_tensor(*shape, entropy=i)
        _ = bn(x)
    return bn.running_mean.cached_data


def batchnorm_running_var(*shape, iters=10):
    bn = nn.BatchNorm1d(shape[1])
    for i in range(iters):
        x = get_tensor(*shape, entropy=i)
        _ = bn(x)
    return bn.running_var.cached_data


def batchnorm_running_grad(*shape, iters=10):
    bn = nn.BatchNorm1d(shape[1])
    for i in range(iters):
        x = get_tensor(*shape, entropy=i)
        y = bn(x)
    bn.eval()
    (y**2).sum().backward()
    return x.grad.cached_data


def check_training_mode():
    model = nn.Sequential(
        nn.BatchNorm1d(4),
        nn.Sequential(
            nn.LayerNorm1d(4),
            nn.Linear(4, 4),
            nn.Dropout(0.1),
        ),
        nn.Linear(4, 4),
        UselessModule(),
    )

    model_refs = [
        model.modules[0],
        model.modules[1].modules[0],
        model.modules[1].modules[1],
        model.modules[1].modules[2],
        model.modules[2],
        model.modules[3],
        model.modules[3].stuff["layer1"],
        model.modules[3].stuff["layer2"][0],
        model.modules[3].stuff["layer2"][1].modules[0],
    ]

    eval_mode = [1 if not x.training else 0 for x in model_refs]
    model.eval()
    eval_mode.extend([1 if not x.training else 0 for x in model_refs])
    model.train()
    eval_mode.extend([1 if not x.training else 0 for x in model_refs])

    return np.array(eval_mode)


def test_nn_batchnorm_check_model_eval_switches_training_flag_1():
    np.testing.assert_allclose(
        check_training_mode(),
        np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_batchnorm_forward_1():
    np.testing.assert_allclose(
        batchnorm_forward(4, 4),
        np.array(
            [
                [7.8712696e-01, -3.1676728e-01, -6.4885163e-01, 2.0828949e-01],
                [-7.9508079e-03, 1.0092355e00, 1.6221288e00, 8.5209310e-01],
                [8.5073310e-01, -1.4954363e00, -9.6686421e-08, -1.6852506e00],
                [-1.6299094e00, 8.0296844e-01, -9.7327745e-01, 6.2486827e-01],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_batchnorm_forward_affine_1():
    np.testing.assert_allclose(
        batchnorm_forward(4, 4, affine=True),
        np.array(
            [
                [7.49529, 0.047213316, 2.690084, 5.5227957],
                [4.116209, 3.8263211, 7.79979, 7.293256],
                [7.765616, -3.3119934, 4.15, 0.31556034],
                [-2.7771149, 3.23846, 1.9601259, 6.6683874],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


# def test_nn_batchnorm_backward_1():
#     np.testing.assert_allclose(
#         batchnorm_backward(5, 4),
#         np.array(
#             [
#                 [2.1338463e-04, 5.2094460e-06, -2.8359889e-05, -4.4368207e-06],
#                 [
#                     -3.8480759e-04,
#                     -4.0292739e-06,
#                     1.8370152e-05,
#                     -1.1172146e-05,
#                 ],
#                 [2.5629997e-04, -1.1003018e-05, -9.0479853e-06, 5.5171549e-06],
#                 [-4.2676926e-04, 3.4213067e-06, 1.3601780e-05, 1.0166317e-05],
#                 [3.4189224e-04, 6.4015389e-06, 5.4359434e-06, -7.4505806e-08],
#             ],
#             dtype=np.float32,
#         ),
#         rtol=1e-5,
#         atol=1e-5,
#     )


# def test_nn_batchnorm_backward_affine_1():
#     np.testing.assert_allclose(
#         batchnorm_backward(5, 4, affine=True),
#         np.array(
#             [
#                 [3.8604736e-03, 4.2676926e-05, -1.4114380e-04, -3.2424927e-05],
#                 [
#                     -6.9427490e-03,
#                     -3.3140182e-05,
#                     9.1552734e-05,
#                     -8.5830688e-05,
#                 ],
#                 [4.6386719e-03, -8.9883804e-05, -4.5776367e-05, 4.3869019e-05],
#                 [-7.7133179e-03, 2.7418137e-05, 6.6757202e-05, 7.4386597e-05],
#                 [6.1874390e-03, 5.2213669e-05, 2.8610229e-05, -1.9073486e-06],
#             ],
#             dtype=np.float32,
#         ),
#         rtol=1e-5,
#         atol=1e-4,
#     )


def test_nn_batchnorm_running_mean_1():
    np.testing.assert_allclose(
        batchnorm_running_mean(4, 3),
        np.array([2.020656, 1.69489, 1.498846], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_batchnorm_running_var_1():
    np.testing.assert_allclose(
        batchnorm_running_var(4, 3),
        np.array([1.412775, 1.386191, 1.096604], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


# def test_nn_batchnorm_running_grad_1():
#     np.testing.assert_allclose(
#         batchnorm_running_grad(4, 3),
#         np.array(
#             [
#                 [8.7022781e-06, -4.9751252e-06, 9.5367432e-05],
#                 [6.5565109e-06, -7.2401017e-06, -2.3484230e-05],
#                 [-3.5762787e-06, -4.5262277e-07, 1.6093254e-05],
#                 [-1.1682510e-05, 1.2667850e-05, -8.7976456e-05],
#             ],
#             dtype=np.float32,
#         ),
#         rtol=1e-5,
#         atol=1e-5,
#     )


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("input_size", INPUT_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("bias", BIAS)
@pytest.mark.parametrize("init_hidden", INIT_HIDDEN)
@pytest.mark.parametrize("nonlinearity", NONLINEARITIES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_rnn_cell(
    batch_size,
    input_size,
    hidden_size,
    bias,
    init_hidden,
    nonlinearity,
    device,
):
    x = np.random.randn(batch_size, input_size).astype(np.float32)
    h0 = np.random.randn(batch_size, hidden_size).astype(np.float32)

    model_ = torch.nn.RNNCell(
        input_size, hidden_size, nonlinearity=nonlinearity, bias=bias
    )
    if init_hidden:
        h_ = model_(torch.tensor(x), torch.tensor(h0))
    else:
        h_ = model_(torch.tensor(x), None)

    model = nn.RNNCell(
        input_size,
        hidden_size,
        device=device,
        bias=bias,
        nonlinearity=nonlinearity,
    )
    model.W_ih = Tensor(
        model_.weight_ih.detach().numpy().transpose(), device=device
    )
    model.W_hh = Tensor(
        model_.weight_hh.detach().numpy().transpose(), device=device
    )
    if bias:
        model.bias_ih = Tensor(model_.bias_ih.detach().numpy(), device=device)
        model.bias_hh = Tensor(model_.bias_hh.detach().numpy(), device=device)
    if init_hidden:
        h = model(Tensor(x, device=device), Tensor(h0, device=device))
    else:
        h = model(Tensor(x, device=device), None)
    assert h.device == device
    np.testing.assert_allclose(
        h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5
    )
    h.sum().backward()
    h_.sum().backward()
    np.testing.assert_allclose(
        model_.weight_ih.grad.detach().numpy().transpose(),
        model.W_ih.grad.numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("input_size", INPUT_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("bias", BIAS)
@pytest.mark.parametrize("init_hidden", INIT_HIDDEN)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_lstm_cell(
    batch_size, input_size, hidden_size, bias, init_hidden, device
):
    x = np.random.randn(batch_size, input_size).astype(np.float32)
    h0 = np.random.randn(batch_size, hidden_size).astype(np.float32)
    c0 = np.random.randn(batch_size, hidden_size).astype(np.float32)

    model_ = torch.nn.LSTMCell(input_size, hidden_size, bias=bias)
    if init_hidden:
        h_, c_ = model_(torch.tensor(x), (torch.tensor(h0), torch.tensor(c0)))
    else:
        h_, c_ = model_(torch.tensor(x), None)

    model = nn.LSTMCell(input_size, hidden_size, device=device, bias=bias)

    model.W_ih = Tensor(
        model_.weight_ih.detach().numpy().transpose(), device=device
    )
    model.W_hh = Tensor(
        model_.weight_hh.detach().numpy().transpose(), device=device
    )
    if bias:
        model.bias_ih = Tensor(model_.bias_ih.detach().numpy(), device=device)
        model.bias_hh = Tensor(model_.bias_hh.detach().numpy(), device=device)

    if init_hidden:
        h, c = model(
            Tensor(x, device=device),
            (Tensor(h0, device=device), Tensor(c0, device=device)),
        )
    else:
        h, c = model(Tensor(x, device=device), None)
    np.testing.assert_allclose(
        h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        c_.detach().numpy(), c.numpy(), atol=1e-5, rtol=1e-5
    )

    h.sum().backward()
    h_.sum().backward()
    np.testing.assert_allclose(
        model_.weight_ih.grad.detach().numpy().transpose(),
        model.W_ih.grad.numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


op_conv_shapes = [
    ((3, 14, 14, 8), (3, 3, 8, 16), 1, 0),
    ((3, 14, 14, 8), (3, 3, 8, 16), 1, 1),
    ((3, 16, 16, 8), (3, 3, 8, 16), 1, 2),
    ((3, 16, 16, 8), (3, 3, 8, 14), 1, 0),
    ((3, 16, 16, 2), (3, 3, 2, 14), 1, 0),
    ((3, 14, 14, 8), (3, 3, 8, 16), 2, 0),
    ((3, 14, 14, 8), (3, 3, 8, 16), 2, 1),
    ((3, 16, 16, 8), (3, 3, 8, 16), 2, 2),
    ((3, 16, 16, 8), (3, 3, 8, 14), 2, 0),
    ((3, 16, 16, 2), (3, 3, 2, 14), 2, 0),
    ((3, 16, 16, 24), (3, 3, 24, 14), 1, 0),
    ((3, 14, 14, 8), (5, 5, 8, 16), 1, 0),
    ((3, 17, 17, 8), (5, 5, 8, 16), 1, 0),
    ((3, 17, 17, 1), (5, 5, 1, 16), 1, 0),
    ((3, 17, 17, 16), (5, 5, 16, 1), 1, 0),
    ((3, 17, 17, 16), (1, 1, 16, 1), 1, 0),
    ((1, 14, 14, 2), (3, 3, 2, 2), 1, 0),
]


@pytest.mark.parametrize("Z_shape, W_shape, stride, padding", op_conv_shapes)
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    "backward", [True, False], ids=["backward", "forward"]
)
def test_op_conv(Z_shape, W_shape, stride, padding, backward, device):
    np.random.seed(0)

    _Z = np.random.randn(*Z_shape) * 5
    _Z = _Z.astype(np.float32)
    _W = np.random.randn(*W_shape) * 5
    _W = _W.astype(np.float32)
    Z = Tensor(_Z, device=device)
    W = Tensor(_W, device=device)
    y = ops.conv(Z, W, padding=padding, stride=stride)
    y2 = y.sum()
    if backward:
        y2.backward()
    Ztch = torch.Tensor(_Z).float()
    Ztch.requires_grad = True
    Wtch = torch.Tensor(_W).float()
    Wtch.requires_grad = True
    out = torch.nn.functional.conv2d(
        Ztch.permute(0, 3, 1, 2),
        Wtch.permute(3, 2, 0, 1),
        padding=padding,
        stride=stride,
    )
    out2 = out.sum()
    if backward:
        out2.backward()
    if backward:
        err1 = np.linalg.norm(Ztch.grad.numpy() - Z.grad.numpy())
        err2 = np.linalg.norm(Wtch.grad.numpy() - W.grad.numpy())
    err3 = np.linalg.norm(out2.detach().numpy() - y2.numpy())
    if backward:
        assert err1 < 1e-2, "input grads match"
        assert err2 < 1e-2, "weight grads match"
    assert err3 < 1e-1, f"outputs match {y2}, {out2}"
