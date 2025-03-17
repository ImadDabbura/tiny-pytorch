import numpy as np

import tiny_pytorch.nn as nn
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
