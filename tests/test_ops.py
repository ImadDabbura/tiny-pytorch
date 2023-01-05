import numpy as np

from tiny_pytorch import Tensor, ops


class TestForward:
    def test_scalar_add(self):
        np.testing.assert_allclose(
            (Tensor([1, 2]) + 2).numpy(), np.array([3, 4])
        )
        np.testing.assert_allclose(
            (Tensor([[3.0, 2.0]]) + 1).numpy(), np.array([[4.0, 3.0]])
        )

    def test_ewise_add(self):
        np.testing.assert_allclose(
            (Tensor([1, 2]) + Tensor([1, 2])).numpy(), np.array([2, 4])
        )
        np.testing.assert_allclose(
            (Tensor([[3.0, 2.0]]) + Tensor([[0.0, 9.0]])).numpy(),
            np.array([[3.0, 11.0]]),
        )

    def test_scalar_mul(self):
        np.testing.assert_allclose(
            (Tensor([1, 2]) * 5).numpy(), np.array([5, 10])
        )
        np.testing.assert_allclose(
            (Tensor([[3.0, 2.0]]) * 7).numpy(), np.array([[21.0, 14.0]])
        )

    def test_ewise_mul(self):
        np.testing.assert_allclose(
            (Tensor([10, 22]) * Tensor([1, 2])).numpy(), np.array([10, 44])
        )
        np.testing.assert_allclose(
            (Tensor([[3.0, 2.0]]) * Tensor([[0.0, 9.0]])).numpy(),
            np.array([[0.0, 18.0]]),
        )

    def test_negate(self):
        np.testing.assert_equal((-Tensor([1, 2])).numpy(), np.array([-1, -2]))
        np.testing.assert_equal(
            (-Tensor([[-3.0, 10.0]])).numpy(), np.array([[3.0, -10.0]])
        )

    def test_scalar_power(self):
        np.testing.assert_allclose(
            (Tensor([1, 2]) ** 2).numpy(), np.array([1, 4])
        )
        np.testing.assert_allclose(
            (Tensor([[3.0, 2.0]]) ** 0).numpy(), np.array([[1.0, 1.0]])
        )

    def test_ewise_power(self):
        np.testing.assert_allclose(
            (Tensor([1, 2]) ** Tensor([1, 2])).numpy(), np.array([1, 4])
        )
        np.testing.assert_allclose(
            (Tensor([[3.0, 2.0]]) ** Tensor([[0.0, 9.0]])).numpy(),
            np.array([[1.0, 512.0]]),
        )

    def test_scalar_divide(self):
        np.testing.assert_allclose(
            (Tensor([10, 2]) / 2).numpy(), np.array([5, 1])
        )
        np.testing.assert_allclose(
            (Tensor([[3.0, 2.0]]) / 7).numpy(),
            np.array([[0.42857142857142855, 0.2857142857142857]]),
        )

    def test_ewise_divide(self):
        np.testing.assert_allclose(
            (Tensor([10, 22]) / Tensor([1, 2])).numpy(), np.array([10.0, 11.0])
        )
        np.testing.assert_allclose(
            (Tensor([[3.0, 2.0]]) / Tensor([[2.0, 9.0]])).numpy(),
            np.array([[1.5, 0.222222222]]),
        )

    def test_reshape_forward(self):
        np.testing.assert_allclose(
            Tensor(
                [
                    [2.9, 2.0, 2.4],
                    [3.95, 3.95, 4.65],
                    [2.1, 2.5, 2.7],
                    [1.9, 4.85, 3.25],
                    [3.35, 3.45, 3.45],
                ]
            )
            .reshape(shape=(15,))
            .numpy(),
            np.array(
                [
                    2.9,
                    2.0,
                    2.4,
                    3.95,
                    3.95,
                    4.65,
                    2.1,
                    2.5,
                    2.7,
                    1.9,
                    4.85,
                    3.25,
                    3.35,
                    3.45,
                    3.45,
                ]
            ),
        )
        np.testing.assert_allclose(
            Tensor(
                [
                    [[4.1, 4.05, 1.35, 1.65], [3.65, 0.9, 0.65, 4.15]],
                    [[4.7, 1.4, 2.55, 4.8], [2.8, 1.75, 2.8, 0.6]],
                    [[3.75, 0.6, 0.0, 3.5], [0.15, 1.9, 4.75, 2.8]],
                ]
            )
            .reshape(shape=(2, 3, 4))
            .numpy(),
            np.array(
                [
                    [
                        [4.1, 4.05, 1.35, 1.65],
                        [3.65, 0.9, 0.65, 4.15],
                        [4.7, 1.4, 2.55, 4.8],
                    ],
                    [
                        [2.8, 1.75, 2.8, 0.6],
                        [3.75, 0.6, 0.0, 3.5],
                        [0.15, 1.9, 4.75, 2.8],
                    ],
                ]
            ),
        )

    def test_log_forward(self):
        np.testing.assert_allclose(
            ops.Log()(Tensor([50])).numpy(), np.array([3.912023])
        )

    def test_exp_forward(self):
        np.testing.assert_allclose(
            ops.Exp()(Tensor([15])).numpy(), np.array([3269017.37247])
        )

    def test_relu_forward(self):
        np.testing.assert_allclose(
            ops.ReLU()(Tensor([-19, 0, 29])).numpy(), np.array([0, 0, 29])
        )

    def test_broadcast_to_forward(self):
        np.testing.assert_allclose(
            Tensor([[1.85, 0.85, 0.6]]).broadcast_to(shape=(3, 3, 3)).numpy(),
            np.array(
                [
                    [[1.85, 0.85, 0.6], [1.85, 0.85, 0.6], [1.85, 0.85, 0.6]],
                    [[1.85, 0.85, 0.6], [1.85, 0.85, 0.6], [1.85, 0.85, 0.6]],
                    [[1.85, 0.85, 0.6], [1.85, 0.85, 0.6], [1.85, 0.85, 0.6]],
                ]
            ),
        )

    def test_summation_forward(self):
        np.testing.assert_allclose(
            Tensor(
                [
                    [2.2, 4.35, 1.4, 0.3, 2.65],
                    [1.0, 0.85, 2.75, 3.8, 1.55],
                    [3.2, 2.3, 3.45, 0.7, 0.0],
                ]
            )
            .sum()
            .numpy(),
            np.array(30.5),
        )
        np.testing.assert_allclose(
            Tensor(
                [
                    [1.05, 2.55, 1.0],
                    [2.95, 3.7, 2.6],
                    [0.1, 4.1, 3.3],
                    [1.1, 3.4, 3.4],
                    [1.8, 4.55, 2.3],
                ]
            )
            .sum(axes=1)
            .numpy(),
            np.array([4.6, 9.25, 7.5, 7.9, 8.65]),
        )
        np.testing.assert_allclose(
            Tensor([[1.5, 3.85, 3.45], [1.35, 1.3, 0.65], [2.6, 4.55, 0.25]])
            .sum(axes=0)
            .numpy(),
            np.array([5.45, 9.7, 4.35]),
        )

    def test_transpose_forward(self):
        np.testing.assert_allclose(
            Tensor([[[1.95]], [[2.7]], [[3.75]]])
            .transpose(axes=(1, 2))
            .numpy(),
            np.array([[[1.95]], [[2.7]], [[3.75]]]),
        )
        np.testing.assert_allclose(
            Tensor([[[[0.95]]], [[[2.55]]], [[[0.45]]]])
            .transpose(axes=(2, 3))
            .numpy(),
            np.array([[[[0.95]]], [[[2.55]]], [[[0.45]]]]),
        )
        np.testing.assert_allclose(
            Tensor(
                [
                    [
                        [[0.4, 0.05], [2.95, 1.3]],
                        [[4.8, 1.2], [1.65, 3.1]],
                    ],
                    [
                        [[1.45, 3.05], [2.25, 0.1]],
                        [[0.45, 4.75], [1.5, 1.8]],
                    ],
                    [
                        [[1.5, 4.65], [1.35, 2.7]],
                        [[2.0, 1.65], [2.05, 1.2]],
                    ],
                ]
            )
            .transpose()
            .numpy(),
            np.array(
                [
                    [[[0.4, 2.95], [0.05, 1.3]], [[4.8, 1.65], [1.2, 3.1]]],
                    [[[1.45, 2.25], [3.05, 0.1]], [[0.45, 1.5], [4.75, 1.8]]],
                    [[[1.5, 1.35], [4.65, 2.7]], [[2.0, 2.05], [1.65, 1.2]]],
                ]
            ),
        )
        np.testing.assert_allclose(
            Tensor([[[2.45]], [[3.5]], [[0.9]]])
            .transpose(axes=(0, 1))
            .numpy(),
            np.array([[[2.45], [3.5], [0.9]]]),
        )
        np.testing.assert_allclose(
            Tensor([[4.4, 2.05], [1.85, 2.25], [0.15, 1.4]])
            .transpose()
            .numpy(),
            np.array([[4.4, 1.85, 0.15], [2.05, 2.25, 1.4]]),
        )
        np.testing.assert_allclose(
            Tensor([[0.05, 3.7, 1.35], [4.45, 3.25, 1.95], [2.45, 4.4, 4.5]])
            .transpose()
            .numpy(),
            np.array(
                [[0.05, 4.45, 2.45], [3.7, 3.25, 4.4], [1.35, 1.95, 4.5]]
            ),
        )
        np.testing.assert_allclose(
            Tensor(
                [
                    [[0.55, 1.8, 0.2], [0.8, 2.75, 3.7], [0.95, 1.4, 0.8]],
                    [
                        [0.75, 1.6, 1.35],
                        [3.75, 4.0, 4.55],
                        [1.85, 2.5, 4.8],
                    ],
                    [
                        [0.2, 3.35, 3.4],
                        [0.3, 4.85, 4.85],
                        [4.35, 4.25, 3.05],
                    ],
                ]
            )
            .transpose(axes=(0, 1))
            .numpy(),
            np.array(
                [
                    [[0.55, 1.8, 0.2], [0.75, 1.6, 1.35], [0.2, 3.35, 3.4]],
                    [[0.8, 2.75, 3.7], [3.75, 4.0, 4.55], [0.3, 4.85, 4.85]],
                    [[0.95, 1.4, 0.8], [1.85, 2.5, 4.8], [4.35, 4.25, 3.05]],
                ]
            ),
        )
