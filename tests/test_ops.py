import numpy as np

from tiny_pytorch.tensor import Tensor


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
