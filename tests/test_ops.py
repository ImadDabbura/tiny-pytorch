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
