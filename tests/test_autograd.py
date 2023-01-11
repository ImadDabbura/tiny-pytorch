import numpy as np

from tiny_pytorch import Tensor
from tiny_pytorch.tensor import find_topo_sort


class TestTopoSort:
    def test_1(self):
        a1, b1 = Tensor(np.asarray([[0.88282157]])), Tensor(
            np.asarray([[0.90170084]])
        )
        c1 = 3 * a1 * a1 + 4 * b1 * a1 - a1

        soln = np.array(
            [
                np.array([[0.88282157]]),
                np.array([[2.64846471]]),
                np.array([[2.33812177]]),
                np.array([[0.90170084]]),
                np.array([[3.60680336]]),
                np.array([[3.1841638]]),
                np.array([[5.52228558]]),
                np.array([[-0.88282157]]),
                np.array([[4.63946401]]),
            ]
        )

        topo_order = np.array([x.numpy() for x in find_topo_sort([c1])])

        assert len(soln) == len(topo_order)
        np.testing.assert_allclose(topo_order, soln, rtol=1e-06, atol=1e-06)

    def test_2(self):
        a1, b1 = Tensor(np.asarray([[0.20914675], [0.65264178]])), Tensor(
            np.asarray([[0.65394286, 0.08218317]])
        )
        c1 = 3 * ((b1 @ a1) + (2.3412 * b1) @ a1) + 1.5

        soln = [
            np.array([[0.65394286, 0.08218317]]),
            np.array([[0.20914675], [0.65264178]]),
            np.array([[0.19040619]]),
            np.array([[1.53101102, 0.19240724]]),
            np.array([[0.44577898]]),
            np.array([[0.63618518]]),
            np.array([[1.90855553]]),
            np.array([[3.40855553]]),
        ]

        topo_order = [x.numpy() for x in find_topo_sort([c1])]

        assert len(soln) == len(topo_order)
        # step through list as entries differ in length
        for t, s in zip(topo_order, soln):
            np.testing.assert_allclose(t, s, rtol=1e-06, atol=1e-06)

    def test_3(self):
        a = Tensor(
            np.asarray([[1.4335016, 0.30559972], [0.08130171, -1.15072371]])
        )
        b = Tensor(
            np.asarray([[1.34571691, -0.95584433], [-0.99428573, -0.04017499]])
        )
        e = (a @ b + b - a) @ a

        topo_order = np.array([x.numpy() for x in find_topo_sort([e])])

        soln = np.array(
            [
                np.array([[1.4335016, 0.30559972], [0.08130171, -1.15072371]]),
                np.array(
                    [[1.34571691, -0.95584433], [-0.99428573, -0.04017499]]
                ),
                np.array(
                    [[1.6252339, -1.38248184], [1.25355725, -0.03148146]]
                ),
                np.array(
                    [[2.97095081, -2.33832617], [0.25927152, -0.07165645]]
                ),
                np.array(
                    [[-1.4335016, -0.30559972], [-0.08130171, 1.15072371]]
                ),
                np.array(
                    [[1.53744921, -2.64392589], [0.17796981, 1.07906726]]
                ),
                np.array(
                    [[1.98898021, 3.51227226], [0.34285002, -1.18732075]]
                ),
            ]
        )

        assert len(soln) == len(topo_order)
        np.testing.assert_allclose(topo_order, soln, rtol=1e-06, atol=1e-06)
