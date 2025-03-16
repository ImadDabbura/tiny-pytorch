import numpy as np

from tiny_pytorch import init


def test_init_xavier_uniform():
    np.random.seed(42)
    np.testing.assert_allclose(
        init.xavier_uniform(3, 5, gain=1.5).numpy(),
        np.array(
            [
                [-0.32595432, 1.1709901, 0.60273796, 0.25632226, -0.8936898],
                [-0.89375246, -1.1481324, 0.95135355, 0.26270452, 0.54058844],
                [-1.245558, 1.2208616, 0.8637113, -0.74736494, -0.826643],
            ],
            dtype=np.float32,
        ),
        rtol=1e-4,
        atol=1e-4,
    )


def test_init_xavier_normal():
    np.random.seed(42)
    np.testing.assert_allclose(
        init.xavier_normal(3, 5, gain=0.33).numpy(),
        np.array(
            [
                [
                    0.08195783,
                    -0.022813609,
                    0.10686861,
                    0.25129992,
                    -0.038635306,
                ],
                [-0.038632598, 0.2605701, 0.12662673, -0.07746328, 0.08952241],
                [
                    -0.07646392,
                    -0.07684541,
                    0.039923776,
                    -0.31569123,
                    -0.28461143,
                ],
            ],
            dtype=np.float32,
        ),
        rtol=1e-4,
        atol=1e-4,
    )
