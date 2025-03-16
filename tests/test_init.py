import numpy as np

from tiny_pytorch import init


def test_init_kaiming_uniform():
    np.random.seed(42)
    np.testing.assert_allclose(
        init.kaiming_uniform(3, 5).numpy(),
        np.array(
            [
                [-0.35485414, 1.2748126, 0.65617794, 0.27904832, -0.9729262],
                [-0.97299445, -1.2499284, 1.0357026, 0.28599644, 0.58851814],
                [-1.3559918, 1.3291057, 0.9402898, -0.81362784, -0.8999349],
            ],
            dtype=np.float32,
        ),
        rtol=1e-4,
        atol=1e-4,
    )


def test_init_kaiming_normal():
    np.random.seed(42)
    np.testing.assert_allclose(
        init.kaiming_normal(3, 5).numpy(),
        np.array(
            [
                [0.4055654, -0.11289233, 0.5288355, 1.2435486, -0.19118543],
                [-0.19117202, 1.2894219, 0.62660784, -0.38332424, 0.4429984],
                [-0.37837896, -0.38026676, 0.19756137, -1.5621868, -1.4083896],
            ],
            dtype=np.float32,
        ),
        rtol=1e-4,
        atol=1e-4,
    )


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
