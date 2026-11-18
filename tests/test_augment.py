import numpy as np

from src.features.augment import speed_perturb, add_noise


def test_speed_perturb_changes_length():
    a = np.sin(np.linspace(0, 10, 1000)).astype("float32")
    assert len(speed_perturb(a, 1.1)) < len(a)
    assert len(speed_perturb(a, 0.9)) > len(a)


def test_add_noise_changes_signal_deterministically():
    a = np.ones(500, dtype="float32")
    n1 = add_noise(a, 20, np.random.default_rng(0))
    n2 = add_noise(a, 20, np.random.default_rng(0))
    assert np.allclose(n1, n2)
    assert not np.allclose(n1, a)
    assert n1.shape == a.shape
