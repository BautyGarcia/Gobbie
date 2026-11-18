import numpy as np
import pandas as pd

from src.normalize import cmvn_by_speaker, cmvn_frames_by_speaker


def _toy():
    X = pd.DataFrame({
        "a": [1.0, 3.0, 5.0, 10.0, 20.0, 30.0],
        "b": [-2.0, 0.0, 2.0, 1.0, 1.0, 4.0],
        "c": [7.0, 7.0, 7.0, 9.0, 9.0, 9.0],
    })
    groups = pd.Series([1, 1, 1, 2, 2, 2])
    return X, groups


def test_cmvn_centers_and_scales_per_speaker():
    X, groups = _toy()
    Z = cmvn_by_speaker(X, groups)
    for _, g in Z.groupby(groups):
        assert np.allclose(g[["a", "b"]].mean(), 0.0, atol=1e-9)
        assert np.allclose(g[["a", "b"]].std(ddof=0), 1.0, atol=1e-9)


def test_cmvn_preserves_shape_index_columns():
    X, groups = _toy()
    Z = cmvn_by_speaker(X, groups)
    assert Z.shape == X.shape
    assert list(Z.columns) == list(X.columns)
    assert list(Z.index) == list(X.index)


def test_cmvn_constant_feature_is_safe():
    X, groups = _toy()
    Z = cmvn_by_speaker(X, groups)
    assert np.isfinite(Z.to_numpy()).all()
    assert np.allclose(Z["c"], 0.0)


def test_cmvn_frames_centers_per_speaker():
    rng = np.random.default_rng(0)
    seqs = [rng.normal(5, 2, size=(7, 3)), rng.normal(5, 2, size=(4, 3)),
            rng.normal(-3, 1, size=(6, 3)), rng.normal(-3, 1, size=(9, 3))]
    groups = [1, 1, 2, 2]
    out = cmvn_frames_by_speaker(seqs, groups)
    assert [s.shape for s in out] == [s.shape for s in seqs]
    for spk in (1, 2):
        idx = [i for i, g in enumerate(groups) if g == spk]
        stacked = np.concatenate([out[i] for i in idx], axis=0)
        assert np.allclose(stacked.mean(axis=0), 0.0, atol=1e-5)
        assert np.allclose(stacked.std(axis=0), 1.0, atol=1e-5)
