import numpy as np
from src.data import build_metadata
from src.features.wav2vec import extract_wav2vec


def _sample(n=2):
    return build_metadata().head(n).reset_index(drop=True)


def test_extract_wav2vec_shape_and_index():
    md = _sample(2)
    feats = extract_wav2vec(md, device="cpu")
    assert feats.shape == (2, 768)
    assert list(feats.index) == md["path"].tolist()
    assert feats.isna().sum().sum() == 0


def test_extract_wav2vec_deterministic():
    md = _sample(2)
    a = extract_wav2vec(md, device="cpu")
    b = extract_wav2vec(md, device="cpu")
    assert np.allclose(a.values, b.values, atol=1e-5)
