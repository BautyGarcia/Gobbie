import numpy as np
from src.data import build_metadata
from src.features.wav2vec import extract_wav2vec_layers


def _sample(n=2):
    return build_metadata().head(n).reset_index(drop=True)


def test_extract_wav2vec_layers_shape():
    md = _sample(2)
    emb, paths = extract_wav2vec_layers(md, device="cpu")
    assert emb.shape == (2, 13, 768)        # 13 = embedding output + 12 capas
    assert paths == md["path"].tolist()
    assert not np.isnan(emb).any()


def test_extract_wav2vec_layers_deterministic():
    md = _sample(2)
    a, _ = extract_wav2vec_layers(md, device="cpu")
    b, _ = extract_wav2vec_layers(md, device="cpu")
    assert np.allclose(a, b, atol=1e-5)
