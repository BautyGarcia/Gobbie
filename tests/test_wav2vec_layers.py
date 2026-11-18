import numpy as np
from src.data import build_metadata
from src.features.wav2vec import extract_wav2vec_layers, extract_wav2vec_frames, layers_to_dataframe


def _sample(n=2):
    return build_metadata().head(n).reset_index(drop=True)


def test_extract_wav2vec_layers_shape():
    md = _sample(2)
    emb, paths = extract_wav2vec_layers(md, device="cpu")
    assert emb.shape == (2, 13, 768)
    assert paths == md["path"].tolist()
    assert not np.isnan(emb).any()


def test_extract_wav2vec_layers_deterministic():
    md = _sample(2)
    a, _ = extract_wav2vec_layers(md, device="cpu")
    b, _ = extract_wav2vec_layers(md, device="cpu")
    assert np.allclose(a, b, atol=1e-5)


def test_extract_wav2vec_frames():
    md = _sample(2)
    seqs, paths = extract_wav2vec_frames(md, device="cpu")
    assert len(seqs) == 2
    assert paths == md["path"].tolist()
    assert all(s.ndim == 2 and s.shape[1] == 768 for s in seqs)
    assert all(s.shape[0] > 1 for s in seqs)
    seqs3, _ = extract_wav2vec_frames(md, device="cpu", layers=(8, 6, 7))
    assert all(s.shape[1] == 768 * 3 for s in seqs3)


def test_layers_to_dataframe():
    emb = np.arange(2 * 13 * 768, dtype="float32").reshape(2, 13, 768)
    paths = ["a.wav", "b.wav"]
    layers = [8, 6, 7]
    df = layers_to_dataframe(emb, paths, layers)
    assert df.shape == (2, len(layers) * 768)
    assert list(df.index) == paths
    assert len(set(df.columns)) == df.shape[1]
    assert np.allclose(df.iloc[0, :768].to_numpy(), emb[0, 8, :])
