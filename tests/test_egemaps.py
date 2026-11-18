from src.data import build_metadata
from src.features.egemaps import extract_egemaps


def _sample_metadata(n=3):
    return build_metadata().head(n).reset_index(drop=True)


def test_extract_egemaps_shape_and_index():
    md = _sample_metadata(3)
    feats = extract_egemaps(md)
    assert feats.shape == (3, 88)
    assert list(feats.index) == md["path"].tolist()
    assert feats.isna().sum().sum() == 0


def test_extract_egemaps_deterministic():
    md = _sample_metadata(2)
    a = extract_egemaps(md)
    b = extract_egemaps(md)
    assert a.equals(b)
