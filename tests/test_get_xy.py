import pandas as pd
from src import config
from src.data import feature_columns, get_xy


def _toy():
    return pd.DataFrame({
        "path": ["p1", "p2", "p3", "p4"],
        "modality": ["audio-only"] * 4,
        "vocal_channel": ["speech", "speech", "song", "speech"],
        "emotion": ["happy", "sad", "happy", "angry"],
        "intensity": ["normal"] * 4,
        "statement": ["kids"] * 4,
        "repetition": [1, 1, 1, 2],
        "actor": [1, 2, 1, 3],
        "gender": ["male", "female", "male", "male"],
        "split": ["train", "val", "train", "train"],
        "feat_a": [0.1, 0.2, 0.3, 0.4],
        "feat_b": [1.0, 2.0, 3.0, 4.0],
    })


def test_feature_columns_excludes_meta():
    assert feature_columns(_toy()) == ["feat_a", "feat_b"]


def test_get_xy_speech_default():
    X, y, g = get_xy(_toy())
    assert list(X.columns) == ["feat_a", "feat_b"]
    assert len(X) == 3
    assert y.tolist() == ["happy", "sad", "angry"]
    assert g.tolist() == [1, 2, 3]
    assert not (set(config.META_COLS) & set(X.columns))


def test_get_xy_split_filter():
    X, y, g = get_xy(_toy(), vocal_channel="speech", split="train")
    assert len(X) == 2
    assert y.tolist() == ["happy", "angry"]


def test_get_xy_all_channels():
    X, y, g = get_xy(_toy(), vocal_channel=None)
    assert len(X) == 4
