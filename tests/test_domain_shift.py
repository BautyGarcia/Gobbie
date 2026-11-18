import pytest

from src.domain_shift import build_emodb_metadata, parse_emodb_filename, SHARED_EMOTIONS


def test_parse_emodb_filename_maps_emotion_and_actor(tmp_path):
    p = tmp_path / "16a02Lb.wav"
    p.touch()
    row = parse_emodb_filename(p)
    assert row == {"path": str(p), "actor": "16", "emotion": "boredom"}


def test_parse_emodb_filename_rejects_unexpected_name(tmp_path):
    p = tmp_path / "not_an_emodb_file.wav"
    p.touch()
    with pytest.raises(ValueError):
        parse_emodb_filename(p)


def test_build_emodb_metadata_reads_all_wavs(tmp_path):
    for name in ["03a01Fa.wav", "03a01Na.wav", "08b02Wc.wav"]:
        (tmp_path / name).touch()
    md = build_emodb_metadata(tmp_path)
    assert len(md) == 3
    assert set(md.columns) == {"path", "actor", "emotion"}
    assert set(md["emotion"]) == {"happy", "neutral", "angry"}


def test_shared_emotions_excludes_boredom_and_is_subset_of_ravdess():
    from src.config import EMOTION_MAP
    assert "boredom" not in SHARED_EMOTIONS
    assert set(SHARED_EMOTIONS) <= set(EMOTION_MAP.values())
