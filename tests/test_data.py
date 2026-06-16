from pathlib import Path
from src import config
from src.data import parse_filename, build_metadata


def test_parse_filename_speech_angry_male():
    row = parse_filename(Path("03-01-05-02-01-01-21.wav"))
    assert row["modality"] == "audio-only"
    assert row["vocal_channel"] == "speech"
    assert row["emotion"] == "angry"
    assert row["intensity"] == "strong"
    assert row["statement"] == "kids"
    assert row["repetition"] == 1
    assert row["actor"] == 21
    assert row["gender"] == "male"


def test_parse_filename_song_female():
    row = parse_filename(Path("03-02-03-01-02-02-22.wav"))
    assert row["vocal_channel"] == "song"
    assert row["emotion"] == "happy"
    assert row["actor"] == 22
    assert row["gender"] == "female"


def test_build_metadata_counts():
    md = build_metadata()
    assert len(md) == 2452  # 1440 speech + 1012 song
    assert (md["vocal_channel"] == "speech").sum() == 1440
    assert (md["vocal_channel"] == "song").sum() == 1012
    assert set(md["emotion"].unique()) <= set(config.EMOTION_MAP.values())
    assert md["actor"].between(1, 24).all()
    assert {"path", "emotion", "actor", "gender", "vocal_channel"} <= set(md.columns)
