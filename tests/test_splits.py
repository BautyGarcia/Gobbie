import json

from src.data import build_metadata
from src.splits import make_split, load_split, assign_split


def test_make_split_speaker_independent(tmp_path):
    md = build_metadata()
    out = tmp_path / "split.json"
    split = make_split(md, val_actor_fraction=0.2, seed=42, out_path=out)

    train, val = set(split["train_actors"]), set(split["val_actors"])
    assert train.isdisjoint(val)
    assert train | val == set(md["actor"].unique())
    val_genders = md[md["actor"].isin(val)][["actor", "gender"]].drop_duplicates()["gender"]
    assert (val_genders == "male").sum() == 2
    assert (val_genders == "female").sum() == 2
    assert json.loads(out.read_text())["val_actors"] == split["val_actors"]


def test_make_split_deterministic(tmp_path):
    md = build_metadata()
    a = make_split(md, seed=42, out_path=tmp_path / "a.json")
    b = make_split(md, seed=42, out_path=tmp_path / "b.json")
    assert a["val_actors"] == b["val_actors"]


def test_assign_split_adds_column_both_channels(tmp_path):
    md = build_metadata()
    split = make_split(md, seed=42, out_path=tmp_path / "s.json")
    labeled = assign_split(md, split)
    assert set(labeled["split"].unique()) == {"train", "val"}
    val_rows = labeled[labeled["actor"].isin(split["val_actors"])]
    assert (val_rows["split"] == "val").all()
    assert set(val_rows["vocal_channel"].unique()) <= {"speech", "song"}
