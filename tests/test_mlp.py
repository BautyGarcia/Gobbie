import numpy as np

from src.models.mlp import MLPClassifier


def _toy(n=240, d=12, seed=0):
    rng = np.random.default_rng(seed)
    classes = np.array(["a", "b", "c"])
    y = classes[rng.integers(0, 3, n)]
    shift = {"a": -3.0, "b": 0.0, "c": 3.0}
    X = rng.normal(size=(n, d)).astype("float32") + np.array([[shift[c]] for c in y])
    return X, y


def test_mlp_fit_predict_shapes_and_labels():
    X, y = _toy()
    clf = MLPClassifier(hidden=(16,), batch_size=16, max_epochs=30, patience=5, device="cpu").fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == (len(y),)
    assert set(pred).issubset(set(clf.classes_))
    assert len(clf.history_) > 0


def test_mlp_learns_separable_data():
    X, y = _toy()
    clf = MLPClassifier(hidden=(32,), batch_size=16, max_epochs=200, patience=40, device="cpu").fit(X, y)
    acc = (clf.predict(X) == y).mean()
    assert acc > 0.85
