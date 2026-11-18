import numpy as np

from src.models.crnn import CRNNClassifier


def _toy(n=48, feat=8, seed=0):
    rng = np.random.default_rng(seed)
    classes = np.array(["a", "b"])
    seqs, y = [], []
    for i in range(n):
        c = classes[i % 2]
        T = int(rng.integers(10, 20))
        base = 2.0 if c == "a" else -2.0
        seqs.append((rng.normal(size=(T, feat)) + base).astype("float32"))
        y.append(c)
    return seqs, np.array(y)


def test_crnn_fit_predict_shapes_and_labels():
    seqs, y = _toy()
    clf = CRNNClassifier(conv_channels=8, rnn_hidden=8, batch_size=8,
                         max_epochs=15, patience=5, device="cpu").fit(seqs, y)
    pred = clf.predict(seqs)
    assert pred.shape == (len(y),)
    assert set(pred).issubset(set(clf.classes_))


def test_crnn_learns_separable_sequences():
    seqs, y = _toy()
    clf = CRNNClassifier(conv_channels=16, rnn_hidden=16, batch_size=8,
                         max_epochs=60, patience=20, device="cpu").fit(seqs, y)
    acc = (clf.predict(seqs) == y).mean()
    assert acc > 0.85
