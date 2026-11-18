import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from src.evaluation import grouped_cv, grouped_cv_proba


def test_grouped_cv_covers_all_no_leak_and_perfect():
    n = 60
    groups = pd.Series(np.repeat(np.arange(6), 10))
    y = pd.Series(["pos" if i % 2 == 0 else "neg" for i in range(n)])
    X = pd.DataFrame({"f": [1.0 if v == "pos" else 0.0 for v in y]})

    oof, folds = grouped_cv(DecisionTreeClassifier(random_state=0), X, y, groups, n_splits=3)

    assert len(oof) == n
    assert oof.isna().sum() == 0
    assert len(folds) == 3
    assert (oof.values == y.values).all()


def test_grouped_cv_proba_shape_and_columns():
    n = 60
    groups = pd.Series(np.repeat(np.arange(6), 10))
    y = np.array(["pos" if i % 2 == 0 else "neg" for i in range(n)])
    X = pd.DataFrame({"f": [1.0 if v == "pos" else 0.0 for v in y]})
    classes = ["neg", "pos"]
    P = grouped_cv_proba(LogisticRegression(), X, y, groups, classes, n_splits=3)
    assert P.shape == (n, 2)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert (np.array(classes)[P.argmax(1)] == y).all()
