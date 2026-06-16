import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from src.evaluation import grouped_cv


def test_grouped_cv_covers_all_no_leak_and_perfect():
    n = 60
    groups = pd.Series(np.repeat(np.arange(6), 10))          # 6 grupos
    y = pd.Series(["pos" if i % 2 == 0 else "neg" for i in range(n)])
    X = pd.DataFrame({"f": [1.0 if v == "pos" else 0.0 for v in y]})  # feature separa perfecto

    oof, folds = grouped_cv(DecisionTreeClassifier(random_state=0), X, y, groups, n_splits=3)

    assert len(oof) == n
    assert oof.isna().sum() == 0                 # cada muestra predicha exactamente una vez
    assert len(folds) == 3
    assert (oof.values == y.values).all()        # perfectamente separable -> oof perfecto
