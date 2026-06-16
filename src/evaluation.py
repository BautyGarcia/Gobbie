"""Cross-validation por hablante (GroupKFold). Agnóstico a las features y al modelo."""
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from src.metrics import evaluate


def grouped_cv(estimator, X, y, groups, n_splits: int = 5):
    """Corre GroupKFold(groups=actor) y devuelve (oof_pred, fold_metrics).

    - oof_pred: Series con la predicción out-of-fold de cada muestra (cada muestra
      se predice exactamente una vez, en el fold donde su grupo es test).
    - fold_metrics: lista de dicts (uno por fold) con las métricas de `evaluate`.
    El estimador se clona en cada fold (no se reutiliza el mismo objeto entrenado).
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)

    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = pd.Series(index=y.index, dtype=object)
    fold_metrics = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        oof_pred.iloc[test_idx] = pred
        fold_metrics.append(evaluate(y.iloc[test_idx], pred))
    return oof_pred, fold_metrics
