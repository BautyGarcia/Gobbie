"""Cross-validation por hablante (GroupKFold). Agnóstico a las features y al modelo."""
import numpy as np
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

    `X` puede ser tabular (DataFrame/2D) o una lista de secuencias de largo variable
    (una lista/object-array de arrays, p.ej. embeddings frame-level para el CRNN).
    """
    y = pd.Series(list(y)).reset_index(drop=True)
    groups = pd.Series(np.asarray(groups)).reset_index(drop=True)

    is_seq = isinstance(X, list) or (isinstance(X, np.ndarray) and X.dtype == object and X.ndim == 1)
    if is_seq:
        Xv = list(X)
        take = lambda idx: [Xv[i] for i in idx]
    else:
        Xv = pd.DataFrame(X).reset_index(drop=True)
        take = lambda idx: Xv.iloc[idx]
    n = len(y)

    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = pd.Series(index=y.index, dtype=object)
    fold_metrics = []
    for train_idx, test_idx in gkf.split(np.zeros(n), y, groups):
        model = clone(estimator)
        model.fit(take(train_idx), y.iloc[train_idx])
        pred = model.predict(take(test_idx))
        oof_pred.iloc[test_idx] = pred
        fold_metrics.append(evaluate(y.iloc[test_idx], pred))
    return oof_pred, fold_metrics


def grouped_cv_proba(estimator, X, y, groups, classes, n_splits: int = 5):
    """Como grouped_cv pero devuelve la matriz de probabilidades out-of-fold (n, n_clases),
    con columnas en el orden de `classes`. Para ensembles por soft-voting (promediar probas).

    Requiere que el estimador exponga `predict_proba` y `classes_` (LogReg, MLP, etc.).
    `X` debe ser tabular (DataFrame). Mismo split GroupKFold por actor que `grouped_cv`.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y)
    groups = np.asarray(groups)
    classes = list(classes)

    P = np.zeros((len(y), len(classes)))
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(np.zeros(len(y)), y, groups):
        clf = clone(estimator).fit(X.iloc[tr], y[tr])
        proba = clf.predict_proba(X.iloc[te])
        for j, c in enumerate(clf.classes_):
            P[te, classes.index(c)] = proba[:, j]
    return P
