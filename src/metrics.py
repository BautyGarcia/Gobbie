"""Métricas de clasificación multiclase para SER. Agnóstico a las features."""
from sklearn.metrics import accuracy_score, f1_score


def evaluate(y_true, y_pred, labels=None) -> dict:
    """Devuelve accuracy, macro-F1, weighted-F1 y F1 por clase."""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "per_class_f1": {lab: float(s) for lab, s in zip(labels, per_class)},
    }
