import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
from src.plots import plot_confusion_matrix


def test_plot_confusion_matrix_returns_fig():
    y_true = ["a", "b", "a", "b"]
    y_pred = ["a", "a", "a", "b"]
    fig, ax = plot_confusion_matrix(y_true, y_pred, labels=["a", "b"])
    assert isinstance(fig, matplotlib.figure.Figure)
    assert ax is not None
