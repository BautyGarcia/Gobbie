from src.metrics import evaluate


def test_evaluate_perfect():
    y = ["a", "b", "a", "c"]
    out = evaluate(y, y)
    assert out["accuracy"] == 1.0
    assert out["macro_f1"] == 1.0
    assert out["per_class_f1"] == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_evaluate_known_case():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    out = evaluate(y_true, y_pred)
    assert out["accuracy"] == 0.75
    assert round(out["macro_f1"], 4) == round((2/3 + 0.8) / 2, 4)
    assert round(out["per_class_f1"]["b"], 4) == 0.8
