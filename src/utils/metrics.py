import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "n_samples": int(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "auroc": _safe_metric(roc_auc_score, y_true, y_prob),
        "auprc": _safe_metric(average_precision_score, y_true, y_prob),
        "accuracy": _safe_metric(accuracy_score, y_true, y_pred),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, y_true, y_pred),
        "precision": _safe_metric(precision_score, y_true, y_pred, zero_division=0),
        "recall": _safe_metric(recall_score, y_true, y_pred, zero_division=0),
        "f1": _safe_metric(f1_score, y_true, y_pred, zero_division=0),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])
    return metrics
