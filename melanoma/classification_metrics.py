from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)


def binary_metrics(
    y_true: Iterable[int],
    probs: Iterable[float],
    threshold: float,
):
    y = np.asarray(list(y_true), dtype=np.int64)
    p = np.asarray(list(probs), dtype=np.float64)
    pred = (p >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    precision = precision_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    balanced_accuracy = (
        (sensitivity + specificity) / 2.0
        if sensitivity == sensitivity and specificity == specificity
        else float("nan")
    )
    auc_roc = float(roc_auc_score(y, p)) if len(set(y.tolist())) >= 2 else float("nan")
    auc_pr = (
        float(average_precision_score(y, p))
        if len(set(y.tolist())) >= 2
        else float("nan")
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "negative_predictive_value": float(npv),
        "f1": float(f1),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_sweep_summary(
    y_true: Iterable[int],
    probs: Iterable[float],
    thresholds: Iterable[float] | None = None,
    specificity_targets: tuple[float, ...] = (0.8, 0.9),
):
    if thresholds is None:
        thresholds = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), [0.5]]))

    rows = [binary_metrics(y_true, probs, float(t)) for t in thresholds]

    def finite(v: float) -> float:
        return v if v == v else -np.inf

    best_balanced = max(rows, key=lambda m: finite(m["balanced_accuracy"]))
    best_youden = max(
        rows,
        key=lambda m: finite(m["sensitivity"]) + finite(m["specificity"]) - 1.0,
    )
    best_f1 = max(rows, key=lambda m: finite(m["f1"]))

    by_specificity: dict[str, dict | None] = {}
    for target in specificity_targets:
        candidates = [m for m in rows if m["specificity"] == m["specificity"] and m["specificity"] >= target]
        key = f"sensitivity_at_specificity_ge_{target:.2f}"
        by_specificity[key] = (
            max(candidates, key=lambda m: finite(m["sensitivity"])) if candidates else None
        )

    return {
        "best_balanced_accuracy": best_balanced,
        "best_youden_j": best_youden,
        "best_f1": best_f1,
        **by_specificity,
    }
