"""Classification metric helpers independent of model execution."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def calculate_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                      class_names: Sequence[str]) -> dict[str, Any]:
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=list(class_names),
        zero_division=0, output_dict=True
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def top_k_accuracy(y_true: Sequence[int], probabilities: Sequence[Sequence[float]],
                   k: int) -> float:
    values = np.asarray(probabilities)
    if values.ndim != 2 or values.shape[0] != len(y_true):
        raise ValueError("probabilities must be a 2D array aligned with y_true")
    k = min(k, values.shape[1])
    top = np.argpartition(values, -k, axis=1)[:, -k:]
    return float(np.mean([actual in candidates for actual, candidates in zip(y_true, top)]))
