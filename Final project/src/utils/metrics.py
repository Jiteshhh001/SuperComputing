"""
═══════════════════════════════════════════════════════════════════════════════
  Metrics — AUC-ROC, F1, NDCG@K, Precision@K, and custom educational metrics
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, mean_squared_error, confusion_matrix
)


def auc_roc(y_true, y_scores):
    """Compute AUC-ROC with error handling."""
    try:
        return roc_auc_score(y_true, y_scores)
    except ValueError:
        return 0.5


def ndcg_at_k(y_true, y_scores, k=5):
    """
    Normalized Discounted Cumulative Gain at K.
    Used for evaluating recommendation ranking quality.
    """
    order = np.argsort(y_scores)[::-1][:k]
    dcg = sum(y_true[i] / np.log2(rank + 2) for rank, i in enumerate(order))

    ideal_order = np.argsort(y_true)[::-1][:k]
    idcg = sum(y_true[i] / np.log2(rank + 2) for rank, i in enumerate(ideal_order))

    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(y_true, y_scores, k=3):
    """Precision@K for recommendation evaluation."""
    top_k_indices = np.argsort(y_scores)[::-1][:k]
    return np.mean([y_true[i] for i in top_k_indices])


def mastery_improvement_rate(before: dict, after: dict) -> float:
    """Calculate the rate of mastery improvement across concepts."""
    improvements = []
    for concept in before:
        if concept in after:
            improvements.append(after[concept] - before[concept])
    return np.mean(improvements) if improvements else 0.0


def compute_all_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Compute a comprehensive set of classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        metrics["auc_roc"] = auc_roc(y_true, y_prob)
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_prob))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)

    return metrics
