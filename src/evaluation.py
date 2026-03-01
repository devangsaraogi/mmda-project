import json
import logging
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)

LABEL_NAMES = {0: "True", 1: "False", 2: "Unverifiable"}


def recall_at_k(gold_ids: list[set], retrieved_ids: list[list[str]], k: int) -> float:
    """Compute Recall@K: fraction of queries where at least one gold image is in top-K.

    Args:
        gold_ids: List of sets of gold image IDs per query.
        retrieved_ids: List of ordered retrieved image ID lists per query.
        k: Number of top results to consider.

    Returns:
        Recall@K score.
    """
    hits = 0
    for gold, retrieved in zip(gold_ids, retrieved_ids):
        top_k = set(retrieved[:k])
        if top_k & gold:
            hits += 1
    return hits / len(gold_ids) if gold_ids else 0.0


def recall_at_k_curve(
    gold_ids: list[set],
    retrieved_ids: list[list[str]],
    k_values: list[int],
) -> dict[int, float]:
    """Compute Recall@K for multiple K values.

    Returns:
        Dict mapping K → Recall@K.
    """
    return {k: recall_at_k(gold_ids, retrieved_ids, k) for k in k_values}


def compute_verification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: dict = None,
) -> dict:
    """Compute comprehensive verification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        label_names: Optional label name mapping.

    Returns:
        Dict with accuracy, macro F1, per-class metrics, confusion matrix.
    """
    names = label_names or LABEL_NAMES
    present_labels = sorted(set(y_true) | set(y_pred))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    # Per-class metrics
    per_class = {}
    for label in present_labels:
        label_name = names.get(label, str(label))
        mask_true = y_true == label
        mask_pred = y_pred == label
        tp = int(np.sum(mask_true & mask_pred))
        fp = int(np.sum(~mask_true & mask_pred))
        fn = int(np.sum(mask_true & ~mask_pred))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall_val / (precision + recall_val) if (precision + recall_val) > 0 else 0.0
        per_class[label_name] = {
            "precision": float(precision),
            "recall": float(recall_val),
            "f1": float(f1),
            "support": int(np.sum(mask_true)),
        }
    metrics["per_class"] = per_class

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_labels"] = [names.get(l, str(l)) for l in present_labels]

    return metrics


def per_type_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    misinfo_types: list[str],
) -> dict:
    """Compute per-misinformation-type metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        misinfo_types: Misinformation type per sample.

    Returns:
        Dict mapping type → {accuracy, f1, support}.
    """
    breakdown = {}
    unique_types = sorted(set(misinfo_types))

    for mtype in unique_types:
        mask = np.array([t == mtype for t in misinfo_types])
        if not np.any(mask):
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        breakdown[mtype] = {
            "accuracy": float(accuracy_score(yt, yp)),
            "f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "support": int(np.sum(mask)),
        }

    return breakdown


def oracle_vs_e2e_comparison(oracle_metrics: dict, e2e_metrics: dict) -> dict:
    """Compare oracle vs E2E verification performance.

    Returns:
        Dict with side-by-side metrics and performance gaps.
    """
    comparison = {
        "oracle": {
            "accuracy": oracle_metrics["accuracy"],
            "macro_f1": oracle_metrics["macro_f1"],
        },
        "e2e": {
            "accuracy": e2e_metrics["accuracy"],
            "macro_f1": e2e_metrics["macro_f1"],
        },
        "gap": {
            "accuracy": oracle_metrics["accuracy"] - e2e_metrics["accuracy"],
            "macro_f1": oracle_metrics["macro_f1"] - e2e_metrics["macro_f1"],
        },
    }

    # Per-class gap
    per_class_gap = {}
    for cls_name in oracle_metrics.get("per_class", {}):
        if cls_name in e2e_metrics.get("per_class", {}):
            per_class_gap[cls_name] = {
                "f1_gap": (
                    oracle_metrics["per_class"][cls_name]["f1"]
                    - e2e_metrics["per_class"][cls_name]["f1"]
                ),
            }
    comparison["per_class_gap"] = per_class_gap

    return comparison


def save_metrics(metrics: dict, path: str) -> None:
    """Save metrics dict to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {path}")


def load_metrics(path: str) -> dict:
    """Load metrics from JSON file."""
    with open(path) as f:
        return json.load(f)
