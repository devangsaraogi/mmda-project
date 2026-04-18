"""Selective-prediction abstention fix for the Unverifiable-class E2E collapse.

Background: at midterm, every E2E variant (visual, text, fusion) predicted
the Unverifiable class with F1=0.000 despite an 8x class weight. The Oracle
variant recovered Unverifiable F1 to ~0.473. That asymmetry says the
classifier is capable of predicting the class but never does when its
inputs are noisy (retrieval failures produce features that look like
low-confidence predictions for True / False, not Unverifiable).

This module implements the standard post-hoc fix: if the top-class
probability is below a threshold, remap the prediction to Unverifiable.
Thresholds are calibrated on the validation split by grid search over
Macro F1, then applied to the test split.

Usage in code:
    from src.abstention import calibrate_abstention, apply_abstention
    thresh = calibrate_abstention(val_probs, val_labels)
    y_pred_new = apply_abstention(test_probs, test_preds, thresh)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

TRUE_LABEL = 0
FALSE_LABEL = 1
UNVERIFIABLE_LABEL = 2


@dataclass
class AbstentionConfig:
    """Calibrated abstention configuration returned by calibrate_abstention."""

    mode: str                 # "global" or "per_class"
    threshold: float          # global threshold (mode=="global") or best found
    class_thresholds: dict | None  # {class_id: threshold} when mode=="per_class"
    val_f1_before: float
    val_f1_after: float
    val_unverif_f1_before: float
    val_unverif_f1_after: float


def _argmax(probs: np.ndarray) -> np.ndarray:
    return np.argmax(probs, axis=1)


def _max_prob(probs: np.ndarray) -> np.ndarray:
    return np.max(probs, axis=1)


def apply_abstention_global(
    probs: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Global-threshold abstention: remap to Unverifiable if top-class prob < threshold."""
    preds = _argmax(probs)
    confidence = _max_prob(probs)
    preds = preds.copy()
    preds[confidence < threshold] = UNVERIFIABLE_LABEL
    return preds


def apply_abstention_per_class(
    probs: np.ndarray,
    class_thresholds: dict,
) -> np.ndarray:
    """Per-class abstention: different threshold depending on the originally-predicted class."""
    preds = _argmax(probs)
    confidence = _max_prob(probs)
    out = preds.copy()
    for cls, tau in class_thresholds.items():
        mask = (preds == cls) & (confidence < tau)
        out[mask] = UNVERIFIABLE_LABEL
    return out


def calibrate_abstention(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    mode: str = "per_class",
    grid: Sequence[float] | None = None,
) -> AbstentionConfig:
    """Grid-search the abstention threshold that maximises validation Macro F1.

    Args:
        val_probs: [N, 3] softmax probabilities from the classifier on val.
        val_labels: [N] integer labels (0=True, 1=False, 2=Unverifiable).
        mode: "global" (one threshold) or "per_class" (one threshold per predicted class).
        grid: Threshold candidates to sweep. Default: np.linspace(0.30, 0.95, 27).

    Returns:
        AbstentionConfig with the selected thresholds and before/after metrics.
    """
    if grid is None:
        grid = np.linspace(0.30, 0.95, 27)
    val_probs = np.asarray(val_probs)
    val_labels = np.asarray(val_labels)
    baseline_preds = _argmax(val_probs)

    f1_before = float(f1_score(val_labels, baseline_preds, average="macro", zero_division=0))
    unverif_mask = val_labels == UNVERIFIABLE_LABEL
    unverif_pred_before = baseline_preds[unverif_mask]
    unverif_f1_before = float(f1_score(
        val_labels == UNVERIFIABLE_LABEL,
        baseline_preds == UNVERIFIABLE_LABEL,
        average="binary",
        zero_division=0,
    ))

    if mode == "global":
        best_tau = grid[0]
        best_f1 = f1_before
        for tau in grid:
            preds = apply_abstention_global(val_probs, tau)
            f1 = float(f1_score(val_labels, preds, average="macro", zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_tau = float(tau)
        final_preds = apply_abstention_global(val_probs, best_tau)
        unverif_f1_after = float(f1_score(
            val_labels == UNVERIFIABLE_LABEL,
            final_preds == UNVERIFIABLE_LABEL,
            average="binary",
            zero_division=0,
        ))
        return AbstentionConfig(
            mode="global",
            threshold=best_tau,
            class_thresholds=None,
            val_f1_before=f1_before,
            val_f1_after=best_f1,
            val_unverif_f1_before=unverif_f1_before,
            val_unverif_f1_after=unverif_f1_after,
        )

    # per_class: independently sweep each non-Unverifiable class's threshold
    class_taus: dict[int, float] = {TRUE_LABEL: 0.0, FALSE_LABEL: 0.0}
    current_preds = baseline_preds.copy()

    for cls in (TRUE_LABEL, FALSE_LABEL):
        best_tau = 0.0
        best_f1 = float(f1_score(val_labels, current_preds, average="macro", zero_division=0))
        for tau in grid:
            trial_thresholds = dict(class_taus)
            trial_thresholds[cls] = float(tau)
            preds = apply_abstention_per_class(val_probs, trial_thresholds)
            f1 = float(f1_score(val_labels, preds, average="macro", zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_tau = float(tau)
        class_taus[cls] = best_tau
        current_preds = apply_abstention_per_class(val_probs, class_taus)

    final_preds = apply_abstention_per_class(val_probs, class_taus)
    f1_after = float(f1_score(val_labels, final_preds, average="macro", zero_division=0))
    unverif_f1_after = float(f1_score(
        val_labels == UNVERIFIABLE_LABEL,
        final_preds == UNVERIFIABLE_LABEL,
        average="binary",
        zero_division=0,
    ))

    return AbstentionConfig(
        mode="per_class",
        threshold=max(class_taus.values()),  # info only
        class_thresholds=class_taus,
        val_f1_before=f1_before,
        val_f1_after=f1_after,
        val_unverif_f1_before=unverif_f1_before,
        val_unverif_f1_after=unverif_f1_after,
    )


def apply_abstention(
    probs: np.ndarray,
    cfg: AbstentionConfig,
) -> np.ndarray:
    """Apply a calibrated AbstentionConfig to a new set of probabilities."""
    probs = np.asarray(probs)
    if cfg.mode == "global":
        return apply_abstention_global(probs, cfg.threshold)
    return apply_abstention_per_class(probs, cfg.class_thresholds)


def summarise(
    cfg: AbstentionConfig,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    """Apply the calibrated abstention to test and return a metrics summary."""
    test_probs = np.asarray(test_probs)
    test_labels = np.asarray(test_labels)
    baseline = _argmax(test_probs)
    fixed = apply_abstention(test_probs, cfg)

    out = {
        "calibration": {
            "mode": cfg.mode,
            "threshold": cfg.threshold,
            "class_thresholds": cfg.class_thresholds,
            "val_f1_before": cfg.val_f1_before,
            "val_f1_after": cfg.val_f1_after,
            "val_unverif_f1_before": cfg.val_unverif_f1_before,
            "val_unverif_f1_after": cfg.val_unverif_f1_after,
        },
        "test_metrics_before": {
            "accuracy": float((baseline == test_labels).mean()),
            "macro_f1": float(f1_score(test_labels, baseline, average="macro", zero_division=0)),
            "unverif_f1": float(f1_score(
                test_labels == UNVERIFIABLE_LABEL,
                baseline == UNVERIFIABLE_LABEL,
                average="binary",
                zero_division=0,
            )),
        },
        "test_metrics_after": {
            "accuracy": float((fixed == test_labels).mean()),
            "macro_f1": float(f1_score(test_labels, fixed, average="macro", zero_division=0)),
            "unverif_f1": float(f1_score(
                test_labels == UNVERIFIABLE_LABEL,
                fixed == UNVERIFIABLE_LABEL,
                average="binary",
                zero_division=0,
            )),
        },
    }
    return out
