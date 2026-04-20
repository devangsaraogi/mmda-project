"""Post-hoc analysis of fusion v2 predictions — deepens the 4-variant table.

Reads the ``fusion_v2_predictions_<tag>.json`` payload(s) dumped by
``run_fusion_v2.py`` and computes, per variant:

  (1) Bootstrap 95% CIs on Macro F1 (1000 resamples, with-replacement).
  (2) Per-class F1 (True / False / Unverifiable).
  (3) Per-manipulation-type F1 breakdown.
  (4) Confusion matrix (3x3).
  (5) Per-class abstention sweep on the val split; report the calibrated
      test-split F1 and Unverifiable-F1 after remapping.
  (6) Cross-modality decomposition using the quick unimodal MLPs that
      ``run_fusion_v2.py`` trains alongside each fusion head.
  (7) Pairwise McNemar-style disagreement counts between fusion heads
      (where do they diverge on which claims).

No cluster, no GPU — pure CPU / numpy. Runs in <30s per tag.

Output: ``results/metrics/fusion_v2_posthoc_<tag>.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LABEL_NAMES = ["True", "False", "Unverifiable"]


# ----------------------------------------------------------------------
# Core metrics
# ----------------------------------------------------------------------
def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {LABEL_NAMES[i]: float(per[i]) for i in range(3)}


def _per_type_f1(y_true: np.ndarray, y_pred: np.ndarray, types: list[str]) -> dict:
    out = {}
    types_arr = np.asarray(types)
    for t in sorted(set(types_arr)):
        mask = types_arr == t
        if mask.sum() == 0:
            continue
        out[t] = {
            "n": int(mask.sum()),
            "macro_f1": _macro_f1(y_true[mask], y_pred[mask]),
            "per_class": _per_class_f1(y_true[mask], y_pred[mask]),
        }
    return out


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()


# ----------------------------------------------------------------------
# Bootstrap CIs
# ----------------------------------------------------------------------
def _bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                  n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1s = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f1s[b] = _macro_f1(y_true[idx], y_pred[idx])
    return {
        "mean":   float(f1s.mean()),
        "ci_lo":  float(np.percentile(f1s, 2.5)),
        "ci_hi":  float(np.percentile(f1s, 97.5)),
        "std":    float(f1s.std()),
        "n_boot": int(n_boot),
    }


# ----------------------------------------------------------------------
# Abstention (per-class threshold sweep on val → evaluate on test)
# ----------------------------------------------------------------------
def _apply_abstention(preds: np.ndarray, probs: np.ndarray,
                      tau_true: float, tau_false: float) -> np.ndarray:
    """Remap low-confidence True/False predictions to Unverifiable (class 2)."""
    out = preds.copy()
    max_p = probs.max(axis=1)
    # Low-confidence True → Unverifiable
    out[(preds == 0) & (max_p < tau_true)] = 2
    # Low-confidence False → Unverifiable
    out[(preds == 1) & (max_p < tau_false)] = 2
    return out


def _calibrate_abstention(val_probs: np.ndarray, val_labels: np.ndarray,
                          test_probs: np.ndarray, test_labels: np.ndarray) -> dict:
    """Grid search per-class thresholds on val, apply to test."""
    val_preds = val_probs.argmax(axis=1)
    test_preds = test_probs.argmax(axis=1)

    best = {"tau_true": 0.0, "tau_false": 0.0, "val_f1": _macro_f1(val_labels, val_preds)}
    grid = np.linspace(0.0, 0.95, 20)
    for t_t in grid:
        for t_f in grid:
            v = _apply_abstention(val_preds, val_probs, t_t, t_f)
            f1 = _macro_f1(val_labels, v)
            if f1 > best["val_f1"]:
                best = {"tau_true": float(t_t), "tau_false": float(t_f), "val_f1": float(f1)}

    calibrated_test = _apply_abstention(test_preds, test_probs,
                                         best["tau_true"], best["tau_false"])
    return {
        "thresholds": {"true": best["tau_true"], "false": best["tau_false"]},
        "val_f1_calibrated": best["val_f1"],
        "test_f1_uncalibrated": _macro_f1(test_labels, test_preds),
        "test_f1_calibrated":   _macro_f1(test_labels, calibrated_test),
        "test_per_class_calibrated": _per_class_f1(test_labels, calibrated_test),
        "test_confusion_calibrated": _confusion(test_labels, calibrated_test),
    }


# ----------------------------------------------------------------------
# Cross-modality decomposition (text-only / visual-only / both / neither)
# ----------------------------------------------------------------------
def _cross_modality(y_true: np.ndarray,
                    vis_probs: np.ndarray, txt_probs: np.ndarray,
                    fusion_preds: np.ndarray) -> dict:
    """Decompose test claims into 4 buckets based on unimodal correctness.

    Also reports fusion accuracy conditional on each bucket — tells us
    whether fusion recovers claims that both modalities missed
    (``neither`` bucket with non-zero fusion accuracy) or just rides
    on the cases where at least one modality was already right.
    """
    vis_pred = vis_probs.argmax(axis=1)
    txt_pred = txt_probs.argmax(axis=1)
    v_ok = vis_pred == y_true
    t_ok = txt_pred == y_true
    f_ok = fusion_preds == y_true

    buckets = {
        "both":         (v_ok & t_ok),
        "text_only":    (~v_ok & t_ok),
        "visual_only":  (v_ok & ~t_ok),
        "neither":      (~v_ok & ~t_ok),
    }
    out = {}
    total = len(y_true)
    for name, mask in buckets.items():
        n = int(mask.sum())
        out[name] = {
            "n":               n,
            "fraction":        round(n / total, 4),
            "fusion_correct":  int((mask & f_ok).sum()),
            "fusion_accuracy": round(float((mask & f_ok).sum() / max(n, 1)), 4),
        }
    return out


# ----------------------------------------------------------------------
# Pairwise variant disagreements
# ----------------------------------------------------------------------
def _variant_disagreements(preds_by_variant: dict[str, np.ndarray]) -> dict:
    """How often do two fusion heads disagree on a test claim?"""
    out = {}
    variants = list(preds_by_variant.keys())
    for i, a in enumerate(variants):
        for b in variants[i + 1:]:
            disagree = int((preds_by_variant[a] != preds_by_variant[b]).sum())
            out[f"{a}_vs_{b}"] = {
                "n_disagree": disagree,
                "fraction":   round(disagree / len(preds_by_variant[a]), 4),
            }
    return out


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def analyze_one(payload: dict, n_boot: int = 1000) -> dict:
    tag = payload["tag"]
    labels_test = np.asarray(payload["labels_test"])
    labels_val  = np.asarray(payload["labels_val"])
    types_test  = payload["misinfo_types_test"]

    vis_probs_test = np.asarray(payload["unimodal"]["visual_probs_test"])
    txt_probs_test = np.asarray(payload["unimodal"]["text_probs_test"])

    report: dict = {"tag": tag, "n_test": int(len(labels_test)), "variants": {}}

    preds_by_variant: dict[str, np.ndarray] = {}
    for variant, pack in payload["variants"].items():
        preds_test = np.asarray(pack["preds_test"])
        probs_test = np.asarray(pack["probs_test"])
        probs_val  = np.asarray(pack["probs_val"])
        preds_by_variant[variant] = preds_test

        report["variants"][variant] = {
            "macro_f1_point":  _macro_f1(labels_test, preds_test),
            "macro_f1_bootstrap_ci": _bootstrap_ci(labels_test, preds_test, n_boot=n_boot),
            "per_class_f1":    _per_class_f1(labels_test, preds_test),
            "per_type_f1":     _per_type_f1(labels_test, preds_test, types_test),
            "confusion":       _confusion(labels_test, preds_test),
            "abstention":      _calibrate_abstention(probs_val, labels_val,
                                                      probs_test, labels_test),
            "cross_modality":  _cross_modality(labels_test, vis_probs_test,
                                                txt_probs_test, preds_test),
        }

    report["pairwise_disagreement"] = _variant_disagreements(preds_by_variant)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("predictions", nargs="+",
                    help="One or more fusion_v2_predictions_<tag>.json files.")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="Bootstrap resamples.")
    ap.add_argument("--out-dir", type=str, default="results/metrics",
                    help="Where to write the post-hoc JSON.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for path in args.predictions:
        logger.info("Analysing %s", path)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        report = analyze_one(payload, n_boot=args.n_boot)

        out_path = os.path.join(args.out_dir, f"fusion_v2_posthoc_{report['tag']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Wrote %s", out_path)

        # Print compact summary to stdout.
        print(f"\n=== {report['tag']} (n={report['n_test']}) ===")
        for variant, m in report["variants"].items():
            ci = m["macro_f1_bootstrap_ci"]
            ab = m["abstention"]
            print(f"  {variant:12s}  F1={m['macro_f1_point']:.4f}  "
                  f"CI=[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]  "
                  f"abst_F1={ab['test_f1_calibrated']:.4f}  "
                  f"τ=({ab['thresholds']['true']:.2f}, {ab['thresholds']['false']:.2f})")
        print("  pairwise disagreements (% of test):")
        for k, v in report["pairwise_disagreement"].items():
            print(f"    {k}: {v['fraction'] * 100:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
