"""Post-hoc Unverifiable-class abstention ablation.

Rescores existing end-to-end predictions with a val-calibrated
class-conditional threshold that remaps low-confidence predictions to
Unverifiable. Does not retrain; does not require GPU; should complete
in under a minute.

Reads saved softmax probabilities + labels from JSON. Expected schema:

    {
      "val":  {"probs": [[p0, p1, p2], ...], "labels": [0, 1, 2, ...]},
      "test": {"probs": [[p0, p1, p2], ...], "labels": [0, 1, 2, ...]}
    }

If that file does not yet exist, run ``scripts/run_multimodal_pipeline.py``
with the ``verification.save_probs=true`` config override (or apply the
small patch described in the project memory) — the pipeline will then
dump ``<metrics_dir>/<variant>_probs.json`` files you can point this
script at.

Usage:
    python scripts/run_abstention_ablation.py \\
        --probs results/runs/<RUN>/metrics/fusion_score_e2e_probs.json \\
        --mode  per_class \\
        --out   results/runs/<RUN>/metrics/abstention_fusion_score_e2e.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.abstention import calibrate_abstention, summarise


def load_probs(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for split in ("val", "test"):
        if split not in data:
            raise KeyError(f"Missing '{split}' block in {path}")
        for key in ("probs", "labels"):
            if key not in data[split]:
                raise KeyError(f"Missing '{split}.{key}' in {path}")
    return data


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probs", type=Path, required=True,
                    help="JSON file with val/test probs + labels")
    ap.add_argument("--mode", choices=("per_class", "global"), default="per_class")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = load_probs(args.probs)

    val_probs = np.asarray(data["val"]["probs"])
    val_labels = np.asarray(data["val"]["labels"])
    test_probs = np.asarray(data["test"]["probs"])
    test_labels = np.asarray(data["test"]["labels"])

    if val_probs.shape[0] == 0 or test_probs.shape[0] == 0:
        print("No data found — empty probs array.")
        sys.exit(2)
    if val_probs.shape[1] != 3 or test_probs.shape[1] != 3:
        print(f"Expected 3-class probs, got shapes {val_probs.shape} / {test_probs.shape}")
        sys.exit(2)

    cfg = calibrate_abstention(val_probs, val_labels, mode=args.mode)
    summary = summarise(cfg, test_probs, test_labels)

    print("=" * 60)
    print("ABSTENTION ABLATION")
    print("=" * 60)
    print(f"Source: {args.probs}")
    print(f"Calibration mode: {cfg.mode}")
    if cfg.mode == "per_class":
        print(f"Class thresholds:  {cfg.class_thresholds}")
    else:
        print(f"Global threshold:  {cfg.threshold:.3f}")
    print(f"Val F1 before:     {cfg.val_f1_before:.4f}")
    print(f"Val F1 after:      {cfg.val_f1_after:.4f}")
    print(f"Val Unverif F1:    {cfg.val_unverif_f1_before:.4f} -> {cfg.val_unverif_f1_after:.4f}")
    print("-" * 60)
    print("Test (before / after):")
    tb = summary["test_metrics_before"]
    ta = summary["test_metrics_after"]
    print(f"  Accuracy:    {tb['accuracy']:.4f}  ->  {ta['accuracy']:.4f}")
    print(f"  Macro F1:    {tb['macro_f1']:.4f}  ->  {ta['macro_f1']:.4f}")
    print(f"  Unverif F1:  {tb['unverif_f1']:.4f}  ->  {ta['unverif_f1']:.4f}")
    print("=" * 60)

    if args.out is None:
        args.out = args.probs.parent / f"abstention_{args.probs.stem}.json"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {args.out}")


if __name__ == "__main__":
    main()
