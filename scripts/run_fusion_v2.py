"""Run the final-paper fusion experiments over pre-computed unimodal features.

Expects you to have already run the visual + text pipelines on the
``final/multimodal-fusion`` branch, which drop per-run feature caches at:

    results/runs/<run_id>/visual_features_<mode>.pt
    results/runs/<run_id>/text_features_<retriever>_<mode>.pt

where ``<mode>`` is ``oracle`` or ``e2e`` and ``<retriever>`` is one of
``bm25`` / ``dense`` / ``hybrid`` / ``hybrid_rerank``. Each .pt file has
train/val/test splits with ``features``, ``labels``, ``claim_ids``, and
``misinfo_types`` arrays.

What this script does:

  1. Load visual features (one path), text features (one path), align
     them by ``claim_id``.
  2. Train + evaluate four fusion variants:
     - score: concat + MLP  (FusionMLP)
     - gated: learned per-sample gate  (GatedFusionClassifier)
     - cross_attn: single cross-attention layer  (CrossAttentionFusion)
     - confidence: unimodal-entropy weighted late fusion  (ConfidenceWeightedFusion)
  3. Report Macro F1 + per-class + per-type for each variant.
  4. Save metrics under ``results/metrics/fusion_v2_<variant>_<tag>.json``.

``confidence`` needs unimodal MLPs trained on each modality to produce
probability vectors. If ``--unimodal-probs-visual`` and
``--unimodal-probs-text`` point at JSON files with the saved val+test
probs from prior runs, it loads them directly; otherwise it trains two
quick unimodal MLPs on the fly and uses their predictions. (Quick MLPs
are fine here — they only feed into the fusion head, which further
transforms the inputs.)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, set_seed, setup_logging, ensure_dirs, create_run_dir
from src.experiment_tracker import ExperimentTracker
from src.evaluation import (
    compute_verification_metrics,
    per_type_breakdown,
    save_metrics,
)
from src.models.fusion_classifier import FusionMLP, GatedFusionClassifier
from src.models.cross_attention_fusion import CrossAttentionFusion, ConfidenceWeightedFusion

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Feature loading + alignment
# ----------------------------------------------------------------------
def load_feature_pt(path: str, mode: str | None = None) -> dict:
    """Load a feature cache .pt file.

    Two on-disk shapes are supported:

      (a) Text pipeline (one file per mode):
              {"train": {...}, "val": {...}, "test": {...}}

      (b) Multimodal pipeline visual features (both modes in one file):
              {"oracle": {"train":..., "val":..., "test":...},
               "e2e":    {"train":..., "val":..., "test":...}}

    For shape (b), ``mode`` must be "oracle" or "e2e" to select which
    inner dict to return.
    """
    data = torch.load(path, weights_only=False)
    if "oracle" in data or "e2e" in data:  # shape (b)
        if mode not in data:
            raise ValueError(
                f"{path} has oracle/e2e layers; pass mode='oracle' or 'e2e' "
                f"to pick one (available keys: {list(data.keys())})"
            )
        data = data[mode]
    for split in ("train", "val", "test"):
        if split not in data:
            raise KeyError(f"{path} missing split '{split}'")
    return data


def align_features(visual: dict, text: dict) -> dict:
    """Align per-split features by claim_id; drop claims missing in either side."""
    out = {}
    for split in ("train", "val", "test"):
        v = visual[split]
        t = text[split]

        # Build claim_id -> row_index maps for each side.
        v_idx = {cid: i for i, cid in enumerate(v["claim_ids"])}
        t_idx = {cid: i for i, cid in enumerate(t["claim_ids"])}

        shared_ids = [cid for cid in v["claim_ids"] if cid in t_idx]
        if len(shared_ids) < min(len(v["claim_ids"]), len(t["claim_ids"])):
            dropped = max(len(v["claim_ids"]), len(t["claim_ids"])) - len(shared_ids)
            logger.info("[%s] dropped %d claims with mismatched ids", split, dropped)

        v_rows = [v_idx[c] for c in shared_ids]
        t_rows = [t_idx[c] for c in shared_ids]

        v_feats = np.asarray(v["features"])[v_rows]
        t_feats = np.asarray(t["features"])[t_rows]

        # Prefer labels from the visual side; sanity-check they match text side.
        v_labels = np.asarray(v["labels"])[v_rows]
        t_labels = np.asarray(t["labels"])[t_rows]
        if not np.array_equal(v_labels, t_labels):
            n_mismatch = int((v_labels != t_labels).sum())
            raise RuntimeError(
                f"[{split}] label mismatch on {n_mismatch} shared claims — "
                "visual and text pipelines disagree on verdicts. Refusing to fuse."
            )

        # misinfo_types: take from visual side (should match; shared source).
        v_types = [v["misinfo_types"][i] for i in v_rows]

        out[split] = {
            "visual_features": v_feats,
            "text_features": t_feats,
            "labels": v_labels,
            "claim_ids": shared_ids,
            "misinfo_types": v_types,
        }
    return out


# ----------------------------------------------------------------------
# Quick unimodal MLP (for ConfidenceWeightedFusion probability inputs)
# ----------------------------------------------------------------------
def _train_quick_mlp(X_tr, y_tr, X_val, y_val, input_dim, device,
                     epochs=30, patience_max=7):
    layers = [
        nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 64),         nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 3),
    ]
    model = nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    cw = torch.tensor([1.0, 0.73, 8.0], dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=cw)

    tr = DataLoader(TensorDataset(X_tr.to(device), y_tr.to(device)),
                    batch_size=64, shuffle=True)
    va = DataLoader(TensorDataset(X_val.to(device), y_val.to(device)),
                    batch_size=64)

    best_f1, best_state, pat = 0.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in tr:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            preds, labels = [], []
            for xb, yb in va:
                preds.extend(model(xb).argmax(1).cpu().numpy())
                labels.extend(yb.cpu().numpy())
        vf1 = f1_score(labels, preds, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience_max:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model.to(device)


def _probs(model, X, device, batch=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i:i + batch].to(device)
            out.append(torch.softmax(model(xb), dim=1).cpu())
    return torch.cat(out, dim=0)


# ----------------------------------------------------------------------
# Main fusion training loop
# ----------------------------------------------------------------------
def _train_fusion(model, train_data, val_data, cfg, device,
                  feat_a_key="visual_features", feat_b_key="text_features",
                  epochs=50, patience_max=7):
    cw = torch.tensor(list(cfg.verification.mlp.class_weights),
                      dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def _pack(d):
        return (
            torch.tensor(d[feat_a_key], dtype=torch.float32),
            torch.tensor(d[feat_b_key], dtype=torch.float32),
            torch.tensor(d["labels"], dtype=torch.long),
        )

    Xa_tr, Xb_tr, y_tr = _pack(train_data)
    Xa_va, Xb_va, y_va = _pack(val_data)

    tr = DataLoader(TensorDataset(Xa_tr, Xb_tr, y_tr), batch_size=64, shuffle=True)
    va = DataLoader(TensorDataset(Xa_va, Xb_va, y_va), batch_size=64)

    best_f1, best_state, pat = 0.0, None, 0
    for _ in range(epochs):
        model.train()
        for ab, bb, yb in tr:
            ab, bb, yb = ab.to(device), bb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(ab, bb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            preds, labels = [], []
            for ab, bb, yb in va:
                ab, bb, yb = ab.to(device), bb.to(device), yb.to(device)
                preds.extend(model(ab, bb).argmax(1).cpu().numpy())
                labels.extend(yb.cpu().numpy())
        vf1 = f1_score(labels, preds, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience_max:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model.to(device), best_f1


def _eval_fusion(model, data, device,
                 feat_a_key="visual_features", feat_b_key="text_features"):
    """Return predictions AND softmax probabilities (for post-hoc abstention + CIs)."""
    Xa = torch.tensor(data[feat_a_key], dtype=torch.float32).to(device)
    Xb = torch.tensor(data[feat_b_key], dtype=torch.float32).to(device)
    y = data["labels"]
    model.eval()
    with torch.no_grad():
        logits = model(Xa, Xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
    return preds, y, probs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--visual-features", type=str, required=True,
                    help="Path to visual_features.pt (or visual_features_<mode>.pt).")
    ap.add_argument("--visual-mode", type=str, default="e2e",
                    choices=["oracle", "e2e"],
                    help="Which inner mode to select if the visual file has both.")
    ap.add_argument("--text-features", type=str, required=True,
                    help="Path to text_features_<retriever>_<mode>.pt")
    ap.add_argument("--tag", type=str, required=True,
                    help="Short label for this feature combination (e.g. 'visE2E_txtE2E_large').")
    ap.add_argument("--variants", type=str, default="score,gated,cross_attn,confidence",
                    help="Comma-separated list of variants to run.")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    set_seed(cfg.seed)
    ensure_dirs(cfg)

    tracker = ExperimentTracker("fusion_v2", cfg.results.experiments_dir)
    tracker.log_config(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load + align
    logger.info("Loading visual features from %s (mode=%s)",
                args.visual_features, args.visual_mode)
    visual = load_feature_pt(args.visual_features, mode=args.visual_mode)
    logger.info("Loading text features from %s", args.text_features)
    text = load_feature_pt(args.text_features)
    aligned = align_features(visual, text)

    vdim = aligned["train"]["visual_features"].shape[1]
    tdim = aligned["train"]["text_features"].shape[1]
    logger.info("Aligned feature dims — visual=%d  text=%d", vdim, tdim)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    summary: dict = {"tag": args.tag, "variants": {}}

    # Train unimodal MLPs ALWAYS — we need their val+test probs both for
    # the confidence variant and for post-hoc cross-modality decomposition.
    logger.info("Training quick unimodal MLPs (needed for confidence fusion + post-hoc).")
    Xv_tr = torch.tensor(aligned["train"]["visual_features"], dtype=torch.float32)
    Xv_va = torch.tensor(aligned["val"]["visual_features"],   dtype=torch.float32)
    Xv_te = torch.tensor(aligned["test"]["visual_features"],  dtype=torch.float32)
    Xt_tr = torch.tensor(aligned["train"]["text_features"],   dtype=torch.float32)
    Xt_va = torch.tensor(aligned["val"]["text_features"],     dtype=torch.float32)
    Xt_te = torch.tensor(aligned["test"]["text_features"],    dtype=torch.float32)
    y_tr = torch.tensor(aligned["train"]["labels"], dtype=torch.long)
    y_va = torch.tensor(aligned["val"]["labels"],   dtype=torch.long)

    vis_mlp = _train_quick_mlp(Xv_tr, y_tr, Xv_va, y_va, vdim, device)
    txt_mlp = _train_quick_mlp(Xt_tr, y_tr, Xt_va, y_va, tdim, device)
    p_vis_va = _probs(vis_mlp, Xv_va, device)
    p_vis_te = _probs(vis_mlp, Xv_te, device)
    p_txt_va = _probs(txt_mlp, Xt_va, device)
    p_txt_te = _probs(txt_mlp, Xt_te, device)

    # Stash per-variant predictions + probs for the post-hoc analysis script.
    predictions_payload: dict = {
        "tag": args.tag,
        "labels_val":  aligned["val"]["labels"].tolist(),
        "labels_test": aligned["test"]["labels"].tolist(),
        "claim_ids_val":  list(aligned["val"]["claim_ids"]),
        "claim_ids_test": list(aligned["test"]["claim_ids"]),
        "misinfo_types_val":  list(aligned["val"]["misinfo_types"]),
        "misinfo_types_test": list(aligned["test"]["misinfo_types"]),
        "unimodal": {
            "visual_probs_val":  p_vis_va.numpy().tolist(),
            "visual_probs_test": p_vis_te.numpy().tolist(),
            "text_probs_val":    p_txt_va.numpy().tolist(),
            "text_probs_test":   p_txt_te.numpy().tolist(),
        },
        "variants": {},
    }

    for variant in variants:
        logger.info("==== Variant: %s ====", variant)

        if variant == "score":
            model = FusionMLP(visual_dim=vdim, text_dim=tdim).to(device)
        elif variant == "gated":
            model = GatedFusionClassifier(visual_dim=vdim, text_dim=tdim).to(device)
        elif variant == "cross_attn":
            model = CrossAttentionFusion(visual_dim=vdim, text_dim=tdim).to(device)
        elif variant == "confidence":
            model = ConfidenceWeightedFusion(num_classes=3).to(device)
        else:
            logger.warning("Unknown variant '%s' — skipping.", variant); continue

        if variant == "confidence":
            # Swap in probability inputs for this variant.
            aligned_probs = {
                "train": {  # use val probs for 'train' to keep API uniform; this variant trains only a tiny head
                    "visual_features": p_vis_va.numpy(),
                    "text_features":   p_txt_va.numpy(),
                    "labels":          aligned["val"]["labels"],
                    "misinfo_types":   aligned["val"]["misinfo_types"],
                },
                "val": {
                    "visual_features": p_vis_va.numpy(),
                    "text_features":   p_txt_va.numpy(),
                    "labels":          aligned["val"]["labels"],
                    "misinfo_types":   aligned["val"]["misinfo_types"],
                },
                "test": {
                    "visual_features": p_vis_te.numpy(),
                    "text_features":   p_txt_te.numpy(),
                    "labels":          aligned["test"]["labels"],
                    "misinfo_types":   aligned["test"]["misinfo_types"],
                },
            }
            model, best_val = _train_fusion(
                model, aligned_probs["train"], aligned_probs["val"], cfg, device,
                epochs=40, patience_max=8,
            )
            preds, y, probs_test = _eval_fusion(model, aligned_probs["test"], device)
            _, _, probs_val = _eval_fusion(model, aligned_probs["val"], device)
        else:
            model, best_val = _train_fusion(
                model, aligned["train"], aligned["val"], cfg, device,
            )
            preds, y, probs_test = _eval_fusion(model, aligned["test"], device)
            _, _, probs_val = _eval_fusion(model, aligned["val"], device)

        metrics = compute_verification_metrics(y, preds)
        type_bd = per_type_breakdown(y, preds, aligned["test"]["misinfo_types"])
        logger.info(
            "[%s][%s]  test Acc=%.4f  Macro F1=%.4f  (best val_f1=%.4f)",
            args.tag, variant, metrics["accuracy"], metrics["macro_f1"], best_val,
        )

        metrics_path = os.path.join(
            cfg.results.metrics_dir, f"fusion_v2_{variant}_{args.tag}.json",
        )
        save_metrics({"metrics": metrics, "per_type": type_bd, "val_best_f1": best_val},
                     metrics_path)
        summary["variants"][variant] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "val_f1": best_val,
        }
        predictions_payload["variants"][variant] = {
            "preds_test": preds.tolist(),
            "probs_test": probs_test.tolist(),
            "probs_val":  probs_val.tolist(),
            "val_f1":     float(best_val),
        }

    summary_path = os.path.join(cfg.results.metrics_dir,
                                f"fusion_v2_summary_{args.tag}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary -> %s", summary_path)

    # Predictions payload for the post-hoc analysis script (bootstrap CIs,
    # per-type breakdown, abstention sweep, cross-modality decomposition).
    preds_path = os.path.join(cfg.results.metrics_dir,
                              f"fusion_v2_predictions_{args.tag}.json")
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(predictions_payload, f)
    logger.info("Predictions -> %s", preds_path)

    tracker.log_results(**{f"{v}_f1": summary["variants"][v]["macro_f1"]
                           for v in summary["variants"]})
    tracker.complete()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
