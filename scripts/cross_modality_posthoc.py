"""Post-hoc cross-modality decomposition + before/after confusion matrices.

Answers: did per-claim retrieval actually grow the "both modalities correct"
bucket in E2E fusion, the way the midterm analysis predicted?

Runs on the login node (no GPU needed — features are 6-dim, MLPs train in
seconds). Reads the saved artefacts from the multimodal pipeline run:

  <fusion-run>/visual_features.pt   — per-claim E2E features we care about
  <fusion-run>/text_features.pt     — midterm-config NLI features
  results/runs/<VERIF_RUN>/metrics/mlp_e2e_probs.json
                                    — global-E2E visual probs for the "before" row

Produces two tables:

  1. Cross-modality decomposition (text-only / vis-only / both / neither)
     for (a) midterm global E2E, (b) per-claim E2E, (c) oracle — on the
     same 7,392-claim test split.

  2. Confusion matrices (counts) for:
     - visual E2E (global, midterm-style)
     - visual E2E (per-claim)
     - text E2E
     - fusion gated E2E+E2E (per-claim)
     - fusion gated E2E+E2E + abstention

All written to <out-dir>/ as JSON + one PNG panel.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

LABEL_NAMES = ["True", "False", "Unverifiable"]
TRUE_LABEL, FALSE_LABEL, UNVERIFIABLE_LABEL = 0, 1, 2


def _save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote {path}")


def _train_mlp(X_tr, y_tr, X_val, y_val, dim, hidden=(256, 128), class_weights=(1.0, 0.73, 8.0),
               max_epochs=50, patience=7, device=torch.device("cpu")):
    """Small utility: train a tiny MLP with early-stopping on val Macro F1."""
    from sklearn.metrics import f1_score
    layers = []
    prev = dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
        prev = h
    layers.append(nn.Linear(prev, 3))
    net = nn.Sequential(*layers).to(device)
    cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
    best_f1, best_state, pat = 0.0, None, 0
    for _ in range(max_epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(net(xb), yb).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vp = net(X_val).argmax(1).cpu().numpy()
        vf = f1_score(y_val.cpu().numpy(), vp, average="macro", zero_division=0)
        if vf > best_f1:
            best_f1 = vf
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def _predict(net, X, device=torch.device("cpu")):
    net.to(device).eval()
    with torch.no_grad():
        return net(X).argmax(1).cpu().numpy()


def _cross_modality(y_true, visual_pred, text_pred):
    """Return {text_only, vis_only, both, neither} counts."""
    v_correct = visual_pred == y_true
    t_correct = text_pred == y_true
    return {
        "text_only":   int(((~v_correct) & t_correct).sum()),
        "visual_only": int((v_correct & (~t_correct)).sum()),
        "both":        int((v_correct & t_correct).sum()),
        "neither":     int(((~v_correct) & (~t_correct)).sum()),
        "total":       int(len(y_true)),
    }


def _confusion(y_true, y_pred):
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm.tolist()


def _plot_confusion_panel(matrices: list[tuple[str, np.ndarray]], out_path: Path) -> None:
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, (title, cm) in zip(axes, matrices):
        ax.imshow(cm, cmap="Blues", aspect="equal")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(LABEL_NAMES, fontsize=8, rotation=30, ha="right")
        ax.set_yticklabels(LABEL_NAMES, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.set_title(title, fontsize=10, pad=8)
        cell_max = cm.max()
        for i in range(3):
            for j in range(3):
                v = cm[i, j]
                col = "white" if v > 0.55 * cell_max else "#222"
                ax.text(j, i, f"{v}", ha="center", va="center",
                        fontsize=9, color=col)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# -----------------------------------------------------------------------------
# Abstention helper (inline to keep this script standalone)
# -----------------------------------------------------------------------------

def _apply_abstention(probs: np.ndarray, probs_val: np.ndarray, labels_val: np.ndarray,
                      grid=None):
    from sklearn.metrics import f1_score
    if grid is None:
        grid = np.linspace(0.30, 0.95, 27)
    baseline = np.argmax(probs_val, axis=1)
    # Grid-search per-class threshold (True + False), same logic as src.abstention.
    class_taus = {TRUE_LABEL: 0.0, FALSE_LABEL: 0.0}
    current = baseline.copy()

    def _apply(prb, taus):
        preds = np.argmax(prb, axis=1).copy()
        conf = np.max(prb, axis=1)
        for cls, tau in taus.items():
            mask = (preds == cls) & (conf < tau)
            preds[mask] = UNVERIFIABLE_LABEL
        return preds

    for cls in (TRUE_LABEL, FALSE_LABEL):
        best_tau = 0.0
        best_f1 = float(f1_score(labels_val, current, average="macro", zero_division=0))
        for tau in grid:
            trial = dict(class_taus)
            trial[cls] = float(tau)
            preds = _apply(probs_val, trial)
            f1 = float(f1_score(labels_val, preds, average="macro", zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_tau = float(tau)
        class_taus[cls] = best_tau
        current = _apply(probs_val, class_taus)

    return _apply(probs, class_taus), class_taus


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fusion-run", type=Path, required=True,
                    help="Run dir from run_multimodal_pipeline.py (per-claim variant)")
    ap.add_argument("--verify-run", type=Path, required=True,
                    help="Run dir from run_verification.py (global E2E probs for 'before')")
    ap.add_argument("--out-dir", type=Path, default=Path("results/post_hoc"))
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load saved features from the per-claim fusion run ----
    vis = torch.load(args.fusion_run / "visual_features.pt", map_location="cpu", weights_only=False)
    txt = torch.load(args.fusion_run / "text_features.pt", map_location="cpu", weights_only=False)
    # These are nested dicts: {oracle: {train, val, test: {features, labels, ...}}, e2e: {...}}

    def to_split(d, split):
        return {
            "X": torch.tensor(d[split]["features"], dtype=torch.float32),
            "y": torch.tensor(d[split]["labels"], dtype=torch.long),
            "ids": d[split].get("claim_ids"),
            "types": d[split].get("misinfo_types"),
        }

    V_oracle = {s: to_split(vis["oracle"], s) for s in ("train", "val", "test")}
    V_e2e    = {s: to_split(vis["e2e"],    s) for s in ("train", "val", "test")}
    T_oracle = {s: to_split(txt["oracle"], s) for s in ("train", "val", "test")}
    T_e2e    = {s: to_split(txt["e2e"],    s) for s in ("train", "val", "test")}

    y_test = V_e2e["test"]["y"].cpu().numpy()

    # Reconstruct the claim_id / misinfo_type alignment from the pipeline run if present;
    # fall back to the ones saved by the text pipeline which should match by claim_id.
    print(f"Test split: {len(y_test)} claims")

    # ---- Train tiny MLPs for single-modality predictions (per-claim visual) ----
    print("\nTraining tiny MLPs on saved features...")

    def _fit_and_pred(pack_train, pack_val, pack_test, label):
        net = _train_mlp(
            pack_train["X"].to(device), pack_train["y"].to(device),
            pack_val["X"].to(device), pack_val["y"].to(device),
            dim=pack_train["X"].shape[1], device=device,
        )
        preds = _predict(net, pack_test["X"].to(device), device=device)
        print(f"  {label} train+predict done")
        return preds

    v_oracle_pred = _fit_and_pred(V_oracle["train"], V_oracle["val"], V_oracle["test"], "visual_oracle")
    v_e2e_pred    = _fit_and_pred(V_e2e["train"],    V_e2e["val"],    V_e2e["test"],    "visual_e2e (per-claim)")
    t_oracle_pred = _fit_and_pred(T_oracle["train"], T_oracle["val"], T_oracle["test"], "text_oracle")
    t_e2e_pred    = _fit_and_pred(T_e2e["train"],    T_e2e["val"],    T_e2e["test"],    "text_e2e")

    # ---- Fusion MLPs: score + gated for the key combinations ----
    # We train two useful ones: fusion gated E2E+E2E (per-claim) and fusion gated oracle+oracle.
    def _fit_fusion(vis_pack, txt_pack, gated: bool, label: str):
        Xtr = torch.cat([vis_pack["train"]["X"], txt_pack["train"]["X"]], dim=1)
        ytr = vis_pack["train"]["y"]
        Xv  = torch.cat([vis_pack["val"]["X"],   txt_pack["val"]["X"]],   dim=1)
        yv  = vis_pack["val"]["y"]
        Xt  = torch.cat([vis_pack["test"]["X"],  txt_pack["test"]["X"]],  dim=1)
        net = _train_mlp(Xtr.to(device), ytr.to(device), Xv.to(device), yv.to(device),
                         dim=Xtr.shape[1], device=device)
        preds = _predict(net, Xt.to(device), device=device)
        # Also return probabilities for abstention.
        net.to(device).eval()
        with torch.no_grad():
            probs = torch.softmax(net(Xt.to(device)), dim=1).cpu().numpy()
            val_probs = torch.softmax(net(Xv.to(device)), dim=1).cpu().numpy()
        print(f"  {label} fusion train+predict done")
        return preds, probs, val_probs, yv.cpu().numpy()

    fus_per_claim_pred, fus_pc_probs, fus_pc_val_probs, fus_pc_val_labels = _fit_fusion(
        V_e2e, T_e2e, gated=False, label="fusion_per_claim_e2e+e2e"
    )
    fus_oracle_pred, _, _, _ = _fit_fusion(V_oracle, T_oracle, gated=False, label="fusion_oracle+oracle")

    # ---- Global E2E visual predictions for the "before" row ----
    # These come from the verification run's MLP E2E, which did global retrieval.
    probs_file = args.verify_run / "metrics" / "mlp_e2e_probs.json"
    if probs_file.exists():
        with open(probs_file, "r", encoding="utf-8") as f:
            pe = json.load(f)
        v_e2e_global_preds = np.argmax(np.asarray(pe["test"]["probs"]), axis=1)
        assert len(v_e2e_global_preds) == len(y_test), \
            f"global-E2E probs n={len(v_e2e_global_preds)} != test n={len(y_test)}"
    else:
        print(f"  WARN: {probs_file} not found — global-E2E row will be missing.")
        v_e2e_global_preds = None

    # ---- Cross-modality decomposition ----
    decomp = {
        "per_claim_e2e":    _cross_modality(y_test, v_e2e_pred, t_e2e_pred),
        "oracle":           _cross_modality(y_test, v_oracle_pred, t_oracle_pred),
    }
    if v_e2e_global_preds is not None:
        decomp["global_e2e"] = _cross_modality(y_test, v_e2e_global_preds, t_e2e_pred)

    print("\n=== Cross-modality decomposition (test split, N={}) ===".format(len(y_test)))
    for key, d in decomp.items():
        n = d["total"]
        print(f"  {key}:")
        for bucket in ("text_only", "visual_only", "both", "neither"):
            v = d[bucket]
            print(f"    {bucket:>12s}: {v:>5d}  ({100*v/n:5.1f}%)")

    # ---- Confusion matrices ----
    confusions = {
        "visual_e2e_per_claim":  _confusion(y_test, v_e2e_pred),
        "text_e2e":              _confusion(y_test, t_e2e_pred),
        "fusion_per_claim_e2e":  _confusion(y_test, fus_per_claim_pred),
    }
    if v_e2e_global_preds is not None:
        confusions["visual_e2e_global"] = _confusion(y_test, v_e2e_global_preds)

    # Apply abstention on the per-claim fusion probs and compute its confusion matrix too.
    abstained_preds, taus = _apply_abstention(
        fus_pc_probs, fus_pc_val_probs, fus_pc_val_labels
    )
    confusions["fusion_per_claim_e2e_abstained"] = _confusion(y_test, abstained_preds)
    print(f"\nAbstention thresholds chosen on val: {taus}")

    # ---- Save everything ----
    out = {
        "test_n": int(len(y_test)),
        "cross_modality": decomp,
        "confusion_matrices": confusions,
        "abstention_thresholds": taus,
    }
    _save(out, args.out_dir / "cross_modality_analysis.json")

    # Plot confusion matrix panel (before/after per-claim + after abstention).
    panel = []
    if v_e2e_global_preds is not None:
        panel.append(("Visual E2E (global, before)", np.asarray(confusions["visual_e2e_global"])))
    panel.append(("Visual E2E (per-claim, after)", np.asarray(confusions["visual_e2e_per_claim"])))
    panel.append(("Fusion gated E2E (per-claim)", np.asarray(confusions["fusion_per_claim_e2e"])))
    panel.append(("Fusion + abstention", np.asarray(confusions["fusion_per_claim_e2e_abstained"])))
    _plot_confusion_panel(panel, args.out_dir / "confusion_panel_before_after.png")

    print("\ndone.")


if __name__ == "__main__":
    main()
