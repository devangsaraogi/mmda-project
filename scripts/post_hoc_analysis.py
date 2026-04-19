"""Post-hoc analyses against existing metrics files.

Three analyses, each written to a single output file so they compose cleanly:

  1. Per-manipulation-type breakdown of retrieval + fusion.
  2. Bootstrap 95% CIs on R@K and Macro F1 for the headline rows.
  3. Abstention coverage-F1 curve (threshold sweep).

Runs on the cluster against the metrics JSON files produced by earlier
jobs. Nothing GPU-side — pure numpy + sklearn + matplotlib.

Usage:
    python scripts/post_hoc_analysis.py \\
        --retrieval-run results/runs/NVIDIA_A100-SXM4-40GB_20260418_142853 \\
        --fusion-run   results/runs/NVIDIA_A10_20260418_200021 \\
        --verify-run   results/runs/NVIDIA_A10_20260418_190246 \\
        --out-dir      results/post_hoc

Outputs:
    <out-dir>/per_type_retrieval.json          per-type Recall@K (global + per-claim)
    <out-dir>/per_type_fusion.json             per-type F1 per fusion variant
    <out-dir>/bootstrap_cis.json               95% CIs for R@K and F1
    <out-dir>/abstention_curve.json            threshold sweep: (tau, coverage, F1s)
    <out-dir>/per_type_recall.png              per-type Recall@K bar chart
    <out-dir>/abstention_curve.png             coverage-F1 plot
    <out-dir>/summary.md                       human-readable summary
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Headless matplotlib (cluster has no DISPLAY).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# =============================================================================
# Helpers
# =============================================================================

def _load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote {path}")


# =============================================================================
# Analysis 1 — Per-type retrieval breakdown
# =============================================================================

def per_type_retrieval(metrics: dict) -> dict:
    """Compute per-manipulation-type Recall@K using the per_claim_hits arrays.

    Requires the metrics file to carry the new per_claim_hits / misinfo_types
    fields (added by the post-hoc patch to run_retrieval_per_claim.py).
    Returns {retriever_name: {type: {K: recall}}}.
    """
    out: dict = {}
    for retriever in ("global", "per_claim"):
        block = metrics.get(retriever)
        if not block or "per_claim_hits" not in block:
            print(f"  skipping '{retriever}' — no per_claim_hits in metrics "
                  "(re-run run_retrieval_per_claim.py with the post-hoc patch)")
            continue
        types = block["misinfo_types"]
        per_type = defaultdict(lambda: defaultdict(list))  # type -> K -> [hits]
        for k_str, hits in block["per_claim_hits"].items():
            for i, h in enumerate(hits):
                per_type[types[i]][int(k_str)].append(h)
        out[retriever] = {
            t: {k: float(np.mean(v)) for k, v in k_to_hits.items()}
            for t, k_to_hits in per_type.items()
        }
        out[f"{retriever}_support"] = {
            t: len(next(iter(k_to_hits.values())))
            for t, k_to_hits in per_type.items()
        }
    return out


# =============================================================================
# Analysis 2 — Per-type fusion F1 (from fusion_*_breakdown.json files)
# =============================================================================

def per_type_fusion(fusion_run: Path) -> dict:
    """Collect per-type F1 from fusion_*_breakdown.json files."""
    metrics_dir = fusion_run / "metrics"
    out: dict = {}
    for path in sorted(metrics_dir.glob("fusion_*_breakdown.json")):
        variant = path.stem.replace("fusion_", "").replace("_breakdown", "")
        data = _load(path)
        # File shape: {misinfo_type: {accuracy, f1, support}}
        out[variant] = {
            t: {
                "f1": v.get("f1"),
                "accuracy": v.get("accuracy"),
                "support": v.get("support"),
            }
            for t, v in data.items()
        }
    return out


# =============================================================================
# Analysis 3 — Bootstrap 95% CIs
# =============================================================================

def _bootstrap_mean(values, n_boot=1000, rng=None) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) for a 0/1 array via bootstrap percentile CIs."""
    values = np.asarray(values, dtype=float)
    rng = rng or np.random.default_rng(42)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = values[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def _bootstrap_macro_f1(
    labels: np.ndarray,
    preds: np.ndarray,
    n_boot: int = 1000,
    rng=None,
) -> tuple[float, float, float]:
    from sklearn.metrics import f1_score
    rng = rng or np.random.default_rng(42)
    n = len(labels)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = f1_score(labels[idx], preds[idx], average="macro", zero_division=0)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    base = float(f1_score(labels, preds, average="macro", zero_division=0))
    return base, float(lo), float(hi)


def bootstrap_cis(metrics: dict, probs_e2e: dict | None) -> dict:
    """Bootstrap CIs for retrieval R@K (if hits present) and for F1 (if probs present)."""
    out: dict = {}
    rng = np.random.default_rng(42)

    # Retrieval bootstrap
    for retriever in ("global", "per_claim"):
        block = metrics.get(retriever)
        if not block or "per_claim_hits" not in block:
            continue
        ret_out = {}
        for k_str, hits in block["per_claim_hits"].items():
            m, lo, hi = _bootstrap_mean(hits, rng=rng)
            ret_out[f"R@{k_str}"] = {"mean": m, "ci95_lo": lo, "ci95_hi": hi}
        # Also do the gold-in-pool-only subset.
        in_pool_mask = np.asarray(block["gold_in_pool_mask"], dtype=bool)
        if in_pool_mask.sum() > 0:
            for k_str, hits in block["per_claim_hits"].items():
                hits_in_pool = np.asarray(hits)[in_pool_mask]
                m, lo, hi = _bootstrap_mean(hits_in_pool, rng=rng)
                ret_out[f"R@{k_str}_in_pool"] = {"mean": m, "ci95_lo": lo, "ci95_hi": hi}
        out[f"retrieval_{retriever}"] = ret_out

    # F1 bootstrap (from probs)
    if probs_e2e:
        test_probs = np.asarray(probs_e2e["test"]["probs"])
        test_labels = np.asarray(probs_e2e["test"]["labels"])
        # Uncalibrated
        preds = np.argmax(test_probs, axis=1)
        base, lo, hi = _bootstrap_macro_f1(test_labels, preds, rng=rng)
        out["macro_f1_e2e_uncal"] = {"mean": base, "ci95_lo": lo, "ci95_hi": hi}
        # Abstention-calibrated (using previously-saved thresholds if possible,
        # else re-calibrate on val here)
        val_probs = np.asarray(probs_e2e["val"]["probs"])
        val_labels = np.asarray(probs_e2e["val"]["labels"])
        from src.abstention import calibrate_abstention, apply_abstention
        cfg = calibrate_abstention(val_probs, val_labels, mode="per_class")
        fixed = apply_abstention(test_probs, cfg)
        base, lo, hi = _bootstrap_macro_f1(test_labels, fixed, rng=rng)
        out["macro_f1_e2e_abstained"] = {"mean": base, "ci95_lo": lo, "ci95_hi": hi}
        out["abstention_thresholds"] = cfg.class_thresholds

    return out


# =============================================================================
# Analysis 4 — Abstention coverage-F1 curve
# =============================================================================

def abstention_curve(probs_e2e: dict) -> dict:
    """Sweep the True-class threshold; report coverage, Macro F1, Unverif F1."""
    from sklearn.metrics import f1_score

    TRUE, FALSE, UNVERIF = 0, 1, 2
    val_probs = np.asarray(probs_e2e["val"]["probs"])
    val_labels = np.asarray(probs_e2e["val"]["labels"])
    test_probs = np.asarray(probs_e2e["test"]["probs"])
    test_labels = np.asarray(probs_e2e["test"]["labels"])

    taus = np.linspace(0.0, 0.95, 20)
    points = []

    base_preds = np.argmax(test_probs, axis=1)
    base_conf = np.max(test_probs, axis=1)

    for tau in taus:
        preds = base_preds.copy()
        # Remap only low-confidence True predictions (matches the asymmetric fix).
        remap_mask = (preds == TRUE) & (base_conf < tau)
        preds[remap_mask] = UNVERIF
        remapped_frac = float(remap_mask.mean())

        macro_f1 = float(f1_score(test_labels, preds, average="macro", zero_division=0))
        unverif_f1 = float(f1_score(
            test_labels == UNVERIF,
            preds == UNVERIF,
            average="binary",
            zero_division=0,
        ))
        points.append({
            "tau_true": float(tau),
            "remapped_fraction": remapped_frac,
            "macro_f1": macro_f1,
            "unverif_f1": unverif_f1,
        })
    return {"points": points}


# =============================================================================
# Plots
# =============================================================================

def plot_per_type_recall(per_type: dict, out_path: Path) -> None:
    if not per_type or "global" not in per_type or "per_claim" not in per_type:
        return
    types = sorted({t for r in ("global", "per_claim") for t in per_type.get(r, {}).keys()})
    # Use @5 for the bars.
    def rk(retriever, t):
        return per_type.get(retriever, {}).get(t, {}).get(5, 0.0)
    global_vals = [rk("global", t) for t in types]
    perclaim_vals = [rk("per_claim", t) for t in types]

    x = np.arange(len(types))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, global_vals, width, label="Global CLIP", color="#888")
    ax.bar(x + width / 2, perclaim_vals, width, label="Per-claim CLIP", color="#D6EAF8",
           edgecolor="#555")
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=25, ha="right")
    ax.set_ylabel("Recall@5")
    ax.set_title("Per-manipulation-type retrieval Recall@5")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_abstention_curve(curve: dict, out_path: Path) -> None:
    pts = curve["points"]
    taus = [p["tau_true"] for p in pts]
    macro = [p["macro_f1"] for p in pts]
    unverif = [p["unverif_f1"] for p in pts]
    remapped = [p["remapped_fraction"] for p in pts]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(taus, macro, marker="o", label="Macro F1", color="#1a2a4a")
    ax1.plot(taus, unverif, marker="s", label="Unverifiable F1", color="#c0392b")
    ax1.set_xlabel(r"$\tau_{\mathrm{True}}$ (abstention threshold)")
    ax1.set_ylabel("F1")
    ax1.set_ylim(0, 0.5)
    ax1.grid(linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(taus, remapped, marker="^", linestyle="--",
             label="Fraction remapped", color="#888")
    ax2.set_ylabel("Fraction remapped to Unverifiable")
    ax2.set_ylim(0, 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title("Abstention threshold sweep (E2E test split)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_path}")


# =============================================================================
# Summary writer
# =============================================================================

def write_summary(
    out_dir: Path,
    per_type_ret: dict,
    per_type_fus: dict,
    cis: dict,
    curve: dict,
) -> None:
    lines: list[str] = []
    lines.append("# Post-hoc analysis summary")
    lines.append("")
    if per_type_ret:
        lines.append("## Per-manipulation-type Recall@5")
        lines.append("")
        lines.append("| Type | Global CLIP | Per-claim CLIP | Δ |")
        lines.append("|---|---|---|---|")
        types = sorted({
            t for r in ("global", "per_claim") for t in per_type_ret.get(r, {})
        })
        for t in types:
            g = per_type_ret.get("global", {}).get(t, {}).get(5, 0.0)
            p = per_type_ret.get("per_claim", {}).get(t, {}).get(5, 0.0)
            lines.append(f"| {t} | {g:.4f} | {p:.4f} | {p - g:+.4f} |")
        lines.append("")
    if per_type_fus:
        lines.append("## Per-type fusion F1 (per-claim run, top variants)")
        lines.append("")
        keep = {k for k in per_type_fus if "gated" in k and "oracle" not in k.split("+")[0].split("_")[-1:]}
        # Keep it readable — show all rows.
        lines.append("| Variant | " + " | ".join(sorted({
            t for v in per_type_fus.values() for t in v.keys()
        })) + " |")
        types = sorted({t for v in per_type_fus.values() for t in v.keys()})
        lines.append("|---" * (len(types) + 1) + "|")
        for variant in sorted(per_type_fus.keys()):
            row = [variant]
            for t in types:
                f1 = per_type_fus[variant].get(t, {}).get("f1")
                row.append(f"{f1:.3f}" if f1 is not None else "-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    if cis:
        lines.append("## Bootstrap 95% CIs (1000 resamples)")
        lines.append("")
        lines.append("| Metric | Mean | 95% CI |")
        lines.append("|---|---|---|")
        for k, v in cis.items():
            if isinstance(v, dict) and "mean" in v:
                lines.append(f"| {k} | {v['mean']:.4f} | [{v['ci95_lo']:.4f}, {v['ci95_hi']:.4f}] |")
            elif isinstance(v, dict):
                for inner_k, inner_v in v.items():
                    lines.append(f"| {k}.{inner_k} | {inner_v['mean']:.4f} | [{inner_v['ci95_lo']:.4f}, {inner_v['ci95_hi']:.4f}] |")
        lines.append("")
    if curve and "points" in curve:
        lines.append("## Abstention coverage-F1 curve (excerpt)")
        lines.append("")
        lines.append("| τ_True | Fraction remapped | Macro F1 | Unverif F1 |")
        lines.append("|---|---|---|---|")
        for p in curve["points"]:
            lines.append(
                f"| {p['tau_true']:.3f} | {p['remapped_fraction']:.3f} | "
                f"{p['macro_f1']:.4f} | {p['unverif_f1']:.4f} |"
            )
        lines.append("")

    out_path = out_dir / "summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retrieval-run", type=Path, required=True)
    ap.add_argument("--fusion-run", type=Path, required=True)
    ap.add_argument("--verify-run", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/post_hoc"))
    args = ap.parse_args()

    # Make the src package importable when invoked from the project root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics.
    retrieval_metrics_path = args.retrieval_run / "metrics" / "retrieval_per_claim_metrics.json"
    probs_path = args.verify_run / "metrics" / "mlp_e2e_probs.json"

    retrieval_metrics = _load(retrieval_metrics_path) if retrieval_metrics_path.exists() else {}
    probs_e2e = _load(probs_path) if probs_path.exists() else None

    print("=== Per-type retrieval breakdown ===")
    per_type_ret = per_type_retrieval(retrieval_metrics)
    _save(per_type_ret, args.out_dir / "per_type_retrieval.json")

    print("=== Per-type fusion breakdown ===")
    per_type_fus = per_type_fusion(args.fusion_run)
    _save(per_type_fus, args.out_dir / "per_type_fusion.json")

    print("=== Bootstrap 95% CIs ===")
    cis = bootstrap_cis(retrieval_metrics, probs_e2e)
    _save(cis, args.out_dir / "bootstrap_cis.json")

    print("=== Abstention coverage-F1 curve ===")
    curve = abstention_curve(probs_e2e) if probs_e2e else {}
    _save(curve, args.out_dir / "abstention_curve.json")

    print("=== Plots ===")
    if per_type_ret:
        plot_per_type_recall(per_type_ret, args.out_dir / "per_type_recall.png")
    if curve:
        plot_abstention_curve(curve, args.out_dir / "abstention_curve.png")

    print("=== Summary ===")
    write_summary(args.out_dir, per_type_ret, per_type_fus, cis, curve)
    print("done.")


if __name__ == "__main__":
    main()
