"""Generate all report figures and tables from experiment results.

Loads metrics JSONs and error_analysis.json, produces publication-ready figures.
Outputs go to a specified directory (default: submissions/second_submission/figures/).
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Consistent style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "visual": "#2980B9",
    "text": "#E67E22",
    "fusion_score": "#27AE60",
    "fusion_gated": "#8E44AD",
    "oracle": "#2980B9",
    "e2e": "#E67E22",
}


def _save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ============================================================
# T1: Master Results Table (LaTeX)
# ============================================================
def generate_master_table(all_metrics, output_path):
    """Generate LaTeX table with all models x settings."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\begin{small}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Evidence} & \textbf{Acc} & \textbf{Macro F1} & \textbf{True F1} & \textbf{False F1} & \textbf{Unverif. F1} \\",
        r"\midrule",
    ]

    for key in sorted(all_metrics.keys()):
        if "_per_type" in key:
            continue
        m = all_metrics[key]
        pc = m.get("per_class", {})
        true_f1 = pc.get("True", {}).get("f1", 0)
        false_f1 = pc.get("False", {}).get("f1", 0)
        unverif_f1 = pc.get("Unverifiable", {}).get("f1", 0)

        # Parse model/evidence from key
        name = key.replace("_", r"\_")
        lines.append(
            f"{name} & & {m['accuracy']:.3f} & {m['macro_f1']:.3f} "
            f"& {true_f1:.3f} & {false_f1:.3f} & {unverif_f1:.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{small}",
        r"\caption{Complete results across all models and evaluation settings.}",
        r"\label{tab:master_results}",
        r"\end{table*}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {output_path}")


# ============================================================
# T2: Retrieval Comparison Table (LaTeX)
# ============================================================
def generate_retrieval_table(clip_recall, bm25_recall, output_path):
    """Generate LaTeX table comparing CLIP vs BM25 retrieval."""
    ks = sorted(set(list(clip_recall.keys()) + list(bm25_recall.keys())))
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{small}",
        r"\begin{tabular}{l" + "c" * len(ks) + "}",
        r"\toprule",
        r"\textbf{Retriever} & " + " & ".join(
            [rf"\textbf{{@{k}}}" for k in ks]
        ) + r" \\",
        r"\midrule",
    ]

    clip_vals = " & ".join([f"{clip_recall.get(k, clip_recall.get(str(k), 0)):.3f}" for k in ks])
    bm25_vals = " & ".join([f"{bm25_recall.get(k, bm25_recall.get(str(k), 0)):.3f}" for k in ks])

    lines.append(f"CLIP ViT-L/14 & {clip_vals} \\\\")
    lines.append(f"BM25 & {bm25_vals} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{small}",
        r"\caption{Retrieval Recall@K comparison: CLIP (visual) vs BM25 (text).}",
        r"\label{tab:retrieval}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {output_path}")


# ============================================================
# F1: Cross-Modality Stacked Bar Chart
# ============================================================
def plot_cross_modality_stacked(cross_modality, output_path):
    """Stacked bar chart: text-only, visual-only, both-correct, both-wrong by type."""
    per_type = cross_modality.get("per_type", {})
    if not per_type:
        print("  SKIP: No per_type cross-modality data")
        return

    types = sorted(per_type.keys())
    categories = ["text_only_correct", "visual_only_correct", "both_correct", "both_wrong"]
    cat_labels = ["Text-only correct", "Visual-only correct", "Both correct", "Both wrong"]
    cat_colors = ["#E67E22", "#2980B9", "#27AE60", "#E74C3C"]

    data = {cat: [] for cat in categories}
    for mtype in types:
        total = per_type[mtype]["total"]
        for cat in categories:
            data[cat].append(per_type[mtype][cat] / total * 100 if total > 0 else 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(types))
    bottom = np.zeros(len(types))

    for cat, label, color in zip(categories, cat_labels, cat_colors):
        vals = np.array(data[cat])
        ax.bar(x, vals, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.5)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=15, ha="right")
    ax.set_ylabel("Percentage of Claims (%)")
    ax.set_title("Cross-Modality Agreement by Misinformation Type")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 105)
    _save_fig(fig, output_path)


# ============================================================
# F2: Per-Type F1 Grouped Bar (Visual vs Text vs Fusion)
# ============================================================
def plot_per_type_f1_grouped(visual_per_type, text_per_type, fusion_per_type, output_path,
                              fusion_label="Fusion"):
    """Grouped bar chart: per-type F1 for visual, text, fusion."""
    all_types = sorted(set(
        list(visual_per_type.keys()) +
        list(text_per_type.keys()) +
        list(fusion_per_type.keys())
    ))

    vis_f1 = [visual_per_type.get(t, {}).get("f1", 0) for t in all_types]
    txt_f1 = [text_per_type.get(t, {}).get("f1", 0) for t in all_types]
    fus_f1 = [fusion_per_type.get(t, {}).get("f1", 0) for t in all_types]

    x = np.arange(len(all_types))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, vis_f1, w, label="Visual", color=COLORS["visual"], edgecolor="gray", linewidth=0.5)
    ax.bar(x, txt_f1, w, label="Text", color=COLORS["text"], edgecolor="gray", linewidth=0.5)
    ax.bar(x + w, fus_f1, w, label=fusion_label, color=COLORS["fusion_score"], edgecolor="gray", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(all_types, rotation=15, ha="right")
    ax.set_ylabel("Macro F1")
    ax.set_title("Per-Type F1: Visual vs Text vs Fusion (E2E)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    _save_fig(fig, output_path)


# ============================================================
# F3: 3-Panel Confusion Matrices (Visual, Text, Fusion)
# ============================================================
def plot_3panel_confusion(vis_cm, txt_cm, fus_cm, labels, output_path):
    """Side-by-side confusion matrices."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, cm, title in [
        (ax1, vis_cm, "Visual (E2E)"),
        (ax2, txt_cm, "Text (E2E)"),
        (ax3, fus_cm, "Fusion (E2E)"),
    ]:
        sns.heatmap(
            np.array(cm), annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)

    plt.suptitle("Confusion Matrices: E2E Setting", fontsize=14, y=1.02)
    plt.tight_layout()
    _save_fig(fig, output_path)


# ============================================================
# F4: Oracle vs E2E Degradation Bar Chart
# ============================================================
def plot_oracle_e2e_degradation(metrics_dict, output_path):
    """Bar chart showing F1 drop from oracle to E2E per model."""
    models = []
    oracle_f1s = []
    e2e_f1s = []

    # Expect keys like visual_oracle, visual_e2e, text_oracle, text_e2e, fusion_*
    seen = set()
    for key in sorted(metrics_dict.keys()):
        if "_per_type" in key:
            continue
        if "oracle" in key:
            base = key.replace("_oracle", "").replace("oracle", "")
            e2e_key = key.replace("oracle", "e2e")
            if e2e_key in metrics_dict and base not in seen:
                seen.add(base)
                models.append(base if base else "model")
                oracle_f1s.append(metrics_dict[key]["macro_f1"])
                e2e_f1s.append(metrics_dict[e2e_key]["macro_f1"])

    if not models:
        print("  SKIP: No oracle/e2e pairs found")
        return

    x = np.arange(len(models))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w/2, oracle_f1s, w, label="Oracle", color=COLORS["oracle"], edgecolor="gray")
    bars2 = ax.bar(x + w/2, e2e_f1s, w, label="E2E", color=COLORS["e2e"], edgecolor="gray")

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Macro F1")
    ax.set_title("Oracle vs E2E: Retrieval-Induced Performance Drop")
    ax.set_ylim(0, 1.05)
    ax.legend()
    _save_fig(fig, output_path)


# ============================================================
# T3: Success/Failure Cases Table (LaTeX)
# ============================================================
def generate_examples_table(examples_dict, output_path, n=3):
    """Generate LaTeX table with success/failure examples."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\begin{small}",
        r"\begin{tabular}{p{4.5cm}lllll}",
        r"\toprule",
        r"\textbf{Claim (truncated)} & \textbf{Type} & \textbf{Gold} & \textbf{Visual} & \textbf{Text} & \textbf{Fusion} \\",
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Success Cases}} \\",
        r"\midrule",
    ]

    # Gather success cases from fusion
    fusion_ex = examples_dict.get("fusion", {})
    for ex in fusion_ex.get("successes", [])[:n]:
        claim = ex["claim_text"][:60].replace("_", r"\_").replace("&", r"\&")
        lines.append(
            f"{claim}... & {ex['misinfo_type']} & {ex['gold_label']} "
            f"& -- & -- & {ex['predicted_label']} \\\\"
        )

    lines.extend([
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Failure Cases}} \\",
        r"\midrule",
    ])

    for ex in fusion_ex.get("failures", [])[:n]:
        claim = ex["claim_text"][:60].replace("_", r"\_").replace("&", r"\&")
        lines.append(
            f"{claim}... & {ex['misinfo_type']} & {ex['gold_label']} "
            f"& -- & -- & {ex['predicted_label']} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{small}",
        r"\caption{Representative success and failure cases from the fusion model (E2E).}",
        r"\label{tab:examples}",
        r"\end{table*}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate report figures from experiment results")
    parser.add_argument("--metrics-dir", type=str, required=True,
                        help="Directory containing metrics JSONs (e.g. results/metrics/)")
    parser.add_argument("--error-analysis", type=str, default=None,
                        help="Path to error_analysis.json")
    parser.add_argument("--output-dir", type=str, default="submissions/second_submission/figures",
                        help="Output directory for figures")
    parser.add_argument("--clip-recall-json", type=str, default=None,
                        help="Path to CLIP retrieval metrics JSON")
    parser.add_argument("--bm25-recall-json", type=str, default=None,
                        help="Path to BM25 retrieval metrics JSON")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tables_dir = os.path.join(args.output_dir, "..", "tables")
    os.makedirs(tables_dir, exist_ok=True)

    print("=" * 60)
    print("Generating report figures and tables")
    print("=" * 60)

    # Load all metrics from metrics directory
    all_metrics = {}
    if os.path.exists(os.path.join(args.metrics_dir, "all_metrics_summary.json")):
        all_metrics = load_json(os.path.join(args.metrics_dir, "all_metrics_summary.json"))
        print(f"Loaded all_metrics_summary.json: {len(all_metrics)} entries")

    # Load individual fusion metrics for per-type data
    per_type_data = {}
    for fname in os.listdir(args.metrics_dir):
        if fname.endswith(".json") and "per_type" not in fname:
            key = fname.replace(".json", "")
            data = load_json(os.path.join(args.metrics_dir, fname))
            if key not in all_metrics and "accuracy" in data:
                all_metrics[key] = data

    # Load error analysis
    error_analysis = None
    if args.error_analysis and os.path.exists(args.error_analysis):
        error_analysis = load_json(args.error_analysis)
        print(f"Loaded error_analysis.json")

    # ---- T1: Master Results Table ----
    if all_metrics:
        print("\n[T1] Master Results Table")
        generate_master_table(all_metrics, os.path.join(tables_dir, "master_results.tex"))

    # ---- T2: Retrieval Comparison ----
    clip_recall = {}
    bm25_recall = {}
    if args.clip_recall_json and os.path.exists(args.clip_recall_json):
        clip_data = load_json(args.clip_recall_json)
        clip_recall = clip_data.get("recall_at_k", clip_data)
    if args.bm25_recall_json and os.path.exists(args.bm25_recall_json):
        bm25_data = load_json(args.bm25_recall_json)
        bm25_recall = bm25_data.get("bm25_recall_at_k", bm25_data)

    if clip_recall or bm25_recall:
        print("\n[T2] Retrieval Comparison Table")
        # Normalize keys to int
        clip_recall = {int(k): v for k, v in clip_recall.items()}
        bm25_recall = {int(k): v for k, v in bm25_recall.items()}
        generate_retrieval_table(clip_recall, bm25_recall,
                                 os.path.join(tables_dir, "retrieval_comparison.tex"))

    # ---- F1: Cross-Modality Stacked Bar ----
    if error_analysis and "cross_modality" in error_analysis:
        print("\n[F1] Cross-Modality Stacked Bar")
        plot_cross_modality_stacked(
            error_analysis["cross_modality"],
            os.path.join(args.output_dir, "cross_modality_stacked.png"),
        )

    # ---- F2: Per-Type F1 Grouped Bar ----
    if error_analysis and "per_type_breakdown" in error_analysis:
        print("\n[F2] Per-Type F1 Grouped Bar")
        ptb = error_analysis["per_type_breakdown"]
        plot_per_type_f1_grouped(
            ptb.get("visual", {}),
            ptb.get("text", {}),
            ptb.get("fusion", {}),
            os.path.join(args.output_dir, "per_type_f1_grouped.png"),
        )

    # ---- F3: 3-Panel Confusion Matrices ----
    if error_analysis and "confusion_matrices" in error_analysis:
        print("\n[F3] 3-Panel Confusion Matrices")
        cms = error_analysis["confusion_matrices"]
        labels = cms.get("visual", {}).get("labels", ["True", "False", "Unverifiable"])
        plot_3panel_confusion(
            cms.get("visual", {}).get("matrix", []),
            cms.get("text", {}).get("matrix", []),
            cms.get("fusion", {}).get("matrix", []),
            labels,
            os.path.join(args.output_dir, "confusion_3panel.png"),
        )

    # ---- F4: Oracle vs E2E Degradation ----
    if all_metrics:
        print("\n[F4] Oracle vs E2E Degradation")
        plot_oracle_e2e_degradation(
            all_metrics,
            os.path.join(args.output_dir, "oracle_e2e_degradation.png"),
        )

    # ---- T3: Success/Failure Cases ----
    if error_analysis and "examples" in error_analysis:
        print("\n[T3] Success/Failure Cases Table")
        generate_examples_table(
            error_analysis["examples"],
            os.path.join(tables_dir, "examples.tex"),
        )

    print("\n" + "=" * 60)
    print("Done! All outputs in:")
    print(f"  Figures: {args.output_dir}")
    print(f"  Tables:  {tables_dir}")


if __name__ == "__main__":
    main()
