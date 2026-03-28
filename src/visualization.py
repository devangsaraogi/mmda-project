import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Consistent style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _save_fig(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved figure: {path}")


def plot_recall_at_k(recall_dict: dict[int, float], save_path: str) -> None:
    """Line plot of Recall@K vs K."""
    ks = sorted(recall_dict.keys())
    recalls = [recall_dict[k] for k in ks]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, recalls, "o-", color="#2980B9", linewidth=2, markersize=8)
    for k, r in zip(ks, recalls):
        ax.annotate(f"{r:.3f}", (k, r), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xlabel("K (top-K retrieved)")
    ax.set_ylabel("Recall@K")
    ax.set_title("CLIP Retrieval: Recall@K")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ks)
    _save_fig(fig, save_path)


def plot_similarity_distributions(
    similarities: np.ndarray,
    labels: np.ndarray,
    label_names: dict,
    save_path: str,
) -> None:
    """Overlapping histograms of similarity scores by verdict label."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"True": "#27AE60", "False": "#E74C3C", "Unverifiable": "#F39C12"}

    for label_id in sorted(set(labels)):
        name = label_names.get(label_id, str(label_id))
        mask = labels == label_id
        ax.hist(
            similarities[mask],
            bins=30,
            alpha=0.5,
            label=f"{name} (n={mask.sum()})",
            color=colors.get(name, "#888"),
            density=True,
        )

    ax.set_xlabel("Max Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Similarity Score Distributions by Verdict")
    ax.legend()
    _save_fig(fig, save_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Seaborn heatmap confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    _save_fig(fig, save_path)


def plot_oracle_vs_e2e_confusion(
    oracle_cm: np.ndarray,
    e2e_cm: np.ndarray,
    labels: list[str],
    save_path: str,
) -> None:
    """Vertically stacked confusion matrices for oracle vs E2E."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))

    sns.heatmap(oracle_cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Oracle (Gold Evidence)")

    sns.heatmap(e2e_cm, annot=True, fmt="d", cmap="Oranges", xticklabels=labels, yticklabels=labels, ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title("End-to-End (Retrieved Evidence)")

    fig.suptitle("Oracle vs E2E Verification", fontsize=14)
    plt.tight_layout()
    _save_fig(fig, save_path)


def plot_per_class_bars(
    per_class_metrics: dict,
    save_path: str,
    metric_key: str = "f1",
    title: str = "Per-Class F1 Score",
) -> None:
    """Grouped bar chart of per-class F1 (or other metric)."""
    classes = list(per_class_metrics.keys())
    values = [per_class_metrics[c][metric_key] for c in classes]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#27AE60", "#E74C3C", "#F39C12"]
    bars = ax.bar(classes, values, color=colors[: len(classes)], edgecolor="gray", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", fontsize=10)

    ax.set_ylabel(metric_key.upper())
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    _save_fig(fig, save_path)


def plot_oracle_vs_e2e_bars(comparison: dict, save_path: str) -> None:
    """Comparison bar chart: oracle vs E2E metrics."""
    metrics = ["accuracy", "macro_f1"]
    oracle_vals = [comparison["oracle"][m] for m in metrics]
    e2e_vals = [comparison["e2e"][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width / 2, oracle_vals, width, label="Oracle", color="#2980B9", edgecolor="gray")
    bars2 = ax.bar(x + width / 2, e2e_vals, width, label="E2E", color="#E67E22", edgecolor="gray")

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center",
                fontsize=9,
            )

    ax.set_ylabel("Score")
    ax.set_title("Oracle vs End-to-End Verification")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Macro F1"])
    ax.set_ylim(0, 1.1)
    ax.legend()
    _save_fig(fig, save_path)


def plot_per_type_breakdown(breakdown: dict, save_path: str) -> None:
    """Bar chart showing F1 per misinformation type."""
    types = list(breakdown.keys())
    f1s = [breakdown[t]["f1"] for t in types]
    supports = [breakdown[t]["support"] for t in types]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette("Set2", len(types))
    bars = ax.bar(types, f1s, color=colors, edgecolor="gray", linewidth=0.5)

    for bar, val, sup in zip(bars, f1s, supports):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}\n(n={sup})",
            ha="center",
            fontsize=9,
        )

    ax.set_ylabel("Macro F1")
    ax.set_title("Verification F1 by Misinformation Type")
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=15)
    _save_fig(fig, save_path)


def plot_qualitative_examples(
    examples: list[dict],
    save_path: str,
    max_images: int = 3,
) -> None:
    """Grid showing claim text + retrieved images + scores.

    Each example dict should have:
        claim_text: str
        images: list[PIL.Image]
        scores: list[float]
        true_label: str
        pred_label: str
    """
    n = len(examples)
    fig, axes = plt.subplots(n, max_images + 1, figsize=(4 * (max_images + 1), 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, ex in enumerate(examples):
        # Claim text cell
        axes[i, 0].text(
            0.5, 0.5,
            f"Claim: {ex['claim_text'][:80]}...\n"
            f"True: {ex['true_label']}\n"
            f"Pred: {ex['pred_label']}",
            ha="center", va="center", fontsize=9, wrap=True,
            transform=axes[i, 0].transAxes,
        )
        axes[i, 0].set_xlim(0, 1)
        axes[i, 0].set_ylim(0, 1)
        axes[i, 0].axis("off")

        # Evidence images
        for j in range(max_images):
            ax = axes[i, j + 1]
            if j < len(ex.get("images", [])):
                ax.imshow(ex["images"][j])
                score = ex["scores"][j] if j < len(ex["scores"]) else 0
                ax.set_title(f"sim={score:.3f}", fontsize=9)
            ax.axis("off")

    plt.suptitle("Qualitative Examples", fontsize=13, y=1.01)
    plt.tight_layout()
    _save_fig(fig, save_path)
