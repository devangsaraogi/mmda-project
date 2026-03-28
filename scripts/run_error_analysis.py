"""Error analysis script for multimodal misinformation detection.

Generates confusion matrices, per-type breakdowns, success/failure examples,
cross-modality analysis, and modality contribution analysis.
"""

import argparse
import json
import os
import sys
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, set_seed, setup_logging, ensure_dirs, create_run_dir
from src.data.webqa_dataset import WebQADataset
from src.evaluation import compute_verification_metrics, per_type_breakdown, LABEL_NAMES
from src.verification_multimodal import align_features_by_claim_id

logger = logging.getLogger(__name__)


def load_features(path, split=None):
    """Load features from .pt file, optionally selecting a split."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if split and split in data:
        return data[split]
    return data


def train_quick_mlp(X_train, y_train, X_val, y_val, class_weights, device, hidden=[256, 128]):
    """Train a quick MLP classifier, return best model."""
    dim = X_train.shape[1]
    layers = []
    prev = dim
    for h in hidden:
        layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)])
        prev = h
    layers.append(nn.Linear(prev, 3))
    model = nn.Sequential(*layers).to(device)

    crit = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(X_train.to(device), y_train.to(device)), batch_size=64, shuffle=True)

    best_f1, best_st, pat = 0, None, 0
    for _ in range(50):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(X_val.to(device)).argmax(1).cpu().numpy()
        vf = f1_score(y_val.numpy(), vp, average="macro", zero_division=0)
        if vf > best_f1:
            best_f1 = vf
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= 7:
                break
    if best_st:
        model.load_state_dict(best_st)
    return model.to(device)


def predict(model, X, device):
    """Run inference."""
    model.eval()
    with torch.no_grad():
        return model(X.to(device)).argmax(1).cpu().numpy()


def predict_fusion(model, Xv, Xt, device):
    """Run fusion inference."""
    model.eval()
    with torch.no_grad():
        return model(Xv.to(device), Xt.to(device)).argmax(1).cpu().numpy()


def extract_examples(
    claim_ids, labels, preds, misinfo_types, dataset,
    visual_features=None, text_features=None,
    n_success=5, n_failure=5,
):
    """Extract success and failure examples with metadata."""
    successes = []
    failures = []

    for i, (cid, label, pred, mtype) in enumerate(zip(claim_ids, labels, preds, misinfo_types)):
        item_data = None
        # Find the claim in dataset
        for j in range(len(dataset)):
            item = dataset[j]
            if item["claim_id"] == cid:
                item_data = item
                break

        example = {
            "claim_id": str(cid),
            "claim_text": item_data["claim_text"] if item_data else "N/A",
            "gold_label": LABEL_NAMES.get(int(label), str(label)),
            "predicted_label": LABEL_NAMES.get(int(pred), str(pred)),
            "misinfo_type": mtype,
            "gold_image_ids": item_data["gold_image_ids"][:3] if item_data else [],
            "gold_text_ids": item_data["gold_text_ids"][:3] if item_data else [],
        }

        if visual_features is not None:
            example["visual_features"] = [round(float(v), 4) for v in visual_features[i][:6]]
        if text_features is not None:
            example["text_features"] = [round(float(v), 4) for v in text_features[i][:6]]

        if label == pred and len(successes) < n_success:
            successes.append(example)
        elif label != pred and len(failures) < n_failure:
            failures.append(example)

        if len(successes) >= n_success and len(failures) >= n_failure:
            break

    return {"successes": successes, "failures": failures}


def cross_modality_analysis(
    visual_preds, text_preds, labels, claim_ids, misinfo_types,
):
    """Find cases where one modality succeeds and the other fails."""
    vis_correct = visual_preds == labels
    txt_correct = text_preds == labels

    # Text succeeds, visual fails
    txt_only = np.where(txt_correct & ~vis_correct)[0]
    # Visual succeeds, text fails
    vis_only = np.where(vis_correct & ~txt_correct)[0]
    # Both succeed
    both_correct = np.where(vis_correct & txt_correct)[0]
    # Both fail
    both_wrong = np.where(~vis_correct & ~txt_correct)[0]

    analysis = {
        "text_only_correct": len(txt_only),
        "visual_only_correct": len(vis_only),
        "both_correct": len(both_correct),
        "both_wrong": len(both_wrong),
        "total": len(labels),
    }

    # Per-type breakdown of cross-modality
    types = sorted(set(misinfo_types))
    per_type = {}
    for mtype in types:
        mask = np.array([t == mtype for t in misinfo_types])
        per_type[mtype] = {
            "text_only_correct": int(np.sum(mask & txt_correct & ~vis_correct)),
            "visual_only_correct": int(np.sum(mask & vis_correct & ~txt_correct)),
            "both_correct": int(np.sum(mask & vis_correct & txt_correct)),
            "both_wrong": int(np.sum(mask & ~vis_correct & ~txt_correct)),
            "total": int(np.sum(mask)),
        }
    analysis["per_type"] = per_type

    return analysis


def class_frequency_analysis(labels, preds, misinfo_types):
    """Analyze relationship between class frequency and performance."""
    unique_labels = sorted(set(labels))
    result = {}
    for label in unique_labels:
        mask = labels == label
        support = int(np.sum(mask))
        if support == 0:
            continue
        correct = int(np.sum(mask & (preds == labels)))
        result[LABEL_NAMES.get(label, str(label))] = {
            "support": support,
            "frequency": round(support / len(labels), 4),
            "accuracy": round(correct / support, 4),
        }
    return result


def modality_contribution_analysis(
    visual_metrics_per_type, text_metrics_per_type, fusion_metrics_per_type,
):
    """Analyze which modality contributes most per misinformation type."""
    all_types = sorted(
        set(list(visual_metrics_per_type.keys()) +
            list(text_metrics_per_type.keys()) +
            list(fusion_metrics_per_type.keys()))
    )
    result = {}
    for mtype in all_types:
        vis_f1 = visual_metrics_per_type.get(mtype, {}).get("f1", 0)
        txt_f1 = text_metrics_per_type.get(mtype, {}).get("f1", 0)
        fus_f1 = fusion_metrics_per_type.get(mtype, {}).get("f1", 0)

        better_modality = "text" if txt_f1 > vis_f1 else "visual"
        fusion_gain = fus_f1 - max(vis_f1, txt_f1)

        result[mtype] = {
            "visual_f1": round(vis_f1, 4),
            "text_f1": round(txt_f1, 4),
            "fusion_f1": round(fus_f1, 4),
            "better_single_modality": better_modality,
            "fusion_gain_over_best_single": round(fusion_gain, 4),
        }
    return result


def generate_markdown_summary(analysis_results, output_path):
    """Generate a markdown summary of the error analysis."""
    lines = ["# Error Analysis Report\n"]

    # Overall metrics comparison
    if "metrics_comparison" in analysis_results:
        lines.append("## Model Comparison\n")
        lines.append("| Model | Accuracy | Macro F1 |")
        lines.append("|-------|----------|----------|")
        for name, m in analysis_results["metrics_comparison"].items():
            lines.append(f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} |")
        lines.append("")

    # Cross-modality
    if "cross_modality" in analysis_results:
        cm = analysis_results["cross_modality"]
        lines.append("## Cross-Modality Analysis\n")
        lines.append(f"- Text-only correct: {cm['text_only_correct']}")
        lines.append(f"- Visual-only correct: {cm['visual_only_correct']}")
        lines.append(f"- Both correct: {cm['both_correct']}")
        lines.append(f"- Both wrong: {cm['both_wrong']}")
        lines.append(f"- Total: {cm['total']}")
        lines.append("")

        if "per_type" in cm:
            lines.append("### Per Misinformation Type\n")
            lines.append("| Type | Text-only | Visual-only | Both OK | Both Wrong | Total |")
            lines.append("|------|-----------|-------------|---------|------------|-------|")
            for mtype, vals in cm["per_type"].items():
                lines.append(
                    f"| {mtype} | {vals['text_only_correct']} | {vals['visual_only_correct']} "
                    f"| {vals['both_correct']} | {vals['both_wrong']} | {vals['total']} |"
                )
            lines.append("")

    # Modality contribution
    if "modality_contribution" in analysis_results:
        lines.append("## Modality Contribution per Type\n")
        lines.append("| Type | Visual F1 | Text F1 | Fusion F1 | Better Modality | Fusion Gain |")
        lines.append("|------|-----------|---------|-----------|-----------------|-------------|")
        for mtype, vals in analysis_results["modality_contribution"].items():
            lines.append(
                f"| {mtype} | {vals['visual_f1']:.4f} | {vals['text_f1']:.4f} | {vals['fusion_f1']:.4f} "
                f"| {vals['better_single_modality']} | {vals['fusion_gain_over_best_single']:+.4f} |"
            )
        lines.append("")

    # Class frequency
    if "class_frequency" in analysis_results:
        lines.append("## Class Frequency vs Performance\n")
        lines.append("| Class | Support | Frequency | Accuracy |")
        lines.append("|-------|---------|-----------|----------|")
        for cls, vals in analysis_results["class_frequency"].items():
            lines.append(f"| {cls} | {vals['support']} | {vals['frequency']:.4f} | {vals['accuracy']:.4f} |")
        lines.append("")

    # Failure examples
    if "examples" in analysis_results:
        for model_name, exs in analysis_results["examples"].items():
            lines.append(f"## Examples: {model_name}\n")
            lines.append("### Successes\n")
            for ex in exs.get("successes", []):
                lines.append(f"- **{ex['claim_id']}** [{ex['misinfo_type']}]: {ex['claim_text'][:100]}...")
                lines.append(f"  - Label: {ex['gold_label']} | Pred: {ex['predicted_label']}")
            lines.append("\n### Failures\n")
            for ex in exs.get("failures", []):
                lines.append(f"- **{ex['claim_id']}** [{ex['misinfo_type']}]: {ex['claim_text'][:100]}...")
                lines.append(f"  - Label: {ex['gold_label']} | Pred: {ex['predicted_label']}")
            lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Markdown summary saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Error analysis for multimodal pipeline")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--visual-features", type=str, required=True,
                        help="Path to visual_features.pt (or run dir)")
    parser.add_argument("--text-features", type=str, required=True,
                        help="Path to text_features.pt (or run dir)")
    parser.add_argument("--mode", type=str, default="e2e",
                        choices=["oracle", "e2e"],
                        help="Primary evaluation mode for error analysis")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    run_dir = create_run_dir(cfg)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    set_seed(cfg.seed)
    ensure_dirs(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = args.mode

    # Load dataset
    dataset = WebQADataset(cfg)
    test_data = dataset.get_split("test")

    # Load features
    visual_all = torch.load(args.visual_features, map_location="cpu", weights_only=False)
    text_all = torch.load(args.text_features, map_location="cpu", weights_only=False)

    # Get features for the chosen mode
    vis_train = visual_all.get(mode, visual_all).get("train", visual_all.get("train", {}))
    vis_val = visual_all.get(mode, visual_all).get("val", visual_all.get("val", {}))
    vis_test = visual_all.get(mode, visual_all).get("test", visual_all.get("test", {}))
    txt_train = text_all.get(mode, text_all).get("train", text_all.get("train", {}))
    txt_val = text_all.get(mode, text_all).get("val", text_all.get("val", {}))
    txt_test = text_all.get(mode, text_all).get("test", text_all.get("test", {}))

    class_weights = list(cfg.verification.mlp.class_weights)

    # Train individual models
    logger.info("Training visual-only MLP...")
    X_vis_tr = torch.tensor(vis_train["features"], dtype=torch.float32)
    y_tr = torch.tensor(vis_train["labels"], dtype=torch.long)
    X_vis_val = torch.tensor(vis_val["features"], dtype=torch.float32)
    y_val = torch.tensor(vis_val["labels"], dtype=torch.long)
    X_vis_te = torch.tensor(vis_test["features"], dtype=torch.float32)
    y_te = np.array(vis_test["labels"])

    visual_mlp = train_quick_mlp(X_vis_tr, y_tr, X_vis_val, y_val, class_weights, device)
    visual_preds = predict(visual_mlp, X_vis_te, device)

    logger.info("Training text-only MLP...")
    X_txt_tr = torch.tensor(txt_train["features"], dtype=torch.float32)
    X_txt_val = torch.tensor(txt_val["features"], dtype=torch.float32)
    X_txt_te = torch.tensor(txt_test["features"], dtype=torch.float32)

    text_mlp = train_quick_mlp(X_txt_tr, y_tr, X_txt_val, y_val, class_weights, device)
    text_preds = predict(text_mlp, X_txt_te, device)

    logger.info("Training fusion MLP...")
    # Align
    test_aligned = align_features_by_claim_id(vis_test, txt_test)
    train_aligned = align_features_by_claim_id(vis_train, txt_train)
    val_aligned = align_features_by_claim_id(vis_val, txt_val)

    from src.models.fusion_classifier import FusionMLP
    vis_dim = train_aligned["visual_features"].shape[1]
    txt_dim = train_aligned["text_features"].shape[1]

    fusion_model = FusionMLP(visual_dim=vis_dim, text_dim=txt_dim, hidden_dims=[256, 128]).to(device)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    opt = torch.optim.Adam(fusion_model.parameters(), lr=0.001, weight_decay=1e-4)

    Xv_tr = torch.tensor(train_aligned["visual_features"], dtype=torch.float32)
    Xt_tr = torch.tensor(train_aligned["text_features"], dtype=torch.float32)
    y_tr_a = torch.tensor(train_aligned["labels"], dtype=torch.long)
    Xv_val = torch.tensor(val_aligned["visual_features"], dtype=torch.float32)
    Xt_val = torch.tensor(val_aligned["text_features"], dtype=torch.float32)
    y_val_a = torch.tensor(val_aligned["labels"], dtype=torch.long)

    loader = DataLoader(TensorDataset(Xv_tr, Xt_tr, y_tr_a), batch_size=64, shuffle=True)
    best_f1, best_st, pat = 0, None, 0
    for _ in range(50):
        fusion_model.train()
        for vb, tb, yb in loader:
            vb, tb, yb = vb.to(device), tb.to(device), yb.to(device)
            opt.zero_grad()
            crit(fusion_model(vb, tb), yb).backward()
            opt.step()
        fusion_model.eval()
        with torch.no_grad():
            vp = fusion_model(Xv_val.to(device), Xt_val.to(device)).argmax(1).cpu().numpy()
        vf = f1_score(y_val_a.numpy(), vp, average="macro", zero_division=0)
        if vf > best_f1:
            best_f1 = vf
            best_st = {k: v.cpu().clone() for k, v in fusion_model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= 7:
                break
    if best_st:
        fusion_model.load_state_dict(best_st)
    fusion_model.to(device)

    Xv_te = torch.tensor(test_aligned["visual_features"], dtype=torch.float32)
    Xt_te = torch.tensor(test_aligned["text_features"], dtype=torch.float32)
    y_te_a = np.array(test_aligned["labels"])
    fusion_preds = predict_fusion(fusion_model, Xv_te, Xt_te, device)

    # === Compute all metrics ===
    logger.info("Computing metrics...")
    vis_metrics = compute_verification_metrics(y_te, visual_preds)
    txt_metrics = compute_verification_metrics(y_te, text_preds)
    fus_metrics = compute_verification_metrics(y_te_a, fusion_preds)

    vis_type_bd = per_type_breakdown(y_te, visual_preds, vis_test["misinfo_types"])
    txt_type_bd = per_type_breakdown(y_te, text_preds, txt_test["misinfo_types"])
    fus_type_bd = per_type_breakdown(y_te_a, fusion_preds, test_aligned["misinfo_types"])

    # === Build analysis results ===
    analysis = {}

    # 1. Metrics comparison
    analysis["metrics_comparison"] = {
        f"visual_{mode}": {"accuracy": vis_metrics["accuracy"], "macro_f1": vis_metrics["macro_f1"]},
        f"text_{mode}": {"accuracy": txt_metrics["accuracy"], "macro_f1": txt_metrics["macro_f1"]},
        f"fusion_{mode}": {"accuracy": fus_metrics["accuracy"], "macro_f1": fus_metrics["macro_f1"]},
    }

    # 2. Per-type breakdown
    analysis["per_type_breakdown"] = {
        "visual": vis_type_bd,
        "text": txt_type_bd,
        "fusion": fus_type_bd,
    }

    # 3. Cross-modality analysis
    analysis["cross_modality"] = cross_modality_analysis(
        visual_preds, text_preds, y_te,
        vis_test["claim_ids"], vis_test["misinfo_types"],
    )

    # 4. Class frequency analysis
    analysis["class_frequency"] = class_frequency_analysis(y_te_a, fusion_preds, test_aligned["misinfo_types"])

    # 5. Modality contribution
    analysis["modality_contribution"] = modality_contribution_analysis(
        vis_type_bd, txt_type_bd, fus_type_bd,
    )

    # 6. Examples
    analysis["examples"] = {}
    for name, preds_arr, feats_v, feats_t, ids, labels_arr, types in [
        ("visual", visual_preds, vis_test["features"], None, vis_test["claim_ids"], y_te, vis_test["misinfo_types"]),
        ("text", text_preds, None, txt_test["features"], txt_test["claim_ids"], y_te, txt_test["misinfo_types"]),
        ("fusion", fusion_preds, test_aligned.get("visual_features"), test_aligned.get("text_features"),
         test_aligned["claim_ids"], y_te_a, test_aligned["misinfo_types"]),
    ]:
        analysis["examples"][name] = extract_examples(
            ids, labels_arr, preds_arr, types, test_data,
            visual_features=feats_v, text_features=feats_t,
        )

    # 7. Confusion matrices (already in metrics)
    analysis["confusion_matrices"] = {
        "visual": {"matrix": vis_metrics["confusion_matrix"], "labels": vis_metrics["confusion_labels"]},
        "text": {"matrix": txt_metrics["confusion_matrix"], "labels": txt_metrics["confusion_labels"]},
        "fusion": {"matrix": fus_metrics["confusion_matrix"], "labels": fus_metrics["confusion_labels"]},
    }

    # === Save outputs ===
    os.makedirs(cfg.results.metrics_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(cfg.results.metrics_dir, "error_analysis.json")
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    logger.info("Error analysis JSON saved to %s", json_path)

    # Markdown
    md_path = os.path.join(cfg.results.metrics_dir, "error_analysis.md")
    generate_markdown_summary(analysis, md_path)

    # Confusion matrix plots
    from src.visualization import plot_confusion_matrix
    for name, cm_data in analysis["confusion_matrices"].items():
        plot_confusion_matrix(
            cm_data["matrix"], cm_data["labels"],
            os.path.join(cfg.results.figures_dir, f"error_analysis_{name}_cm.png"),
            title=f"Error Analysis: {name} ({mode})",
        )

    # Print summary
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nMode: {mode}")
    print(f"\nMetrics Comparison:")
    for name, m in analysis["metrics_comparison"].items():
        print(f"  {name:<30} Acc: {m['accuracy']:.4f}  F1: {m['macro_f1']:.4f}")

    cm = analysis["cross_modality"]
    print(f"\nCross-Modality (N={cm['total']}):")
    print(f"  Text-only correct:   {cm['text_only_correct']}")
    print(f"  Visual-only correct: {cm['visual_only_correct']}")
    print(f"  Both correct:        {cm['both_correct']}")
    print(f"  Both wrong:          {cm['both_wrong']}")

    print(f"\nOutputs saved to: {cfg.results.metrics_dir}")
    print(f"Figures saved to: {cfg.results.figures_dir}")


if __name__ == "__main__":
    main()
