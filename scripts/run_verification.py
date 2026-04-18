"""Run verification evaluation (threshold + MLP, oracle + E2E modes)."""

import argparse
import os
import sys
import logging

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, set_seed, setup_logging, ensure_dirs, create_run_dir
from src.experiment_tracker import ExperimentTracker
from src.models.clip_encoder import CLIPEncoder
from src.models.mlp_classifier import MLPClassifier
from src.models.threshold_classifier import ThresholdClassifier
from src.data.webqa_dataset import WebQADataset
from src.retrieval import EmbeddingIndex, CLIPRetriever
from src.verification import verify_oracle, verify_e2e, build_evidence_map_oracle, extract_features
from src.evaluation import (
    compute_verification_metrics,
    per_type_breakdown,
    oracle_vs_e2e_comparison,
    save_metrics,
)
from src.visualization import (
    plot_confusion_matrix,
    plot_oracle_vs_e2e_confusion,
    plot_per_class_bars,
    plot_oracle_vs_e2e_bars,
    plot_similarity_distributions,
    plot_per_type_breakdown,
)

logger = logging.getLogger(__name__)


def run_threshold_verification(cfg, encoder, dataset, retriever=None, embedding_index=None):
    """Run threshold-based verification in both oracle and E2E modes."""
    test_data = dataset.get_split("test")
    val_data = dataset.get_split("val")

    # Oracle features
    oracle_feats = verify_oracle(encoder, test_data, embedding_index=embedding_index)
    val_oracle_feats = verify_oracle(encoder, val_data, embedding_index=embedding_index)

    # Fit threshold on val set
    thresh_clf = ThresholdClassifier(cfg)
    val_max_sims = val_oracle_feats["features"][:, 0]  # max similarity
    thresh_clf.fit(val_max_sims, val_oracle_feats["labels"])

    # Oracle predictions
    oracle_max_sims = oracle_feats["features"][:, 0]
    oracle_preds = thresh_clf.predict(oracle_max_sims)
    oracle_metrics = compute_verification_metrics(oracle_feats["labels"], oracle_preds)
    oracle_breakdown = per_type_breakdown(oracle_feats["labels"], oracle_preds, oracle_feats["misinfo_types"])

    logger.info(f"Oracle threshold — Macro F1: {oracle_metrics['macro_f1']:.4f}")

    # Save oracle results
    save_metrics(oracle_metrics, os.path.join(cfg.results.metrics_dir, "threshold_oracle_metrics.json"))
    save_metrics(oracle_breakdown, os.path.join(cfg.results.metrics_dir, "threshold_oracle_breakdown.json"))

    # Plots
    plot_confusion_matrix(
        np.array(oracle_metrics["confusion_matrix"]),
        oracle_metrics["confusion_labels"],
        os.path.join(cfg.results.figures_dir, "threshold_oracle_cm.png"),
        title="Threshold Classifier — Oracle Evidence",
    )
    plot_similarity_distributions(
        oracle_max_sims, oracle_feats["labels"],
        {0: "True", 1: "False", 2: "Unverifiable"},
        os.path.join(cfg.results.figures_dir, "similarity_distributions.png"),
    )
    plot_per_class_bars(
        oracle_metrics["per_class"],
        os.path.join(cfg.results.figures_dir, "threshold_oracle_per_class.png"),
    )
    plot_per_type_breakdown(
        oracle_breakdown,
        os.path.join(cfg.results.figures_dir, "threshold_oracle_per_type.png"),
    )

    # E2E mode (if retriever available)
    e2e_metrics = None
    if retriever:
        e2e_feats = verify_e2e(encoder, retriever, test_data, cfg.retrieval.default_top_k,
                              embedding_index=embedding_index)
        e2e_max_sims = e2e_feats["features"][:, 0]
        e2e_preds = thresh_clf.predict(e2e_max_sims)
        e2e_metrics = compute_verification_metrics(e2e_feats["labels"], e2e_preds)
        e2e_breakdown = per_type_breakdown(e2e_feats["labels"], e2e_preds, e2e_feats["misinfo_types"])

        logger.info(f"E2E threshold — Macro F1: {e2e_metrics['macro_f1']:.4f}")

        save_metrics(e2e_metrics, os.path.join(cfg.results.metrics_dir, "threshold_e2e_metrics.json"))
        save_metrics(e2e_breakdown, os.path.join(cfg.results.metrics_dir, "threshold_e2e_breakdown.json"))

        plot_confusion_matrix(
            np.array(e2e_metrics["confusion_matrix"]),
            e2e_metrics["confusion_labels"],
            os.path.join(cfg.results.figures_dir, "threshold_e2e_cm.png"),
            title="Threshold Classifier — E2E Retrieved Evidence",
        )

        # Comparison
        comparison = oracle_vs_e2e_comparison(oracle_metrics, e2e_metrics)
        save_metrics(comparison, os.path.join(cfg.results.metrics_dir, "threshold_comparison.json"))

        plot_oracle_vs_e2e_confusion(
            np.array(oracle_metrics["confusion_matrix"]),
            np.array(e2e_metrics["confusion_matrix"]),
            oracle_metrics["confusion_labels"],
            os.path.join(cfg.results.figures_dir, "threshold_oracle_vs_e2e_cm.png"),
        )
        plot_oracle_vs_e2e_bars(
            comparison,
            os.path.join(cfg.results.figures_dir, "threshold_oracle_vs_e2e_bars.png"),
        )

    return oracle_metrics, e2e_metrics


def run_mlp_verification(cfg, encoder, dataset, retriever=None, checkpoint_override=None, embedding_index=None):
    """Run MLP-based verification in both oracle and E2E modes."""
    test_data = dataset.get_split("test")
    val_data = dataset.get_split("val")

    # Oracle features (test + val; val is needed for abstention calibration)
    oracle_feats = verify_oracle(encoder, test_data, embedding_index=embedding_index)
    val_oracle_feats = verify_oracle(encoder, val_data, embedding_index=embedding_index)

    # Load trained MLP
    device = torch.device(cfg.clip.device if torch.cuda.is_available() or cfg.clip.device == "cpu" else "cpu")
    model = MLPClassifier(cfg).to(device)

    checkpoint_path = checkpoint_override or os.path.join(cfg.results.checkpoints_dir, "mlp_best.pt")
    if not os.path.exists(checkpoint_path):
        logger.warning(f"No MLP checkpoint found at {checkpoint_path}. Run train_mlp.py first.")
        return None, None

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    # Predict (oracle)
    input_mode = cfg.verification.mlp.input_mode
    feat_key = "features" if input_mode == "features" else "embeddings"
    X = torch.tensor(oracle_feats[feat_key], dtype=torch.float32).to(device)
    X_val = torch.tensor(val_oracle_feats[feat_key], dtype=torch.float32).to(device)

    with torch.no_grad():
        oracle_logits = model(X)
        oracle_probs = torch.softmax(oracle_logits, dim=1).cpu().numpy()
        oracle_preds = oracle_logits.argmax(dim=1).cpu().numpy()
        val_oracle_logits = model(X_val)
        val_oracle_probs = torch.softmax(val_oracle_logits, dim=1).cpu().numpy()

    oracle_metrics = compute_verification_metrics(oracle_feats["labels"], oracle_preds)
    oracle_breakdown = per_type_breakdown(oracle_feats["labels"], oracle_preds, oracle_feats["misinfo_types"])

    # Save oracle probs for abstention ablation
    save_metrics(
        {
            "val": {
                "probs": val_oracle_probs.tolist(),
                "labels": val_oracle_feats["labels"].tolist(),
            },
            "test": {
                "probs": oracle_probs.tolist(),
                "labels": oracle_feats["labels"].tolist(),
            },
        },
        os.path.join(cfg.results.metrics_dir, "mlp_oracle_probs.json"),
    )

    logger.info(f"MLP Oracle — Macro F1: {oracle_metrics['macro_f1']:.4f}")

    save_metrics(oracle_metrics, os.path.join(cfg.results.metrics_dir, "mlp_oracle_metrics.json"))
    save_metrics(oracle_breakdown, os.path.join(cfg.results.metrics_dir, "mlp_oracle_breakdown.json"))

    plot_confusion_matrix(
        np.array(oracle_metrics["confusion_matrix"]),
        oracle_metrics["confusion_labels"],
        os.path.join(cfg.results.figures_dir, "mlp_oracle_cm.png"),
        title="MLP Classifier — Oracle Evidence",
    )
    plot_per_class_bars(
        oracle_metrics["per_class"],
        os.path.join(cfg.results.figures_dir, "mlp_oracle_per_class.png"),
    )
    plot_per_type_breakdown(
        oracle_breakdown,
        os.path.join(cfg.results.figures_dir, "mlp_oracle_per_type.png"),
    )

    # E2E mode
    e2e_metrics = None
    if retriever:
        e2e_feats = verify_e2e(encoder, retriever, test_data, cfg.retrieval.default_top_k,
                              embedding_index=embedding_index)
        val_e2e_feats = verify_e2e(encoder, retriever, val_data, cfg.retrieval.default_top_k,
                                   embedding_index=embedding_index)

        X_e2e = torch.tensor(e2e_feats[feat_key], dtype=torch.float32).to(device)
        X_val_e2e = torch.tensor(val_e2e_feats[feat_key], dtype=torch.float32).to(device)

        with torch.no_grad():
            e2e_logits = model(X_e2e)
            e2e_probs = torch.softmax(e2e_logits, dim=1).cpu().numpy()
            e2e_preds = e2e_logits.argmax(dim=1).cpu().numpy()
            val_e2e_probs = torch.softmax(model(X_val_e2e), dim=1).cpu().numpy()

        e2e_metrics = compute_verification_metrics(e2e_feats["labels"], e2e_preds)
        e2e_breakdown = per_type_breakdown(e2e_feats["labels"], e2e_preds, e2e_feats["misinfo_types"])

        # Save E2E probs for abstention ablation
        save_metrics(
            {
                "val": {
                    "probs": val_e2e_probs.tolist(),
                    "labels": val_e2e_feats["labels"].tolist(),
                },
                "test": {
                    "probs": e2e_probs.tolist(),
                    "labels": e2e_feats["labels"].tolist(),
                },
            },
            os.path.join(cfg.results.metrics_dir, "mlp_e2e_probs.json"),
        )

        logger.info(f"MLP E2E — Macro F1: {e2e_metrics['macro_f1']:.4f}")

        save_metrics(e2e_metrics, os.path.join(cfg.results.metrics_dir, "mlp_e2e_metrics.json"))
        save_metrics(e2e_breakdown, os.path.join(cfg.results.metrics_dir, "mlp_e2e_breakdown.json"))

        plot_confusion_matrix(
            np.array(e2e_metrics["confusion_matrix"]),
            e2e_metrics["confusion_labels"],
            os.path.join(cfg.results.figures_dir, "mlp_e2e_cm.png"),
            title="MLP Classifier — E2E Retrieved Evidence",
        )

        comparison = oracle_vs_e2e_comparison(oracle_metrics, e2e_metrics)
        save_metrics(comparison, os.path.join(cfg.results.metrics_dir, "mlp_comparison.json"))

        plot_oracle_vs_e2e_confusion(
            np.array(oracle_metrics["confusion_matrix"]),
            np.array(e2e_metrics["confusion_matrix"]),
            oracle_metrics["confusion_labels"],
            os.path.join(cfg.results.figures_dir, "mlp_oracle_vs_e2e_cm.png"),
        )
        plot_oracle_vs_e2e_bars(
            comparison,
            os.path.join(cfg.results.figures_dir, "mlp_oracle_vs_e2e_bars.png"),
        )

    return oracle_metrics, e2e_metrics


def main():
    parser = argparse.ArgumentParser(description="Run verification evaluation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--mode", choices=["threshold", "mlp", "both"], default="both")
    parser.add_argument("--embeddings", type=str, default=None,
                        help="Path to pre-computed image_index.pt from a previous run")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained mlp_best.pt from a previous run")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    create_run_dir(cfg)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    set_seed(cfg.seed)
    ensure_dirs(cfg)

    tracker = ExperimentTracker("verification_eval", cfg.results.experiments_dir)
    tracker.log_config(cfg)

    try:
        # Load dataset
        dataset = WebQADataset(cfg)
        tracker.log_dataset(
            name="WebQA-Adv",
            total=len(dataset),
            train_size=len(dataset.get_split("train")),
            val_size=len(dataset.get_split("val")),
            test_size=len(dataset.get_split("test")),
            image_pool_size=len(dataset.get_all_image_ids()),
        )

        encoder = CLIPEncoder(cfg)

        # Try to load retriever for E2E
        retriever = None
        embedding_index = None
        index_path = args.embeddings or os.path.join(cfg.results.embeddings_dir, "image_index.pt")
        if os.path.exists(index_path):
            embedding_index = EmbeddingIndex()
            embedding_index.load(index_path)
            retriever = CLIPRetriever(encoder, embedding_index, cfg)
        else:
            logger.warning("No embedding index found. Skipping E2E evaluation.")

        results = {}

        if args.mode in ("threshold", "both"):
            print("\n=== Threshold-based Verification ===")
            oracle, e2e = run_threshold_verification(cfg, encoder, dataset, retriever,
                                                     embedding_index=embedding_index)
            results["threshold_oracle_f1"] = oracle["macro_f1"]
            if e2e:
                results["threshold_e2e_f1"] = e2e["macro_f1"]
            tracker.log_step("threshold", oracle_f1=oracle["macro_f1"],
                             e2e_f1=e2e["macro_f1"] if e2e else None)

        if args.mode in ("mlp", "both"):
            print("\n=== MLP-based Verification ===")
            oracle, e2e = run_mlp_verification(cfg, encoder, dataset, retriever,
                                               checkpoint_override=args.checkpoint,
                                               embedding_index=embedding_index)
            if oracle:
                results["mlp_oracle_f1"] = oracle["macro_f1"]
                if e2e:
                    results["mlp_e2e_f1"] = e2e["macro_f1"]
                tracker.log_step("mlp", oracle_f1=oracle["macro_f1"],
                                 e2e_f1=e2e["macro_f1"] if e2e else None)

        tracker.log_results(**results)
        tracker.complete()

    except Exception as e:
        tracker.fail(str(e))
        raise


if __name__ == "__main__":
    main()
