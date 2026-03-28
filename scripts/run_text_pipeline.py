"""Text-only pipeline: BM25 retrieval + NLI verification."""

import argparse
import os
import sys
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, set_seed, setup_logging, ensure_dirs, create_run_dir
from src.experiment_tracker import ExperimentTracker
from src.data.webqa_dataset import WebQADataset
from src.retrieval_text import BM25Retriever
from src.verification_text import (
    NLIVerifier,
    build_text_evidence_map_oracle,
    build_text_evidence_map_bm25,
)
from src.evaluation import compute_verification_metrics, per_type_breakdown, save_metrics
from src.visualization import plot_confusion_matrix, plot_per_type_breakdown

logger = logging.getLogger(__name__)


def train_text_mlp(X_train, y_train, X_val, y_val, cfg, device):
    """Train a small MLP on NLI features for text-only verdict prediction."""
    input_dim = X_train.shape[1]
    hidden_dims = list(cfg.get("fusion", {}).get("hidden_dims", [256, 128]))

    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)])
        prev = h
    layers.append(nn.Linear(prev, 3))
    model = nn.Sequential(*layers).to(device)

    class_weights = torch.tensor(
        list(cfg.verification.mlp.class_weights), dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    train_loader = DataLoader(
        TensorDataset(X_train.to(device), y_train.to(device)),
        batch_size=64, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val.to(device), y_val.to(device)),
        batch_size=64,
    )

    best_f1 = 0.0
    best_state = None
    patience = 0
    max_patience = cfg.verification.mlp.early_stopping_patience

    for epoch in range(cfg.verification.mlp.num_epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.extend(model(xb).argmax(1).cpu().numpy())
                labels.extend(yb.cpu().numpy())
        vf1 = f1_score(labels, preds, average="macro", zero_division=0)

        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                logger.info("Text MLP early stop at epoch %d (best val F1 %.4f)", epoch + 1, best_f1)
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)
    return model, best_f1


def evaluate_with_mlp(model, X, y, device):
    """Run MLP inference and return predictions."""
    model.eval()
    loader = DataLoader(TensorDataset(X.to(device), y.to(device)), batch_size=256)
    preds = []
    with torch.no_grad():
        for xb, yb in loader:
            preds.extend(model(xb).argmax(1).cpu().numpy())
    return np.array(preds)


def evaluate_with_threshold(features, labels):
    """Simple threshold classifier on NLI features.

    Uses max_entailment and max_contradiction to classify:
        - entailment > contradiction → True (0)
        - contradiction > entailment → False (1)
        - else → Unverifiable (2)
    """
    max_ent = features[:, 0]  # max_entailment
    max_con = features[:, 2]  # max_contradiction
    diff = features[:, 5]     # entailment_minus_contradiction

    preds = np.full(len(labels), 2)  # default: Unverifiable
    preds[diff > 0.3] = 0   # True
    preds[diff < -0.3] = 1  # False
    return preds


def main():
    parser = argparse.ArgumentParser(description="Text-only pipeline (BM25 + NLI)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--nli-scores", type=str, default=None,
                        help="Path to pre-computed NLI scores (.pt file)")
    parser.add_argument("--debug", action="store_true",
                        help="Run on small subset (100 claims)")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    run_dir = create_run_dir(cfg)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    set_seed(cfg.seed)
    ensure_dirs(cfg)

    tracker = ExperimentTracker("text_pipeline", cfg.results.experiments_dir)
    tracker.log_config(cfg)

    try:
        # === Step 1: Load dataset ===
        print("\n" + "=" * 60)
        print("STEP 1: Loading dataset")
        print("=" * 60)

        dataset = WebQADataset(cfg)
        print(f"  Total claims: {len(dataset)}")

        tracker.log_dataset(
            name="WebQA-Adv",
            total=len(dataset),
            train_size=len(dataset.get_split("train")),
            val_size=len(dataset.get_split("val")),
            test_size=len(dataset.get_split("test")),
            image_pool_size=len(dataset.get_all_image_ids()),
        )

        # === Step 2: BM25 Retrieval ===
        print("\n" + "=" * 60)
        print("STEP 2: BM25 Text Retrieval")
        print("=" * 60)

        text_cfg = cfg.get("text", {})
        bm25_cfg = text_cfg.get("bm25", {})
        top_k_values = list(bm25_cfg.get("top_k_values", [1, 3, 5, 10]))
        default_top_k = int(bm25_cfg.get("default_top_k", 5))

        bm25 = BM25Retriever()
        recall_results = bm25.evaluate_recall(dataset, split="test", top_k_values=top_k_values)

        for k, r in sorted(recall_results.items()):
            print(f"  BM25 Recall@{k}: {r:.4f}")

        save_metrics(
            {"bm25_recall_at_k": recall_results},
            os.path.join(cfg.results.metrics_dir, "text_retrieval_metrics.json"),
        )
        tracker.log_step("bm25_retrieval", **{f"recall@{k}": v for k, v in recall_results.items()})

        # === Step 3: NLI Verification ===
        print("\n" + "=" * 60)
        print("STEP 3: NLI Text Verification")
        print("=" * 60)

        nli_cfg = text_cfg.get("nli", {})
        nli_model_name = str(nli_cfg.get("model_name", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"))
        nli_batch_size = int(nli_cfg.get("batch_size", 32))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        nli = NLIVerifier(nli_model_name, device=device, batch_size=nli_batch_size)

        # --- Oracle mode ---
        print("\n--- Oracle mode (gold text evidence) ---")
        for split_name in ["train", "val", "test"]:
            oracle_evidence = build_text_evidence_map_oracle(dataset, split=split_name)
            oracle_results = nli.extract_features_batch(
                dataset, oracle_evidence, split=split_name, top_k=default_top_k,
            )

            if split_name == "train":
                train_oracle = oracle_results
            elif split_name == "val":
                val_oracle = oracle_results
            else:
                test_oracle = oracle_results

        # --- E2E mode (test only — train/val reuse oracle features) ---
        print("\n--- E2E mode (BM25 retrieved text evidence, test split only) ---")
        bm25_evidence = build_text_evidence_map_bm25(
            bm25, dataset, split="test", top_k=default_top_k,
        )
        test_e2e = nli.extract_features_batch(
            dataset, bm25_evidence, split="test", top_k=default_top_k,
        )
        # Reuse oracle features for train/val (MLP trains on oracle, evaluates on E2E)
        train_e2e = train_oracle
        val_e2e = val_oracle

        # Cache NLI scores
        cache_path = os.path.join(run_dir, "nli_scores.pt")
        torch.save({
            "oracle": {s: {k: v for k, v in r.items() if k != "raw_scores"}
                       for s, r in [("train", train_oracle), ("val", val_oracle), ("test", test_oracle)]},
            "e2e": {s: {k: v for k, v in r.items() if k != "raw_scores"}
                    for s, r in [("train", train_e2e), ("val", val_e2e), ("test", test_e2e)]},
        }, cache_path)
        logger.info("NLI scores cached to %s", cache_path)

        # === Step 4: Classify ===
        print("\n" + "=" * 60)
        print("STEP 4: Text-Only Classification")
        print("=" * 60)

        torch_device = torch.device(device)
        all_text_metrics = {}

        for mode_name, train_data, val_data, test_data in [
            ("oracle", train_oracle, val_oracle, test_oracle),
            ("e2e", train_e2e, val_e2e, test_e2e),
        ]:
            print(f"\n--- {mode_name.upper()} mode ---")

            X_train = torch.tensor(train_data["features"], dtype=torch.float32)
            y_train = torch.tensor(train_data["labels"], dtype=torch.long)
            X_val = torch.tensor(val_data["features"], dtype=torch.float32)
            y_val = torch.tensor(val_data["labels"], dtype=torch.long)
            X_test = torch.tensor(test_data["features"], dtype=torch.float32)
            y_test = torch.tensor(test_data["labels"], dtype=torch.long)

            # Threshold classifier
            thresh_preds = evaluate_with_threshold(test_data["features"], test_data["labels"])
            thresh_metrics = compute_verification_metrics(test_data["labels"], thresh_preds)
            print(f"  Threshold — Acc: {thresh_metrics['accuracy']:.4f}, F1: {thresh_metrics['macro_f1']:.4f}")

            # MLP classifier
            model, best_val_f1 = train_text_mlp(X_train, y_train, X_val, y_val, cfg, torch_device)
            mlp_preds = evaluate_with_mlp(model, X_test, y_test, torch_device)
            mlp_metrics = compute_verification_metrics(test_data["labels"], mlp_preds)
            print(f"  MLP — Acc: {mlp_metrics['accuracy']:.4f}, F1: {mlp_metrics['macro_f1']:.4f}")

            # Per-type breakdown
            mlp_type_breakdown = per_type_breakdown(test_data["labels"], mlp_preds, test_data["misinfo_types"])

            # Save
            all_text_metrics[f"text_{mode_name}_threshold"] = thresh_metrics
            all_text_metrics[f"text_{mode_name}_mlp"] = mlp_metrics
            all_text_metrics[f"text_{mode_name}_mlp_per_type"] = mlp_type_breakdown

            save_metrics(
                mlp_metrics,
                os.path.join(cfg.results.metrics_dir, f"text_{mode_name}_mlp_metrics.json"),
            )
            save_metrics(
                thresh_metrics,
                os.path.join(cfg.results.metrics_dir, f"text_{mode_name}_threshold_metrics.json"),
            )

            # Visualization
            plot_confusion_matrix(
                mlp_metrics["confusion_matrix"],
                mlp_metrics["confusion_labels"],
                os.path.join(cfg.results.figures_dir, f"text_{mode_name}_mlp_cm.png"),
                title=f"Text {mode_name.upper()} MLP",
            )
            plot_per_type_breakdown(
                mlp_type_breakdown,
                os.path.join(cfg.results.figures_dir, f"text_{mode_name}_per_type.png"),
            )

            # Save checkpoint
            ckpt_path = os.path.join(cfg.results.checkpoints_dir, f"text_mlp_{mode_name}.pt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

            # Save features for fusion
            feat_path = os.path.join(run_dir, f"text_features_{mode_name}.pt")
            torch.save({
                "train": {"features": train_data["features"], "labels": train_data["labels"],
                          "claim_ids": train_data["claim_ids"], "misinfo_types": train_data["misinfo_types"]},
                "val": {"features": val_data["features"], "labels": val_data["labels"],
                        "claim_ids": val_data["claim_ids"], "misinfo_types": val_data["misinfo_types"]},
                "test": {"features": test_data["features"], "labels": test_data["labels"],
                         "claim_ids": test_data["claim_ids"], "misinfo_types": test_data["misinfo_types"]},
            }, feat_path)

        # === Summary ===
        print("\n" + "=" * 60)
        print("TEXT PIPELINE COMPLETE — Summary")
        print("=" * 60)

        print("\nBM25 Retrieval:")
        for k, r in sorted(recall_results.items()):
            print(f"  Recall@{k}: {r:.4f}")

        print("\nText Verification (MLP):")
        for key in ["text_oracle_mlp", "text_e2e_mlp"]:
            m = all_text_metrics[key]
            label = key.replace("text_", "").replace("_mlp", "").upper()
            print(f"  {label} — Acc: {m['accuracy']:.4f}, F1: {m['macro_f1']:.4f}")

        # Final tracker
        tracker.log_results(
            bm25_recall_at_5=recall_results.get(5, 0),
            text_oracle_mlp_f1=all_text_metrics["text_oracle_mlp"]["macro_f1"],
            text_e2e_mlp_f1=all_text_metrics["text_e2e_mlp"]["macro_f1"],
            text_oracle_thresh_f1=all_text_metrics["text_oracle_threshold"]["macro_f1"],
            text_e2e_thresh_f1=all_text_metrics["text_e2e_threshold"]["macro_f1"],
        )
        tracker.complete()
        tracker.print_summary()

    except Exception as e:
        tracker.fail(str(e))
        raise


if __name__ == "__main__":
    main()
