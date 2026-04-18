"""Evaluate per-claim CLIP retrieval on WebQA-Adv.

Uses the enriched JSONL (image_candidate_ids present per claim) to rank
only within each claim's ~10 candidate images instead of the 390K-image
global pool. Reuses the existing global CLIP embedding index — no
re-embedding needed.

Reports Recall@K, average candidate-pool size, and the fraction of claims
whose gold image is present in the candidate pool (an oracle ceiling on
what per-claim retrieval can possibly recover).
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, set_seed, setup_logging, ensure_dirs, create_run_dir
from src.experiment_tracker import ExperimentTracker
from src.models.clip_encoder import CLIPEncoder
from src.data.webqa_dataset import WebQADataset
from src.retrieval import (
    EmbeddingIndex,
    CLIPRetriever,
    evaluate_retrieval,
    evaluate_retrieval_scoped,
)
from src.evaluation import save_metrics
from src.visualization import plot_recall_at_k


def main():
    parser = argparse.ArgumentParser(description="Run per-claim CLIP retrieval evaluation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--embeddings", type=str, default=None,
                        help="Path to pre-computed image_index.pt from a previous run")
    parser.add_argument("--also-global", action="store_true",
                        help="Also evaluate global-pool retrieval for side-by-side comparison")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    create_run_dir(cfg)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    set_seed(cfg.seed)
    ensure_dirs(cfg)

    tracker = ExperimentTracker("retrieval_per_claim", cfg.results.experiments_dir)
    tracker.log_config(cfg)

    try:
        dataset = WebQADataset(cfg)
        tracker.log_dataset(
            name="WebQA-Adv",
            total=len(dataset),
            train_size=len(dataset.get_split("train")),
            val_size=len(dataset.get_split("val")),
            test_size=len(dataset.get_split("test")),
            image_pool_size=len(dataset.get_all_image_ids()),
        )

        test_data = dataset.get_split("test")

        encoder = CLIPEncoder(cfg)

        index_path = args.embeddings or os.path.join(cfg.results.embeddings_dir, "image_index.pt")
        index = EmbeddingIndex()
        if os.path.exists(index_path):
            index.load(index_path)
        else:
            print("No pre-computed index found. Computing embeddings...")
            from src.retrieval import precompute_embeddings
            precompute_embeddings(encoder, dataset, index_path, batch_size=cfg.clip.batch_size)
            index.load(index_path)

        retriever = CLIPRetriever(encoder, index, cfg)

        # ---- Per-claim scoped retrieval ----
        scoped = evaluate_retrieval_scoped(retriever, test_data, cfg.retrieval.top_k_values)
        metrics_out = {
            "per_claim": scoped,
        }

        # ---- Optional: also run global retrieval so results tables show both ----
        if args.also_global:
            print("Running global retrieval for comparison...")
            global_recall = evaluate_retrieval(retriever, test_data, cfg.retrieval.top_k_values)
            metrics_out["global"] = {
                "recall_at_k": global_recall,
                "image_pool_size": len(dataset.get_all_image_ids()),
            }

        save_metrics(
            metrics_out,
            os.path.join(cfg.results.metrics_dir, "retrieval_per_claim_metrics.json"),
        )

        # Plot per-claim Recall@K
        plot_recall_at_k(
            scoped["recall_at_k"],
            os.path.join(cfg.results.figures_dir, "recall_at_k_per_claim.png"),
        )

        tracker_fields = {f"per_claim_recall@{k}": v for k, v in scoped["recall_at_k"].items()}
        tracker_fields["avg_pool_size"] = scoped["avg_pool_size"]
        tracker_fields["gold_in_pool_coverage"] = scoped["gold_in_pool_coverage"]
        tracker_fields["empty_pool_claims"] = scoped["empty_pool_claims"]
        if args.also_global:
            for k, v in metrics_out["global"]["recall_at_k"].items():
                tracker_fields[f"global_recall@{k}"] = v
        tracker.log_results(**tracker_fields)
        tracker.complete()

    except Exception as e:
        tracker.fail(str(e))
        raise

    print("=" * 50)
    print("Per-claim retrieval evaluation complete")
    for k, r in sorted(scoped["recall_at_k"].items()):
        print(f"  ScopedRecall@{k}: {r:.4f}")
    print(f"  avg pool size: {scoped['avg_pool_size']:.2f}")
    print(f"  gold-in-pool:  {scoped['gold_in_pool_coverage']:.4f}")
    if args.also_global:
        print("--- global retrieval (for comparison) ---")
        for k, r in sorted(metrics_out["global"]["recall_at_k"].items()):
            print(f"  GlobalRecall@{k}: {r:.4f}")


if __name__ == "__main__":
    main()
