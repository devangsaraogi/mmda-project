"""Evaluate caption-augmented hybrid retrieval on WebQA-Adv.

Adds a third retrieval row to the paper's ablation table:

    Row 1  Global CLIP                 — 389,740-image pool (midterm)
    Row 2  Per-claim CLIP              — ranks ~12 per-claim candidates
    Row 3  Per-claim CLIP + caption-BM25 (RRF) ← THIS SCRIPT

Uses the enriched JSONL produced by ``enrich_webqa_adv_candidates.py``
(now also carries ``image_candidate_metadata`` with title + caption per
candidate). No new model weights, no re-embedding.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, set_seed, setup_logging, ensure_dirs, create_run_dir
from src.experiment_tracker import ExperimentTracker
from src.models.clip_encoder import CLIPEncoder
from src.data.webqa_dataset import WebQADataset
from src.retrieval import EmbeddingIndex, CLIPRetriever
from src.retrieval_hybrid import (
    CaptionBM25Retriever,
    RRFHybridRetriever,
    evaluate_retrieval_hybrid,
)
from src.evaluation import save_metrics
from src.visualization import plot_recall_at_k


def main():
    parser = argparse.ArgumentParser(description="Caption-augmented hybrid retrieval eval")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--embeddings", type=str, default=None,
                        help="Path to pre-computed image_index.pt from a previous run")
    parser.add_argument("--k-rrf", type=int, default=60,
                        help="RRF k (Cormack et al. 2009 default: 60)")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    create_run_dir(cfg)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    set_seed(cfg.seed)
    ensure_dirs(cfg)

    tracker = ExperimentTracker("retrieval_hybrid", cfg.results.experiments_dir)
    tracker.log_config(cfg)

    try:
        dataset = WebQADataset(cfg)
        test_data = dataset.get_split("test")
        tracker.log_dataset(
            name="WebQA-Adv",
            total=len(dataset),
            train_size=len(dataset.get_split("train")),
            val_size=len(dataset.get_split("val")),
            test_size=len(test_data),
            image_pool_size=len(dataset.get_all_image_ids()),
        )

        encoder = CLIPEncoder(cfg)

        index_path = args.embeddings or os.path.join(cfg.results.embeddings_dir, "image_index.pt")
        index = EmbeddingIndex()
        if not os.path.exists(index_path):
            print(f"ERROR: no pre-computed CLIP index at {index_path}")
            sys.exit(2)
        index.load(index_path)

        clip_retriever = CLIPRetriever(encoder, index, cfg)
        caption_retriever = CaptionBM25Retriever(default_top_k=cfg.retrieval.default_top_k)
        hybrid = RRFHybridRetriever(clip_retriever, caption_retriever, k_rrf=args.k_rrf)

        metrics = evaluate_retrieval_hybrid(hybrid, test_data, cfg.retrieval.top_k_values)
        metrics["k_rrf"] = args.k_rrf

        save_metrics(
            {"hybrid": metrics},
            os.path.join(cfg.results.metrics_dir, "retrieval_hybrid_metrics.json"),
        )

        plot_recall_at_k(
            metrics["recall_at_k"],
            os.path.join(cfg.results.figures_dir, "recall_at_k_hybrid.png"),
        )

        tracker.log_results(
            **{f"hybrid_recall@{k}": v for k, v in metrics["recall_at_k"].items()},
            **{f"hybrid_recall_in_pool@{k}": v for k, v in metrics["recall_at_k_gold_in_pool"].items()},
            k_rrf=args.k_rrf,
            n_has_gold=metrics["n_has_gold"],
            n_gold_in_pool=metrics["n_gold_in_pool"],
        )
        tracker.complete()

    except Exception as e:
        tracker.fail(str(e))
        raise

    print("=" * 60)
    print(f"Hybrid retrieval (RRF, k_rrf={args.k_rrf}) | "
          f"n_total={metrics['n_total']} n_has_gold={metrics['n_has_gold']} "
          f"n_in_pool={metrics['n_gold_in_pool']}")
    print(f"Avg pool size: {metrics['avg_pool_size']:.2f}")
    print("-" * 60)
    print(f"{'K':>4}  {'hybrid (all)':>14}  {'hybrid (has-gold)':>20}  {'hybrid (in-pool)':>20}")
    for k in sorted(metrics["recall_at_k"].keys()):
        print(f"{k:>4}  {metrics['recall_at_k'][k]:>14.4f}  "
              f"{metrics['recall_at_k_has_gold'][k]:>20.4f}  "
              f"{metrics['recall_at_k_gold_in_pool'][k]:>20.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
