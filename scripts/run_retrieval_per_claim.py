"""Evaluate per-claim CLIP retrieval on WebQA-Adv.

Uses the enriched JSONL (image_candidate_ids present per claim) to rank
only within each claim's ~10 candidate images instead of the 390K-image
global pool. Reuses the existing global CLIP embedding index — no
re-embedding needed.

Reports Recall@K sliced three ways:
  * over all test claims,
  * over claims that have at least one gold image (excludes Unverifiable),
  * over claims whose gold is actually in the candidate pool (the ceiling).

Comparable slices are produced for the global-pool baseline.
"""
from __future__ import annotations

import argparse
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, set_seed, setup_logging, ensure_dirs, create_run_dir
from src.experiment_tracker import ExperimentTracker
from src.models.clip_encoder import CLIPEncoder
from src.data.webqa_dataset import WebQADataset
from src.retrieval import (
    EmbeddingIndex,
    CLIPRetriever,
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

        # ---- Optional: also run global retrieval on the SAME claim subsets
        #      so per-claim vs global is apples-to-apples ----
        if args.also_global:
            print("Running global retrieval for comparison...")
            n = len(test_data)
            max_k = max(cfg.retrieval.top_k_values)

            claim_texts = [test_data[i]["claim_text"] for i in range(n)]
            all_results = retriever.retrieve_batch(claim_texts, top_k=max_k)

            # Build the same masks used by evaluate_retrieval_scoped so the
            # recall subsets match up between global and per-claim.
            has_gold_mask: list[bool] = []
            gold_in_pool_mask: list[bool] = []
            gold_sets: list[set[str]] = []
            for i in range(n):
                item = test_data[i]
                pool = {str(c) for c in item.get("image_candidate_ids", []) if c is not None}
                gold = {str(g) for g in item["gold_image_ids"]}
                gold_sets.append(gold)
                has_gold_mask.append(bool(gold))
                gold_in_pool_mask.append(bool(gold) and bool(gold & pool))

            global_recall_all = {k: [] for k in cfg.retrieval.top_k_values}
            global_recall_has_gold = {k: [] for k in cfg.retrieval.top_k_values}
            global_recall_in_pool = {k: [] for k in cfg.retrieval.top_k_values}
            for i in range(n):
                results = all_results[i]
                gold = gold_sets[i]
                for k in cfg.retrieval.top_k_values:
                    top_k_ids = {r[0] for r in results[:k]}
                    hit = bool(gold) and len(top_k_ids & gold) > 0
                    hit_f = float(hit)
                    global_recall_all[k].append(hit_f)
                    if has_gold_mask[i]:
                        global_recall_has_gold[k].append(hit_f)
                    if gold_in_pool_mask[i]:
                        global_recall_in_pool[k].append(hit_f)

            def _mean(seq):
                return float(np.mean(seq)) if seq else 0.0

            metrics_out["global"] = {
                "recall_at_k": {k: _mean(s) for k, s in global_recall_all.items()},
                "recall_at_k_has_gold": {k: _mean(s) for k, s in global_recall_has_gold.items()},
                "recall_at_k_gold_in_pool": {k: _mean(s) for k, s in global_recall_in_pool.items()},
                "image_pool_size": len(dataset.get_all_image_ids()),
                "n_total": n,
                "n_has_gold": int(sum(has_gold_mask)),
                "n_gold_in_pool": int(sum(gold_in_pool_mask)),
                # Per-claim hit indicators for bootstrap CIs + per-type breakdown.
                "per_claim_hits": {str(k): [int(v) for v in global_recall_all[k]] for k in cfg.retrieval.top_k_values},
                "has_gold_mask": [int(v) for v in has_gold_mask],
                "gold_in_pool_mask": [int(v) for v in gold_in_pool_mask],
                "claim_ids": [test_data[i]["claim_id"] for i in range(n)],
                "misinfo_types": [test_data[i]["misinfo_type"] for i in range(n)],
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
        tracker_fields.update(
            {f"per_claim_recall_in_pool@{k}": v for k, v in scoped["recall_at_k_gold_in_pool"].items()}
        )
        tracker_fields["avg_pool_size"] = scoped["avg_pool_size"]
        tracker_fields["gold_in_pool_coverage"] = scoped["gold_in_pool_coverage"]
        tracker_fields["n_has_gold"] = scoped["n_has_gold"]
        tracker_fields["n_gold_in_pool"] = scoped["n_gold_in_pool"]
        if args.also_global:
            for k, v in metrics_out["global"]["recall_at_k"].items():
                tracker_fields[f"global_recall@{k}"] = v
            for k, v in metrics_out["global"]["recall_at_k_gold_in_pool"].items():
                tracker_fields[f"global_recall_in_pool@{k}"] = v
        tracker.log_results(**tracker_fields)
        tracker.complete()

    except Exception as e:
        tracker.fail(str(e))
        raise

    print("=" * 60)
    print(f"Per-claim retrieval | n_total={scoped['n_total']} "
          f"n_has_gold={scoped['n_has_gold']} n_in_pool={scoped['n_gold_in_pool']}")
    print(f"Avg pool size: {scoped['avg_pool_size']:.2f}  "
          f"Gold-in-pool coverage: {scoped['gold_in_pool_coverage']:.4f}")
    print("-" * 60)
    print(f"{'K':>4}  {'per-claim (all)':>18}  {'per-claim (in-pool)':>22}" +
          ("  {:>18}".format("global (all)") if args.also_global else "") +
          ("  {:>22}".format("global (in-pool)") if args.also_global else ""))
    for k in sorted(scoped["recall_at_k"].keys()):
        row = f"{k:>4}  {scoped['recall_at_k'][k]:>18.4f}  {scoped['recall_at_k_gold_in_pool'][k]:>22.4f}"
        if args.also_global:
            row += f"  {metrics_out['global']['recall_at_k'][k]:>18.4f}"
            row += f"  {metrics_out['global']['recall_at_k_gold_in_pool'][k]:>22.4f}"
        print(row)
    print("=" * 60)


if __name__ == "__main__":
    main()
