"""Per-claim retrieval with SigLIP 2 instead of CLIP — encoder ablation.

Only embeds the images that actually appear in a test-split candidate
pool (about 27K unique IDs for the 7,392 test claims, not the full 390K).
Runs scoped retrieval with the same subset slicing as the CLIP pipeline,
so the comparison to ``scripts/run_retrieval_per_claim.py`` is
apples-to-apples.

Output: ``results/runs/<GPU>_<TS>/metrics/retrieval_siglip_per_claim_metrics.json``
with the same schema as the CLIP version (recall_at_k +
recall_at_k_has_gold + recall_at_k_gold_in_pool + per_claim_hits).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, set_seed, setup_logging, ensure_dirs, create_run_dir
from src.experiment_tracker import ExperimentTracker
from src.models.siglip_encoder import SigLIPEncoder
from src.data.webqa_dataset import WebQADataset
from src.retrieval import EmbeddingIndex, CLIPRetriever, evaluate_retrieval_scoped
from src.evaluation import save_metrics
from src.visualization import plot_recall_at_k


def _collect_candidate_ids(dataset) -> list[str]:
    """Return the union of image_candidate_ids across the given (sub)dataset."""
    seen: set[str] = set()
    out: list[str] = []
    for i in range(len(dataset)):
        for iid in dataset[i].get("image_candidate_ids", []):
            s = str(iid)
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
    return out


def _embed_image_ids(
    encoder: SigLIPEncoder,
    dataset,
    image_ids: list[str],
    batch_size: int = 64,
) -> tuple[torch.Tensor, list[str]]:
    """Load images for each ID via the dataset, embed with SigLIP."""
    tensors = []
    valid_ids = []
    batch_pil = []
    batch_valid_ids = []
    total = len(image_ids)
    log_every = max(1, total // 20)

    for idx, iid in enumerate(image_ids):
        img = dataset.get_image(iid)
        if img is None:
            continue
        batch_pil.append(img)
        batch_valid_ids.append(iid)
        if len(batch_pil) >= batch_size:
            emb = encoder.encode_images(batch_pil, batch_size=batch_size)
            tensors.append(emb)
            valid_ids.extend(batch_valid_ids)
            batch_pil, batch_valid_ids = [], []
        if (idx + 1) % log_every == 0:
            print(f"  SigLIP encoded {idx + 1}/{total} candidates", flush=True)
    if batch_pil:
        emb = encoder.encode_images(batch_pil, batch_size=batch_size)
        tensors.append(emb)
        valid_ids.extend(batch_valid_ids)

    if not tensors:
        return torch.empty(0, encoder.embed_dim), []
    return torch.cat(tensors, dim=0), valid_ids


def main():
    ap = argparse.ArgumentParser(description="Per-claim retrieval with SigLIP 2")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--model", type=str, default="google/siglip2-base-patch16-256",
                    help="HuggingFace SigLIP checkpoint identifier")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    create_run_dir(cfg)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    set_seed(cfg.seed)
    ensure_dirs(cfg)

    tracker = ExperimentTracker("retrieval_siglip_per_claim", cfg.results.experiments_dir)
    tracker.log_config(cfg)

    try:
        # Dataset — we only need the test split for evaluation.
        dataset = WebQADataset(cfg)
        test_data = dataset.get_split("test")

        # Unique candidate IDs across the test split.
        cand_ids = _collect_candidate_ids(test_data)
        print(f"Collected {len(cand_ids)} unique candidate image IDs "
              f"across the {len(test_data)} test claims.")

        # Load SigLIP and embed those candidates.
        encoder = SigLIPEncoder(cfg, model_name=args.model)
        embeddings, valid_ids = _embed_image_ids(
            encoder, dataset, cand_ids, batch_size=args.batch_size
        )
        print(f"SigLIP embeddings: {embeddings.shape}  valid {len(valid_ids)}/{len(cand_ids)}")

        # Build an EmbeddingIndex over just these images, save it.
        index = EmbeddingIndex()
        index.build(embeddings, valid_ids)
        index_path = os.path.join(cfg.results.embeddings_dir, "siglip_test_index.pt")
        index.save(index_path)

        # Reuse CLIPRetriever since its scoped methods are encoder-agnostic.
        retriever = CLIPRetriever(encoder, index, cfg)

        metrics = evaluate_retrieval_scoped(retriever, test_data, cfg.retrieval.top_k_values)

        save_metrics(
            {"per_claim_siglip": metrics, "model": args.model},
            os.path.join(cfg.results.metrics_dir, "retrieval_siglip_per_claim_metrics.json"),
        )

        plot_recall_at_k(
            metrics["recall_at_k"],
            os.path.join(cfg.results.figures_dir, "recall_at_k_siglip_per_claim.png"),
        )

        tracker.log_results(
            **{f"siglip_recall@{k}": v for k, v in metrics["recall_at_k"].items()},
            **{f"siglip_recall_in_pool@{k}": v for k, v in metrics["recall_at_k_gold_in_pool"].items()},
            avg_pool_size=metrics["avg_pool_size"],
            gold_in_pool_coverage=metrics["gold_in_pool_coverage"],
        )
        tracker.complete()

    except Exception as e:
        tracker.fail(str(e))
        raise

    print("=" * 60)
    print(f"SigLIP per-claim retrieval | n_total={metrics['n_total']} "
          f"n_has_gold={metrics['n_has_gold']} n_in_pool={metrics['n_gold_in_pool']}")
    print(f"Avg pool size: {metrics['avg_pool_size']:.2f}")
    print("-" * 60)
    print(f"{'K':>4}  {'SigLIP (all)':>14}  {'SigLIP (in-pool)':>18}")
    for k in sorted(metrics["recall_at_k"].keys()):
        print(f"{k:>4}  {metrics['recall_at_k'][k]:>14.4f}  "
              f"{metrics['recall_at_k_gold_in_pool'][k]:>18.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
