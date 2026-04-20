"""Retrieval-only text ablation: BM25 vs dense vs hybrid vs hybrid+rerank.

Reports Recall@K on the test split for each retriever and saves the
per-claim ranked results as JSON. Outputs are picked up by
``run_text_pipeline.py --retriever ... --evidence-map ...`` for the
downstream NLI + MLP stage.

No GPU strictly required, but GPU makes the dense encoder and
cross-encoder ~10x faster.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, set_seed, setup_logging, ensure_dirs, create_run_dir
from src.data.webqa_dataset import WebQADataset
from src.retrieval_text import BM25Retriever
from src.retrieval_text_dense import DenseTextRetriever
from src.retrieval_text_hybrid import RRFTextHybridRetriever
from src.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


def _save_ranked(ranked: dict, path: str) -> None:
    serialized = {
        cid: [[rid, float(score)] for rid, score in ranks]
        for cid, ranks in ranked.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialized, f)
    logger.info("Wrote ranked results -> %s", path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument(
        "--dense-model", type=str, default="BAAI/bge-small-en-v1.5",
        help="HuggingFace model for dense retriever.",
    )
    ap.add_argument(
        "--rerank-model", type=str,
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        help="HuggingFace model for cross-encoder rerank.",
    )
    ap.add_argument(
        "--pool-size", type=int, default=20,
        help="Candidates per base ranker into RRF / into rerank.",
    )
    ap.add_argument(
        "--k-rrf", type=int, default=60,
        help="RRF smoothing constant.",
    )
    ap.add_argument(
        "--split", type=str, default="test",
        choices=["train", "val", "test"],
    )
    ap.add_argument(
        "--skip-rerank", action="store_true",
        help="Skip the cross-encoder rerank stage (useful if GPU is unavailable).",
    )
    ap.add_argument(
        "--out-dir", type=str, default="results/text_ablation",
    )
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    set_seed(cfg.seed)
    ensure_dirs(cfg)
    os.makedirs(args.out_dir, exist_ok=True)

    top_k_values = [1, 3, 5, 10]
    max_k = max(top_k_values + [args.pool_size])

    print("=" * 60)
    print("Text Retrieval Ablation")
    print(f"  split: {args.split}")
    print(f"  dense: {args.dense_model}")
    print(f"  rerank: {'<skipped>' if args.skip_rerank else args.rerank_model}")
    print(f"  pool_size: {args.pool_size}")
    print(f"  k_rrf: {args.k_rrf}")
    print("=" * 60)

    dataset = WebQADataset(cfg)

    # --- 1. BM25 ---
    print("\n[1/4] BM25")
    t0 = time.time()
    bm25 = BM25Retriever()
    bm25_ranked = bm25.retrieve_all(dataset, split=args.split, top_k=max_k)
    bm25_recall = {}
    for k in top_k_values:
        hits, total = 0, 0
        for item in dataset.get_split(args.split):
            gold = set(item["gold_text_ids"])
            if not gold:
                continue
            total += 1
            retrieved = {cid for cid, _ in bm25_ranked.get(item["claim_id"], [])[:k]}
            if retrieved & gold:
                hits += 1
        bm25_recall[k] = hits / total if total > 0 else 0.0
    for k, r in bm25_recall.items():
        print(f"  BM25 Recall@{k}: {r:.4f}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    _save_ranked(bm25_ranked, os.path.join(args.out_dir, f"ranked_bm25_{args.split}.json"))

    # --- 2. Dense ---
    print(f"\n[2/4] Dense ({args.dense_model})")
    t0 = time.time()
    dense = DenseTextRetriever(model_name=args.dense_model)
    dense_ranked = dense.retrieve_all(dataset, split=args.split, top_k=max_k)
    dense_recall = {}
    for k in top_k_values:
        hits, total = 0, 0
        for item in dataset.get_split(args.split):
            gold = set(item["gold_text_ids"])
            if not gold:
                continue
            total += 1
            retrieved = {cid for cid, _ in dense_ranked.get(item["claim_id"], [])[:k]}
            if retrieved & gold:
                hits += 1
        dense_recall[k] = hits / total if total > 0 else 0.0
    for k, r in dense_recall.items():
        print(f"  Dense Recall@{k}: {r:.4f}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    _save_ranked(dense_ranked, os.path.join(args.out_dir, f"ranked_dense_{args.split}.json"))

    # --- 3. Hybrid (RRF over BM25 + dense) ---
    print(f"\n[3/4] Hybrid BM25+dense (RRF, k_rrf={args.k_rrf})")
    t0 = time.time()
    hybrid = RRFTextHybridRetriever(bm25, dense, k_rrf=args.k_rrf, pool_size=args.pool_size)
    hybrid_ranked = hybrid.retrieve_all(dataset, split=args.split, top_k=max_k)
    hybrid_recall = {}
    for k in top_k_values:
        hits, total = 0, 0
        for item in dataset.get_split(args.split):
            gold = set(item["gold_text_ids"])
            if not gold:
                continue
            total += 1
            retrieved = {cid for cid, _ in hybrid_ranked.get(item["claim_id"], [])[:k]}
            if retrieved & gold:
                hits += 1
        hybrid_recall[k] = hits / total if total > 0 else 0.0
    for k, r in hybrid_recall.items():
        print(f"  Hybrid Recall@{k}: {r:.4f}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    _save_ranked(hybrid_ranked, os.path.join(args.out_dir, f"ranked_hybrid_{args.split}.json"))

    rerank_recall: dict[int, float] = {}
    if not args.skip_rerank:
        print(f"\n[4/4] CE Rerank over hybrid top-{args.pool_size}")
        t0 = time.time()
        reranker = CrossEncoderReranker(model_name=args.rerank_model)
        # Rerank takes the top-`pool_size` from hybrid as input (recall-maximizing pool).
        rerank_input = {
            cid: ranks[: args.pool_size] for cid, ranks in hybrid_ranked.items()
        }
        reranked = reranker.rerank_all(dataset, rerank_input, split=args.split, top_k=max_k)
        rerank_recall = CrossEncoderReranker.evaluate_recall(
            dataset, reranked, split=args.split, top_k_values=top_k_values,
        )
        for k, r in rerank_recall.items():
            print(f"  Rerank Recall@{k}: {r:.4f}")
        print(f"  Elapsed: {time.time() - t0:.1f}s")
        _save_ranked(reranked, os.path.join(args.out_dir, f"ranked_rerank_{args.split}.json"))
    else:
        print("\n[4/4] CE Rerank -- SKIPPED (--skip-rerank)")

    # --- Summary ---
    summary = {
        "split": args.split,
        "pool_size": args.pool_size,
        "k_rrf": args.k_rrf,
        "dense_model": args.dense_model,
        "rerank_model": None if args.skip_rerank else args.rerank_model,
        "recall": {
            "bm25": bm25_recall,
            "dense": dense_recall,
            "hybrid": hybrid_recall,
            "rerank": rerank_recall,
        },
    }
    summary_path = os.path.join(args.out_dir, f"retrieval_summary_{args.split}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
