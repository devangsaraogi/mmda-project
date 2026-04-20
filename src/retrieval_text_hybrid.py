"""BM25 + dense hybrid retrieval for per-claim text candidates (RRF).

Mirrors the visual-side RRF hybrid (``src/retrieval_hybrid.py``) but for
the text modality. Combines:

  - Sparse BM25 over candidate text (``BM25Retriever``)
  - Dense sentence-encoder cosine (``DenseTextRetriever``)

using Reciprocal Rank Fusion with ``k_rrf=60`` (Cormack & Clarke 2009):

    score(d) = sum_i 1 / (k_rrf + rank_i(d))

This is the rank-based fusion, robust to score-scale differences between
BM25 (unbounded additive) and cosine sim (bounded [-1, 1]).

Interface matches the per-claim retrievers: ``retrieve``,
``retrieve_all``, ``evaluate_recall``.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class RRFTextHybridRetriever:
    """Reciprocal Rank Fusion over BM25 + dense text rankers."""

    def __init__(
        self,
        bm25_retriever,
        dense_retriever,
        k_rrf: int = 60,
        pool_size: int = 20,
    ):
        """Args:
            bm25_retriever: instance of ``BM25Retriever``.
            dense_retriever: instance of ``DenseTextRetriever``.
            k_rrf: RRF smoothing constant. 60 is the Cormack default.
            pool_size: how many candidates each base ranker contributes
                before fusion. Larger = more recall, slower.
        """
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.k_rrf = k_rrf
        self.pool_size = pool_size

    def retrieve(
        self,
        claim_text: str,
        text_candidates: list[dict],
        dataset,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        bm25_ranked = self.bm25.retrieve(
            claim_text, text_candidates, dataset, top_k=self.pool_size,
        )
        dense_ranked = self.dense.retrieve(
            claim_text, text_candidates, dataset, top_k=self.pool_size,
        )

        rrf_scores: dict[str, float] = {}
        for rank, (cid, _) in enumerate(bm25_ranked):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.k_rrf + rank + 1)
        for rank, (cid, _) in enumerate(dense_ranked):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.k_rrf + rank + 1)

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return fused[:top_k]

    def retrieve_all(
        self,
        dataset,
        split: str = "test",
        top_k: int = 5,
    ) -> dict[str, list[tuple[str, float]]]:
        split_data = dataset.get_split(split) if split else dataset
        n = len(split_data)
        results: dict[str, list[tuple[str, float]]] = {}

        for i in range(n):
            item = split_data[i]
            ranked = self.retrieve(
                item["claim_text"],
                item["text_candidates"],
                dataset,
                top_k=top_k,
            )
            results[item["claim_id"]] = ranked

            if (i + 1) % 500 == 0 or (i + 1) == n:
                logger.info("Hybrid retrieval: %d/%d claims", i + 1, n)

        return results

    def evaluate_recall(
        self,
        dataset,
        split: str = "test",
        top_k_values: list[int] | None = None,
    ) -> dict[int, float]:
        if top_k_values is None:
            top_k_values = [1, 3, 5, 10]

        max_k = max(top_k_values)
        all_results = self.retrieve_all(dataset, split=split, top_k=max_k)
        split_data = dataset.get_split(split)

        recall: dict[int, float] = {}
        for k in top_k_values:
            hits, total = 0, 0
            for i in range(len(split_data)):
                item = split_data[i]
                gold = set(item["gold_text_ids"])
                if not gold:
                    continue
                total += 1
                retrieved = {cid for cid, _ in all_results.get(item["claim_id"], [])[:k]}
                if retrieved & gold:
                    hits += 1
            recall[k] = hits / total if total > 0 else 0.0
            logger.info("Hybrid Recall@%d: %.4f (%d/%d)", k, recall[k], hits, total)
        return recall
