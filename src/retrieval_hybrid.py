"""Caption-augmented hybrid retrieval for per-claim image scoring.

Motivation
----------
Per-claim CLIP retrieval (see ``src.retrieval.CLIPRetriever.retrieve_scoped``)
ranks a claim's ~10 image candidates by vision-language similarity between
the claim text and each candidate's pixel content. WebQA images come with
rich text metadata (``title`` and ``caption`` fields inside
``img_posFacts`` / ``img_negFacts``) that we currently ignore.

This module adds:
  * ``CaptionBM25Retriever`` — BM25 over each candidate's ``title +
    " " + caption`` string, one BM25 index per claim.
  * ``RRFHybridRetriever`` — reciprocal-rank fusion of per-claim CLIP
    scores and caption-BM25 scores. Standard RRF:
        score(i) = 1 / (k + rank_clip(i)) + 1 / (k + rank_bm25(i))
    with k=60 (the value from Cormack et al. 2009 that most IR work
    uses as the default).

Both retrievers run entirely on the already-enriched JSONL
(``image_candidate_metadata``) --- no new model download, no re-embedding.
The caption tokeniser is the same NLTK word-tokenise that
``src.retrieval_text.BM25Retriever`` uses for text snippets, so the two
BM25 paths behave identically.

Memory / runtime
----------------
For each claim we build a fresh BM25Okapi index over its ~10 candidates'
title+caption strings, then score the claim text against it. Building
BM25Okapi over 10 short docs costs microseconds; the overall loop over
7,392 test claims finishes in seconds.
"""
from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .retrieval import CLIPRetriever
from .retrieval_text import _tokenize as bm25_tokenize  # module-level tokeniser

logger = logging.getLogger(__name__)


def _caption_text(meta: dict) -> str:
    """Build the BM25 document string for a single candidate image."""
    title = meta.get("title", "") or ""
    caption = meta.get("caption", "") or ""
    return f"{title} {caption}".strip()


class CaptionBM25Retriever:
    """BM25 over title+caption of each candidate image, per-claim.

    Unlike ``CLIPRetriever.retrieve_scoped``, this does not need the
    global CLIP embedding index --- all state lives in the dataset.
    """

    def __init__(self, default_top_k: int = 5):
        self.default_top_k = default_top_k
        # Reuse the module-level tokeniser used for per-claim BM25 over
        # text_candidates, so caption-side and text-side tokenisation stay
        # identical (lowercase + NLTK word_tokenize with a split fallback).
        self._tokenize = bm25_tokenize

    def rank(
        self,
        claim_text: str,
        candidates: Sequence[dict],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Rank a claim's candidate images by caption-BM25 score.

        Args:
            claim_text: The claim.
            candidates: Iterable of ``{id, title, caption}`` dicts (aligned
                with ``image_candidate_metadata``).
            top_k: Number of results (defaults to the retriever's default_top_k).

        Returns:
            List of (image_id, bm25_score) tuples, descending by score.
            If the pool is empty, returns ``[]``.
        """
        k = top_k or self.default_top_k
        cand_list = list(candidates)
        if not cand_list:
            return []

        docs = [self._tokenize(_caption_text(m)) for m in cand_list]
        # Empty docs (no title, no caption) would break BM25Okapi with
        # IDF NaNs. Drop them.
        valid = [(i, d) for i, d in enumerate(docs) if d]
        if not valid:
            return []
        kept_indices = [i for i, _ in valid]
        kept_docs = [d for _, d in valid]

        query_tokens = self._tokenize(claim_text)
        if not query_tokens:
            # No query tokens → zero score for everything.
            return []

        bm25 = BM25Okapi(kept_docs)
        scores = bm25.get_scores(query_tokens)

        ranked = sorted(
            zip(kept_indices, scores),
            key=lambda p: -p[1],
        )
        out: list[tuple[str, float]] = []
        for cand_idx, score in ranked[:k]:
            out.append((str(cand_list[cand_idx]["id"]), float(score)))
        return out


class RRFHybridRetriever:
    """Per-claim RRF over (per-claim CLIP) + (caption-BM25).

    Scoring:
        score(i) = 1/(k_rrf + rank_clip(i)) + 1/(k_rrf + rank_bm25(i))
    where rank is 1-indexed (1 = best). Candidates that only appear in
    one ranker are still scored using the other's rank penalty; candidates
    that appear in neither score 0.
    """

    def __init__(
        self,
        clip_retriever: CLIPRetriever,
        caption_retriever: CaptionBM25Retriever,
        k_rrf: int = 60,
    ):
        self.clip = clip_retriever
        self.captions = caption_retriever
        self.k_rrf = int(k_rrf)

    def _clip_scope(
        self,
        claim_text: str,
        candidate_ids: Sequence[str],
        top_n: int,
    ) -> list[tuple[str, float]]:
        return self.clip.retrieve_scoped(claim_text, list(candidate_ids), top_k=top_n)

    def _caption_scope(
        self,
        claim_text: str,
        metadata: Sequence[dict],
        top_n: int,
    ) -> list[tuple[str, float]]:
        return self.captions.rank(claim_text, metadata, top_k=top_n)

    def retrieve_scoped(
        self,
        claim_text: str,
        candidate_ids: Sequence[str],
        metadata: Sequence[dict],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Hybrid rank a single claim's candidates.

        Each sub-retriever is asked for the whole pool
        (``top_n = len(candidate_ids)``) so RRF sees every candidate's
        rank rather than treating an unlisted candidate as a miss.
        """
        cand_list = [str(c) for c in candidate_ids]
        if not cand_list:
            return []
        n = len(cand_list)

        clip_ranked = self._clip_scope(claim_text, cand_list, top_n=n)
        caption_ranked = self._caption_scope(claim_text, list(metadata), top_n=n)

        clip_rank = {cid: r + 1 for r, (cid, _) in enumerate(clip_ranked)}
        caption_rank = {cid: r + 1 for r, (cid, _) in enumerate(caption_ranked)}

        scored: list[tuple[str, float]] = []
        for cid in cand_list:
            s = 0.0
            if cid in clip_rank:
                s += 1.0 / (self.k_rrf + clip_rank[cid])
            if cid in caption_rank:
                s += 1.0 / (self.k_rrf + caption_rank[cid])
            scored.append((cid, s))

        scored.sort(key=lambda p: -p[1])
        k_eff = min(top_k, len(scored))
        return scored[:k_eff]


def evaluate_retrieval_hybrid(
    retriever: RRFHybridRetriever,
    dataset,
    top_k_values: Iterable[int],
) -> dict:
    """Same recall-slicing contract as ``evaluate_retrieval_scoped``.

    Reported slices:
      * ``recall_at_k``                — over all dataset items (task level).
      * ``recall_at_k_has_gold``       — over items whose gold_image_ids is non-empty.
      * ``recall_at_k_gold_in_pool``   — over items whose gold is in the candidate pool.
    """
    top_k_values = list(top_k_values)
    max_k = max(top_k_values)
    n = len(dataset)

    logger.info("Batch hybrid retrieval over %d claims...", n)
    has_gold_mask: list[bool] = []
    gold_in_pool_mask: list[bool] = []
    pool_sizes: list[int] = []

    recall_scores = {k: [] for k in top_k_values}
    recall_has_gold = {k: [] for k in top_k_values}
    recall_in_pool = {k: [] for k in top_k_values}

    log_every = max(1, n // 20)
    for i in range(n):
        item = dataset[i]
        claim_text = item["claim_text"]
        candidate_ids = [str(c) for c in item.get("image_candidate_ids", []) if c is not None]
        metadata = item.get("image_candidate_metadata", [])
        gold = {str(g) for g in item["gold_image_ids"]}
        pool = set(candidate_ids)

        pool_sizes.append(len(candidate_ids))
        has_gold = bool(gold)
        in_pool = has_gold and bool(gold & pool)
        has_gold_mask.append(has_gold)
        gold_in_pool_mask.append(in_pool)

        results = retriever.retrieve_scoped(claim_text, candidate_ids, metadata, top_k=max_k)

        for k in top_k_values:
            top_k_ids = {r[0] for r in results[:k]}
            hit = has_gold and bool(top_k_ids & gold)
            hit_f = float(hit)
            recall_scores[k].append(hit_f)
            if has_gold:
                recall_has_gold[k].append(hit_f)
            if in_pool:
                recall_in_pool[k].append(hit_f)

        if (i + 1) % log_every == 0 or (i + 1) == n:
            logger.info("Hybrid retrieval %d/%d", i + 1, n)

    def _mean(seq):
        return float(np.mean(seq)) if seq else 0.0

    out = {
        "recall_at_k": {k: _mean(v) for k, v in recall_scores.items()},
        "recall_at_k_has_gold": {k: _mean(v) for k, v in recall_has_gold.items()},
        "recall_at_k_gold_in_pool": {k: _mean(v) for k, v in recall_in_pool.items()},
        "avg_pool_size": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        "n_total": n,
        "n_has_gold": int(sum(has_gold_mask)),
        "n_gold_in_pool": int(sum(gold_in_pool_mask)),
    }
    for k, r in sorted(out["recall_at_k"].items()):
        logger.info("HybridRecall@%d (all):           %.4f", k, r)
    for k, r in sorted(out["recall_at_k_has_gold"].items()):
        logger.info("HybridRecall@%d (has gold):      %.4f", k, r)
    for k, r in sorted(out["recall_at_k_gold_in_pool"].items()):
        logger.info("HybridRecall@%d (gold in pool):  %.4f", k, r)
    return out
