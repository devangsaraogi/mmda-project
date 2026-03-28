"""BM25 text retrieval over per-claim text candidates.

Each claim in WebQA-Adv has ~15-33 inline text_candidates. This module
ranks them with BM25 and evaluates Recall@K against gold text IDs.
"""

import logging
from collections import defaultdict

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from nltk.tokenize import word_tokenize
except ImportError:
    word_tokenize = None

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase + word tokenize (falls back to split if nltk unavailable)."""
    text = text.lower()
    if word_tokenize is not None:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    return text.split()


class BM25Retriever:
    """Rank per-claim text candidates using BM25."""

    def retrieve(
        self,
        claim_text: str,
        text_candidates: list[dict],
        dataset,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Rank a claim's text candidates by BM25 relevance.

        Args:
            claim_text: The claim string.
            text_candidates: List of candidate dicts from the JSONL.
            dataset: WebQADataset instance (for candidate text/ID extraction).
            top_k: Number of candidates to return.

        Returns:
            List of (candidate_id, bm25_score) tuples, descending by score.
        """
        if not text_candidates or BM25Okapi is None:
            return []

        # Build corpus
        corpus_texts = []
        corpus_ids = []
        for cand in text_candidates:
            text = dataset.get_candidate_text(cand)
            cid = dataset.get_candidate_id(cand)
            if text.strip():
                corpus_texts.append(text)
                corpus_ids.append(cid)

        if not corpus_texts:
            return []

        tokenized_corpus = [_tokenize(t) for t in corpus_texts]
        bm25 = BM25Okapi(tokenized_corpus)

        query_tokens = _tokenize(claim_text)
        scores = bm25.get_scores(query_tokens)

        # Rank by score descending
        ranked = sorted(zip(corpus_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def retrieve_all(
        self,
        dataset,
        split: str = "test",
        top_k: int = 5,
    ) -> dict[str, list[tuple[str, float]]]:
        """Run BM25 retrieval for all claims in a split.

        Returns:
            Dict mapping claim_id -> [(candidate_id, score), ...].
        """
        split_data = dataset.get_split(split) if split else dataset
        n = len(split_data)
        results = {}

        for i in range(n):
            item = split_data[i]
            ranked = self.retrieve(
                item["claim_text"],
                item["text_candidates"],
                dataset,
                top_k=top_k,
            )
            results[item["claim_id"]] = ranked

            if (i + 1) % 2000 == 0 or (i + 1) == n:
                logger.info("BM25 retrieval: %d/%d claims", i + 1, n)

        return results

    def evaluate_recall(
        self,
        dataset,
        split: str = "test",
        top_k_values: list[int] = None,
    ) -> dict[int, float]:
        """Evaluate BM25 Recall@K: fraction of claims where at least one
        gold text ID appears in the top-K retrieved candidates.

        Returns:
            Dict mapping K -> recall value.
        """
        if top_k_values is None:
            top_k_values = [1, 3, 5, 10]

        max_k = max(top_k_values)
        all_results = self.retrieve_all(dataset, split=split, top_k=max_k)

        split_data = dataset.get_split(split)
        recall = {}
        for k in top_k_values:
            hits = 0
            total = 0
            for i in range(len(split_data)):
                item = split_data[i]
                gold_ids = set(item["gold_text_ids"])
                if not gold_ids:
                    continue
                total += 1
                retrieved_ids = {cid for cid, _ in all_results.get(item["claim_id"], [])[:k]}
                if retrieved_ids & gold_ids:
                    hits += 1
            recall[k] = hits / total if total > 0 else 0.0
            logger.info("BM25 Recall@%d: %.4f (%d/%d)", k, recall[k], hits, total)

        return recall
