"""Cross-encoder reranker for (claim, text-candidate) pairs.

Takes the top-K output of a base retriever (BM25 / dense / hybrid) and
re-scores each (claim, candidate) pair with a cross-encoder. Unlike a
bi-encoder, the cross-encoder sees the query and passage jointly, so it
can catch fine-grained semantic matches that lexical or dense-cosine
approaches miss.

Default model: ``cross-encoder/ms-marco-MiniLM-L-12-v2`` — 33M params,
strong on MS MARCO passage reranking, loads via
``AutoModelForSequenceClassification`` (single-logit regression head).
"""
from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Rerank (claim, candidate-text) pairs with a cross-encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: str | None = None,
        batch_size: int = 32,
        max_seq_length: int = 512,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        logger.info("Loading cross-encoder: %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        num_labels = int(getattr(self.model.config, "num_labels", 1))
        self._is_regression = num_labels == 1
        logger.info(
            "Cross-encoder ready: %d params, num_labels=%d (%s)",
            sum(p.numel() for p in self.model.parameters()),
            num_labels,
            "regression" if self._is_regression else "classification",
        )

    @torch.no_grad()
    def score_pairs(self, claims: list[str], passages: list[str]) -> np.ndarray:
        """Score a flat list of (claim, passage) pairs.

        Returns a float array of shape [N] — higher = more relevant.
        For ms-marco regression models this is the raw logit.
        """
        assert len(claims) == len(passages), "claims and passages must match length"
        if not claims:
            return np.zeros(0, dtype=np.float32)

        all_scores: list[np.ndarray] = []
        for start in range(0, len(claims), self.batch_size):
            end = min(start + self.batch_size, len(claims))
            enc = self.tokenizer(
                claims[start:end],
                passages[start:end],
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits  # [B, C] or [B, 1]
            if self._is_regression:
                scores = logits.squeeze(-1).cpu().numpy()
            else:
                # Multi-class: use positive class probability if available,
                # otherwise max softmax as a relevance proxy.
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                scores = probs[:, -1] if probs.shape[1] > 1 else probs[:, 0]
            all_scores.append(scores)

        return np.concatenate(all_scores).astype(np.float32)

    def rerank(
        self,
        claim_text: str,
        text_candidates: list[dict],
        dataset,
        base_ranked: list[tuple[str, float]],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Rerank a single claim's top-K candidates from a base retriever.

        Args:
            claim_text: the claim string.
            text_candidates: the claim's full candidate list (JSONL entry).
            dataset: WebQADataset, for id->text resolution.
            base_ranked: [(candidate_id, base_score), ...] from BM25/dense/hybrid.
            top_k: number of reranked candidates to return.

        Returns:
            [(candidate_id, cross_encoder_score), ...] sorted descending.
        """
        if not base_ranked:
            return []

        id_to_text: dict[str, str] = {}
        for cand in text_candidates:
            cid = dataset.get_candidate_id(cand)
            id_to_text[cid] = dataset.get_candidate_text(cand)

        candidate_ids: list[str] = []
        passages: list[str] = []
        for cid, _ in base_ranked:
            text = id_to_text.get(cid, "")
            if text.strip():
                candidate_ids.append(cid)
                passages.append(text)
        if not candidate_ids:
            return []

        claims = [claim_text] * len(candidate_ids)
        ce_scores = self.score_pairs(claims, passages)

        ranked = sorted(
            zip(candidate_ids, ce_scores.tolist()),
            key=lambda x: x[1], reverse=True,
        )
        return ranked[:top_k]

    def rerank_all(
        self,
        dataset,
        base_results: dict[str, list[tuple[str, float]]],
        split: str = "test",
        top_k: int = 5,
    ) -> dict[str, list[tuple[str, float]]]:
        """Apply rerank across every claim in a split.

        Args:
            dataset: WebQADataset.
            base_results: per-claim base ranking {claim_id: [(cid, score), ...]}.
            split: which split to iterate.
            top_k: top-K to keep after rerank.
        """
        split_data = dataset.get_split(split) if split else dataset
        n = len(split_data)
        results: dict[str, list[tuple[str, float]]] = {}

        for i in range(n):
            item = split_data[i]
            cid = item["claim_id"]
            base = base_results.get(cid, [])
            results[cid] = self.rerank(
                item["claim_text"],
                item["text_candidates"],
                dataset,
                base,
                top_k=top_k,
            )
            if (i + 1) % 500 == 0 or (i + 1) == n:
                logger.info("Rerank: %d/%d claims", i + 1, n)
        return results

    @staticmethod
    def evaluate_recall(
        dataset,
        reranked: dict[str, list[tuple[str, float]]],
        split: str = "test",
        top_k_values: list[int] | None = None,
    ) -> dict[int, float]:
        """Recall@K after reranking. Same denom rule: claims with gold text IDs."""
        if top_k_values is None:
            top_k_values = [1, 3, 5, 10]

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
                retrieved = {cid for cid, _ in reranked.get(item["claim_id"], [])[:k]}
                if retrieved & gold:
                    hits += 1
            recall[k] = hits / total if total > 0 else 0.0
            logger.info("Rerank Recall@%d: %.4f (%d/%d)", k, recall[k], hits, total)
        return recall
