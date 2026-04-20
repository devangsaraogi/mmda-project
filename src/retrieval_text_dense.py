"""Dense text retrieval over per-claim text candidates.

Mirror of ``src/retrieval_text.BM25Retriever`` but backed by a
sentence-encoder (BGE-small / E5-small by default). Uses pure HuggingFace
transformers — no ``sentence-transformers`` dependency — to stay
consistent with the rest of the repo (which already pulls CLIP and
SigLIP through ``transformers`` directly).

Interface mirrors BM25Retriever so the downstream pipeline can swap
retrievers with no logic change:

    retriever.retrieve(claim_text, text_candidates, dataset, top_k=K)
    retriever.retrieve_all(dataset, split="test", top_k=K)
    retriever.evaluate_recall(dataset, split="test", top_k_values=[...])

Default encoder: ``BAAI/bge-small-en-v1.5`` (33M params, strong MTEB,
cheap on CPU for small per-claim pools).
"""
from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


# BGE / E5 style mean-pooling; handles attention mask.
def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


class DenseTextRetriever:
    """Rank per-claim text candidates using a dense sentence encoder."""

    # E5 family expects "query: " / "passage: " prefixes; BGE does not.
    E5_QUERY_PREFIX = "query: "
    E5_PASSAGE_PREFIX = "passage: "

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        batch_size: int = 64,
        max_seq_length: int = 256,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        logger.info("Loading dense text encoder: %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        name_lower = model_name.lower()
        self._is_e5 = "e5" in name_lower
        # BGE-v1.5 retrieval guidance recommends a query instruction prefix.
        self._is_bge = "bge" in name_lower and "v1.5" in name_lower
        self._bge_query_instruction = (
            "Represent this sentence for searching relevant passages: "
        )

        logger.info(
            "Dense retriever ready: %d params, is_e5=%s, is_bge_v15=%s",
            sum(p.numel() for p in self.model.parameters()),
            self._is_e5,
            self._is_bge,
        )

    def _format_query(self, text: str) -> str:
        if self._is_e5:
            return self.E5_QUERY_PREFIX + text
        if self._is_bge:
            return self._bge_query_instruction + text
        return text

    def _format_passage(self, text: str) -> str:
        if self._is_e5:
            return self.E5_PASSAGE_PREFIX + text
        return text

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> torch.Tensor:
        all_vecs = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc).last_hidden_state  # [B, T, H]
            pooled = _mean_pool(out, enc["attention_mask"])  # [B, H]
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            all_vecs.append(pooled.cpu())
        return torch.cat(all_vecs, dim=0) if all_vecs else torch.zeros((0, 1))

    def retrieve(
        self,
        claim_text: str,
        text_candidates: list[dict],
        dataset,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Rank a claim's text candidates by dense cosine similarity.

        Returns list of (candidate_id, similarity_score) descending by score.
        """
        if not text_candidates:
            return []

        corpus_texts: list[str] = []
        corpus_ids: list[str] = []
        for cand in text_candidates:
            text = dataset.get_candidate_text(cand)
            cid = dataset.get_candidate_id(cand)
            if text.strip():
                corpus_texts.append(self._format_passage(text))
                corpus_ids.append(cid)
        if not corpus_texts:
            return []

        query_vec = self._encode([self._format_query(claim_text)])  # [1, H]
        passage_vecs = self._encode(corpus_texts)                   # [N, H]
        sims = (query_vec @ passage_vecs.T).squeeze(0).tolist()     # [N]

        ranked = sorted(zip(corpus_ids, sims), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def retrieve_all(
        self,
        dataset,
        split: str = "test",
        top_k: int = 5,
    ) -> dict[str, list[tuple[str, float]]]:
        """Run dense retrieval for all claims in a split."""
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
                logger.info("Dense retrieval: %d/%d claims", i + 1, n)

        return results

    def evaluate_recall(
        self,
        dataset,
        split: str = "test",
        top_k_values: list[int] | None = None,
    ) -> dict[int, float]:
        """Recall@K using gold text IDs (denominator = claims with gold)."""
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
            logger.info("Dense Recall@%d: %.4f (%d/%d)", k, recall[k], hits, total)
        return recall
