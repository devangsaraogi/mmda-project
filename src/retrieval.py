from __future__ import annotations

import os
import logging
import queue
import threading
from typing import Optional

import torch
import numpy as np

from .models.clip_encoder import CLIPEncoder
from .data.base_dataset import BaseClaimDataset

logger = logging.getLogger(__name__)


def _progress_bar(done: int, total: int, label: str, width: int = 30) -> str:
    """Build a text progress bar string for logging."""
    frac = done / total if total else 0
    filled = int(width * frac)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"{label}: [{bar}] {100*frac:5.1f}%  {done}/{total}"


class EmbeddingIndex:
    """Brute-force cosine similarity index over pre-computed embeddings."""

    def __init__(self):
        self.embeddings = None  # [N, D] tensor
        self.image_ids = []     # ordered list of image IDs

    def build(self, embeddings: torch.Tensor, image_ids: list[str]) -> None:
        """Build index from embeddings and IDs."""
        self.embeddings = embeddings.float()
        self.image_ids = list(image_ids)
        logger.info(f"Built index with {len(image_ids)} embeddings, dim={embeddings.shape[1]}")

    def save(self, path: str) -> None:
        """Save index to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "embeddings": self.embeddings,
            "image_ids": self.image_ids,
        }, path)
        logger.info(f"Saved index to {path}")

    def load(self, path: str) -> None:
        """Load index from disk."""
        data = torch.load(path, weights_only=True)
        self.embeddings = data["embeddings"]
        self.image_ids = data["image_ids"]
        logger.info(f"Loaded index from {path}: {len(self.image_ids)} embeddings")

    def search(self, query: torch.Tensor, top_k: int) -> list[tuple[str, float]]:
        """Find top-K most similar images to a query embedding.

        Args:
            query: Shape [D] or [1, D], L2-normalized.
            top_k: Number of results to return.

        Returns:
            List of (image_id, similarity_score) tuples, descending by score.
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        sims = (query @ self.embeddings.T).squeeze(0)  # [N]
        top_k = min(top_k, len(self.image_ids))
        values, indices = torch.topk(sims, top_k)
        return [(self.image_ids[idx], val.item()) for idx, val in zip(indices.tolist(), values)]


class CLIPRetriever:
    """End-to-end CLIP retrieval pipeline: claim text → top-K evidence images."""

    def __init__(self, encoder: CLIPEncoder, index: EmbeddingIndex, cfg):
        self.encoder = encoder
        self.index = index
        self.default_top_k = cfg.retrieval.default_top_k
        self._id_to_idx: dict[str, int] | None = None  # built lazily for scoped search

    def _ensure_id_lookup(self) -> None:
        if self._id_to_idx is None:
            self._id_to_idx = {iid: i for i, iid in enumerate(self.index.image_ids)}

    def retrieve(self, claim_text: str, top_k: int = None) -> list[tuple[str, float]]:
        """Retrieve top-K images for a single claim.

        Args:
            claim_text: The claim to find evidence for.
            top_k: Number of results (defaults to config value).

        Returns:
            List of (image_id, similarity_score) tuples.
        """
        k = top_k or self.default_top_k
        text_emb = self.encoder.encode_texts([claim_text])  # [1, D]
        return self.index.search(text_emb.squeeze(0), k)

    def retrieve_batch(self, claims: list[str], top_k: int = None) -> list[list[tuple[str, float]]]:
        """Retrieve top-K images for a batch of claims.

        Args:
            claims: List of claim texts.
            top_k: Number of results per claim.

        Returns:
            List of lists of (image_id, similarity_score) tuples.
        """
        k = top_k or self.default_top_k
        text_embs = self.encoder.encode_texts(claims)  # [B, D]
        results = []
        for i in range(len(claims)):
            results.append(self.index.search(text_embs[i], k))
        return results

    def retrieve_scoped(
        self,
        claim_text: str,
        candidate_ids: list[str],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieve top-K images from a per-claim candidate pool.

        Reuses the existing global embedding index — pulls the rows for
        ``candidate_ids`` and ranks only within that subset. Candidates
        missing from the index are silently dropped.

        Args:
            claim_text: The claim to find evidence for.
            candidate_ids: Restricted pool of image IDs (per-claim).
            top_k: Number of results (defaults to config value).

        Returns:
            List of (image_id, similarity_score) tuples, descending by score.
            Empty list if no candidate IDs match the index.
        """
        k = top_k or self.default_top_k
        if not candidate_ids:
            return []
        self._ensure_id_lookup()
        valid_pairs: list[tuple[str, int]] = []
        for iid in candidate_ids:
            s = str(iid)
            idx = self._id_to_idx.get(s)
            if idx is not None:
                valid_pairs.append((s, idx))
        if not valid_pairs:
            return []
        valid_ids = [p[0] for p in valid_pairs]
        row_indices = torch.tensor([p[1] for p in valid_pairs], dtype=torch.long)
        cand_embeds = self.index.embeddings.index_select(0, row_indices)  # [M, D]
        text_emb = self.encoder.encode_texts([claim_text]).squeeze(0).float()  # [D]
        sims = (cand_embeds @ text_emb)  # [M]
        k_eff = min(k, len(valid_ids))
        values, top_indices = torch.topk(sims, k_eff)
        return [(valid_ids[idx], val.item()) for idx, val in zip(top_indices.tolist(), values)]

    def retrieve_scoped_batch(
        self,
        claims: list[str],
        candidate_ids_per_claim: list[list[str]],
        top_k: int | None = None,
    ) -> list[list[tuple[str, float]]]:
        """Per-claim scoped retrieval over a batch of claims.

        Encodes all claims in one pass, then ranks each claim against its
        own candidate pool. Claims with no valid candidate IDs yield [].
        """
        if len(claims) != len(candidate_ids_per_claim):
            raise ValueError(
                f"claims ({len(claims)}) and candidate_ids_per_claim "
                f"({len(candidate_ids_per_claim)}) length mismatch"
            )
        k = top_k or self.default_top_k
        self._ensure_id_lookup()
        text_embs = self.encoder.encode_texts(claims).float()  # [B, D]

        results: list[list[tuple[str, float]]] = []
        for i, cand_ids in enumerate(candidate_ids_per_claim):
            if not cand_ids:
                results.append([])
                continue
            valid_pairs: list[tuple[str, int]] = []
            for iid in cand_ids:
                s = str(iid)
                idx = self._id_to_idx.get(s)
                if idx is not None:
                    valid_pairs.append((s, idx))
            if not valid_pairs:
                results.append([])
                continue
            valid_ids = [p[0] for p in valid_pairs]
            row_indices = torch.tensor([p[1] for p in valid_pairs], dtype=torch.long)
            cand_embeds = self.index.embeddings.index_select(0, row_indices)  # [M, D]
            sims = (cand_embeds @ text_embs[i])  # [M]
            k_eff = min(k, len(valid_ids))
            values, top_indices = torch.topk(sims, k_eff)
            results.append(
                [(valid_ids[idx], val.item()) for idx, val in zip(top_indices.tolist(), values)]
            )
        return results


def precompute_embeddings(
    encoder: CLIPEncoder,
    dataset: BaseClaimDataset,
    save_path: str,
    batch_size: int = 64,
) -> EmbeddingIndex:
    """Pre-compute CLIP embeddings for all images in a dataset.

    Args:
        encoder: CLIP encoder instance.
        dataset: Dataset providing images.
        save_path: Where to save the index.
        batch_size: Images per batch.

    Returns:
        Built EmbeddingIndex.
    """
    total_images = len(dataset.get_all_image_ids())
    logger.info(f"Pre-computing embeddings for {total_images} images (sequential scan)...")

    # Sequential scan with prefetch: background thread reads TSV linearly
    # while GPU encodes the current batch
    prefetch_q = queue.Queue(maxsize=2)

    def _loader():
        for batch_ids, batch_imgs in dataset.iter_all_images(batch_size):
            prefetch_q.put((batch_ids, batch_imgs))
        prefetch_q.put(None)  # sentinel

    loader_thread = threading.Thread(target=_loader, daemon=True)
    loader_thread.start()

    all_embeds = []
    valid_ids = []
    skipped = 0
    done = 0
    num_batches = (total_images + batch_size - 1) // batch_size
    log_every = max(1, num_batches // 20)
    batch_idx = 0
    while True:
        item = prefetch_q.get()
        if item is None:
            break
        batch_ids, batch_images = item
        # Filter out unreadable images
        valid_pairs = [(img_id, img) for img_id, img in zip(batch_ids, batch_images) if img is not None]
        skipped += len(batch_ids) - len(valid_pairs)
        if valid_pairs:
            ids, imgs = zip(*valid_pairs)
            embeds = encoder.encode_images(list(imgs), batch_size=len(imgs))
            all_embeds.append(embeds)
            valid_ids.extend(ids)
        done += len(batch_ids)
        batch_idx += 1
        if batch_idx % log_every == 0 or batch_idx == num_batches:
            logger.info(_progress_bar(done, total_images, "Encoding"))

    logger.info("=" * 50)
    logger.info("EMBEDDING SUMMARY")
    logger.info(f"  Total images in dataset: {total_images}")
    logger.info(f"  Successfully encoded:    {len(valid_ids)}")
    logger.info(f"  Skipped (corrupt):       {skipped}")
    if total_images > 0:
        logger.info(f"  Success rate:            {100*len(valid_ids)/total_images:.2f}%")
    logger.info("=" * 50)

    if not valid_ids:
        raise RuntimeError("All images failed to load — no embeddings produced")

    if skipped > total_images * 0.05:
        logger.warning(f"High corruption rate: {skipped}/{total_images} images failed")

    all_embeds = torch.cat(all_embeds, dim=0)

    index = EmbeddingIndex()
    index.build(all_embeds, valid_ids)
    index.save(save_path)

    return index


def evaluate_retrieval(
    retriever: CLIPRetriever,
    dataset: BaseClaimDataset,
    top_k_values: list[int],
) -> dict:
    """Evaluate retrieval with Recall@K.

    Args:
        retriever: CLIPRetriever instance.
        dataset: Dataset with gold image IDs.
        top_k_values: List of K values to evaluate.

    Returns:
        Dict mapping K to Recall@K.
    """
    max_k = max(top_k_values)
    n = len(dataset)

    # Batch encode all claim texts at once (instead of one-by-one)
    logger.info("Batch encoding %d claim texts...", n)
    claim_texts = [dataset[i]["claim_text"] for i in range(n)]
    all_results = retriever.retrieve_batch(claim_texts, top_k=max_k)

    recall_scores = {k: [] for k in top_k_values}
    log_every = max(1, n // 20)
    for i in range(n):
        item = dataset[i]
        results = all_results[i]
        gold_ids = set(item["gold_image_ids"])

        for k in top_k_values:
            top_k_ids = {r[0] for r in results[:k]}
            hit = len(top_k_ids & gold_ids) > 0
            recall_scores[k].append(float(hit))

        if (i + 1) % log_every == 0 or (i + 1) == n:
            logger.info(_progress_bar(i + 1, n, "Retrieval eval"))

    recall_at_k = {k: float(np.mean(scores)) for k, scores in recall_scores.items()}
    for k, r in sorted(recall_at_k.items()):
        logger.info(f"Recall@{k}: {r:.4f}")

    return recall_at_k


def _mean_scores(scores: list[float]) -> float:
    return float(np.mean(scores)) if scores else 0.0


def evaluate_retrieval_scoped(
    retriever: CLIPRetriever,
    dataset: BaseClaimDataset,
    top_k_values: list[int],
) -> dict:
    """Evaluate per-claim scoped retrieval with Recall@K over three subsets.

    Each dataset item is expected to provide ``image_candidate_ids`` —
    the per-claim candidate pool built by ``enrich_webqa_adv_candidates.py``.

    Returns recall sliced three ways so the headline numbers are apples-to-apples:
      * ``recall_at_k``                  — all claims (standard denominator).
      * ``recall_at_k_has_gold``         — only claims with non-empty gold_image_ids
                                           (excludes Unverifiable claims where
                                           visual retrieval has nothing to hit).
      * ``recall_at_k_gold_in_pool``     — only claims where at least one gold
                                           image is present in the candidate pool
                                           (this is the achievable upper bound
                                           for per-claim retrieval).
    """
    max_k = max(top_k_values)
    n = len(dataset)

    logger.info("Batch encoding %d claim texts (scoped retrieval)...", n)
    claim_texts: list[str] = []
    claim_ids: list[str] = []
    misinfo_types: list[str] = []
    pools: list[list[str]] = []
    gold_sets: list[set[str]] = []
    has_gold_mask: list[bool] = []
    gold_in_pool_mask: list[bool] = []
    pool_sizes: list[int] = []
    empty_pool_claims = 0

    for i in range(n):
        item = dataset[i]
        claim_texts.append(item["claim_text"])
        claim_ids.append(item.get("claim_id", str(i)))
        misinfo_types.append(item.get("misinfo_type", "unknown"))
        pool = [str(c) for c in item.get("image_candidate_ids", []) if c is not None]
        pools.append(pool)
        pool_sizes.append(len(pool))
        if not pool:
            empty_pool_claims += 1
        gold = set(str(g) for g in item["gold_image_ids"])
        gold_sets.append(gold)
        has_gold_mask.append(bool(gold))
        gold_in_pool_mask.append(bool(gold) and bool(gold & set(pool)))

    all_results = retriever.retrieve_scoped_batch(claim_texts, pools, top_k=max_k)

    recall_scores = {k: [] for k in top_k_values}
    recall_has_gold = {k: [] for k in top_k_values}
    recall_in_pool = {k: [] for k in top_k_values}
    log_every = max(1, n // 20)

    for i in range(n):
        results = all_results[i]
        gold_ids = gold_sets[i]
        for k in top_k_values:
            top_k_ids = {r[0] for r in results[:k]}
            hit = bool(gold_ids) and len(top_k_ids & gold_ids) > 0
            hit_f = float(hit)
            recall_scores[k].append(hit_f)
            if has_gold_mask[i]:
                recall_has_gold[k].append(hit_f)
            if gold_in_pool_mask[i]:
                recall_in_pool[k].append(hit_f)
        if (i + 1) % log_every == 0 or (i + 1) == n:
            logger.info(_progress_bar(i + 1, n, "Scoped retrieval"))

    recall_at_k = {k: _mean_scores(s) for k, s in recall_scores.items()}
    recall_at_k_has_gold = {k: _mean_scores(s) for k, s in recall_has_gold.items()}
    recall_at_k_gold_in_pool = {k: _mean_scores(s) for k, s in recall_in_pool.items()}

    avg_pool_size = float(np.mean(pool_sizes)) if pool_sizes else 0.0
    n_has_gold = int(sum(has_gold_mask))
    n_gold_in_pool = int(sum(gold_in_pool_mask))
    coverage = n_gold_in_pool / n if n else 0.0

    for k, r in sorted(recall_at_k.items()):
        logger.info(f"ScopedRecall@{k} (all):            {r:.4f}")
    for k, r in sorted(recall_at_k_has_gold.items()):
        logger.info(f"ScopedRecall@{k} (has gold):       {r:.4f}")
    for k, r in sorted(recall_at_k_gold_in_pool.items()):
        logger.info(f"ScopedRecall@{k} (gold in pool):   {r:.4f}")
    logger.info(f"Avg candidate pool size:           {avg_pool_size:.2f}")
    logger.info(f"n_total / n_has_gold / n_in_pool:  {n} / {n_has_gold} / {n_gold_in_pool}")
    logger.info(f"Gold-in-pool coverage:             {coverage:.4f}")
    logger.info(f"Empty pools:                       {empty_pool_claims}/{n}")

    return {
        "recall_at_k": recall_at_k,
        "recall_at_k_has_gold": recall_at_k_has_gold,
        "recall_at_k_gold_in_pool": recall_at_k_gold_in_pool,
        "avg_pool_size": avg_pool_size,
        "gold_in_pool_coverage": coverage,
        "empty_pool_claims": empty_pool_claims,
        "n_total": n,
        "n_has_gold": n_has_gold,
        "n_gold_in_pool": n_gold_in_pool,
        # Per-claim hit indicators (for bootstrap CIs and per-type breakdowns).
        # All lists below are length n_total, aligned with dataset index order.
        "per_claim_hits": {str(k): [int(v) for v in recall_scores[k]] for k in top_k_values},
        "has_gold_mask": [int(v) for v in has_gold_mask],
        "gold_in_pool_mask": [int(v) for v in gold_in_pool_mask],
        "claim_ids": claim_ids,
        "misinfo_types": misinfo_types,
    }
