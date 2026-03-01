import os
import logging

import torch
import numpy as np
from tqdm import tqdm

from .models.clip_encoder import CLIPEncoder
from .data.base_dataset import BaseClaimDataset

logger = logging.getLogger(__name__)


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
    image_ids = dataset.get_all_image_ids()
    logger.info(f"Pre-computing embeddings for {len(image_ids)} images...")

    all_embeds = []
    num_batches = (len(image_ids) + batch_size - 1) // batch_size
    log_every = max(1, num_batches // 20)  # ~5% increments
    for batch_idx, i in enumerate(range(0, len(image_ids), batch_size)):
        batch_ids = image_ids[i : i + batch_size]
        batch_images = [dataset.get_image(img_id) for img_id in batch_ids]
        embeds = encoder.encode_images(batch_images, batch_size=len(batch_images))
        all_embeds.append(embeds)
        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == num_batches:
            done = min(i + batch_size, len(image_ids))
            logger.info(f"Encoded {done}/{len(image_ids)} images ({100*done/len(image_ids):.0f}%)")

    all_embeds = torch.cat(all_embeds, dim=0)

    index = EmbeddingIndex()
    index.build(all_embeds, image_ids)
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
    recall_scores = {k: [] for k in top_k_values}

    for i in tqdm(range(len(dataset)), desc="Evaluating retrieval"):
        item = dataset[i]
        results = retriever.retrieve(item["claim_text"], top_k=max_k)
        retrieved_ids = {r[0] for r in results}
        gold_ids = set(item["gold_image_ids"])

        for k in top_k_values:
            top_k_ids = {r[0] for r in results[:k]}
            hit = len(top_k_ids & gold_ids) > 0
            recall_scores[k].append(float(hit))

    recall_at_k = {k: float(np.mean(scores)) for k, scores in recall_scores.items()}
    for k, r in sorted(recall_at_k.items()):
        logger.info(f"Recall@{k}: {r:.4f}")

    return recall_at_k
