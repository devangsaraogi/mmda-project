import logging

import torch
import numpy as np

from .models.clip_encoder import CLIPEncoder
from .retrieval import CLIPRetriever, EmbeddingIndex, _progress_bar
from .data.base_dataset import BaseClaimDataset

logger = logging.getLogger(__name__)


def _resolve_image_embeddings(
    encoder: CLIPEncoder,
    dataset: BaseClaimDataset,
    needed_ids: set[str],
    embedding_index: EmbeddingIndex = None,
) -> dict[str, torch.Tensor]:
    """Resolve image embeddings either from a pre-computed index or by loading/encoding.

    Args:
        encoder: CLIP encoder (used only if no index).
        dataset: Dataset for loading images (used only if no index).
        needed_ids: Set of image IDs to resolve.
        embedding_index: Pre-computed embedding index (fast path).

    Returns:
        Dict mapping image_id → [D] embedding tensor.
    """
    if embedding_index is not None:
        # Fast path: look up from pre-computed index (no I/O, no GPU encoding)
        logger.info("Looking up %d image embeddings from index...", len(needed_ids))
        id_to_idx = {img_id: idx for idx, img_id in enumerate(embedding_index.image_ids)}
        img_emb_map = {}
        missing = 0
        for eid in needed_ids:
            idx = id_to_idx.get(eid)
            if idx is not None:
                img_emb_map[eid] = embedding_index.embeddings[idx]
            else:
                missing += 1
        if missing:
            logger.warning("%d image IDs not found in embedding index.", missing)
        return img_emb_map

    # Slow path: load images from TSV and encode with CLIP
    logger.info("Loading and encoding %d evidence images...", len(needed_ids))
    needed_list = list(needed_ids)
    images = dataset.get_images_batch(needed_list)
    valid = [(eid, img) for eid, img in zip(needed_list, images) if img is not None]
    if not valid:
        return {}
    ids, imgs = zip(*valid)
    embs = encoder.encode_images(list(imgs))
    return {eid: embs[i] for i, eid in enumerate(ids)}


def extract_features(
    encoder: CLIPEncoder,
    dataset: BaseClaimDataset,
    evidence_map: dict[str, list[tuple[str, float]]],
    embedding_index: EmbeddingIndex = None,
) -> dict:
    """Extract verification features for each claim-evidence pair.

    For each claim, computes:
        - Similarity statistics: max, mean, min, std over evidence images
        - Top-1 and top-3 mean similarity
        - Concatenated text + mean image embedding (for embedding mode)

    Args:
        encoder: CLIP encoder.
        dataset: Dataset providing claims and images.
        evidence_map: Dict mapping claim_id → list of (image_id, score).
        embedding_index: Pre-computed index for fast image embedding lookup.

    Returns:
        Dict with:
            features: np.array [N, 6] — hand-crafted features
            embeddings: np.array [N, 2*D] — concat text+image embeddings
            labels: np.array [N] — ground truth labels
            claim_ids: list[str]
            misinfo_types: list[str]
    """
    n = len(dataset)
    embed_dim = encoder.embed_dim

    # Collect all items
    items = [dataset[i] for i in range(n)]

    # Batch encode all claim texts at once
    logger.info("Batch encoding %d claim texts...", n)
    claim_texts = [item["claim_text"] for item in items]
    text_embs = encoder.encode_texts(claim_texts)  # [N, D]

    # Resolve all needed image embeddings (from index or by loading)
    all_needed_ids = set()
    for item in items:
        for eid, _ in evidence_map.get(item["claim_id"], []):
            all_needed_ids.add(eid)
    img_emb_map = _resolve_image_embeddings(encoder, dataset, all_needed_ids, embedding_index)

    # Compute features per claim
    all_features = []
    all_embeddings = []
    all_labels = []
    all_claim_ids = []
    all_misinfo_types = []

    log_every = max(1, n // 20)
    for i, item in enumerate(items):
        claim_id = item["claim_id"]
        text_emb = text_embs[i]  # [D]

        evidence = evidence_map.get(claim_id, [])
        evidence_embs = [img_emb_map[eid] for eid, _ in evidence if eid in img_emb_map]

        if not evidence_embs:
            all_features.append(np.zeros(6))
            all_embeddings.append(np.zeros(embed_dim * 2))
            all_labels.append(item["label"])
            all_claim_ids.append(claim_id)
            all_misinfo_types.append(item["misinfo_type"])
            if (i + 1) % log_every == 0 or (i + 1) == n:
                logger.info(_progress_bar(i + 1, n, "Feature extraction"))
            continue

        image_embs = torch.stack(evidence_embs)  # [K, D]
        sims = encoder.compute_similarity(text_emb, image_embs)  # [K]
        sims_np = sims.numpy()

        # Hand-crafted features
        max_sim = float(np.max(sims_np))
        mean_sim = float(np.mean(sims_np))
        min_sim = float(np.min(sims_np))
        std_sim = float(np.std(sims_np)) if len(sims_np) > 1 else 0.0
        top1_sim = float(sims_np[0]) if len(sims_np) > 0 else 0.0
        top3_mean = float(np.mean(sims_np[:3])) if len(sims_np) > 0 else 0.0

        features = np.array([max_sim, mean_sim, min_sim, std_sim, top1_sim, top3_mean])
        all_features.append(features)

        mean_img_emb = image_embs.mean(dim=0)  # [D]
        concat_emb = torch.cat([text_emb, mean_img_emb], dim=0)  # [2D]
        all_embeddings.append(concat_emb.numpy())

        all_labels.append(item["label"])
        all_claim_ids.append(claim_id)
        all_misinfo_types.append(item["misinfo_type"])

        if (i + 1) % log_every == 0 or (i + 1) == n:
            logger.info(_progress_bar(i + 1, n, "Feature extraction"))

    return {
        "features": np.array(all_features),
        "embeddings": np.array(all_embeddings),
        "labels": np.array(all_labels),
        "claim_ids": all_claim_ids,
        "misinfo_types": all_misinfo_types,
    }


def build_evidence_map_oracle(dataset: BaseClaimDataset) -> dict[str, list[tuple[str, float]]]:
    """Build evidence map using gold (oracle) evidence.

    Args:
        dataset: Dataset with gold_image_ids.

    Returns:
        Dict mapping claim_id → list of (image_id, 1.0) tuples.
    """
    evidence_map = {}
    for i in range(len(dataset)):
        item = dataset[i]
        evidence_map[item["claim_id"]] = [(img_id, 1.0) for img_id in item["gold_image_ids"]]
    return evidence_map


def build_evidence_map_retrieved(
    retriever: CLIPRetriever,
    dataset: BaseClaimDataset,
    top_k: int,
) -> dict[str, list[tuple[str, float]]]:
    """Build evidence map using retrieved evidence.

    Args:
        retriever: CLIPRetriever instance.
        dataset: Dataset with claims.
        top_k: Number of images to retrieve per claim.

    Returns:
        Dict mapping claim_id → list of (image_id, score) tuples.
    """
    n = len(dataset)

    # Batch encode all texts + search (instead of one-by-one)
    logger.info("Batch retrieving evidence for %d claims...", n)
    claim_texts = [dataset[i]["claim_text"] for i in range(n)]
    all_results = retriever.retrieve_batch(claim_texts, top_k=top_k)

    evidence_map = {}
    for i in range(n):
        evidence_map[dataset[i]["claim_id"]] = all_results[i]
    logger.info("Evidence retrieval complete.")
    return evidence_map


def verify_oracle(
    encoder: CLIPEncoder,
    dataset: BaseClaimDataset,
    embedding_index: EmbeddingIndex = None,
) -> dict:
    """Run verification with oracle (gold) evidence.

    Returns:
        Feature extraction results dict.
    """
    logger.info("Running oracle verification (gold evidence)...")
    evidence_map = build_evidence_map_oracle(dataset)
    return extract_features(encoder, dataset, evidence_map, embedding_index=embedding_index)


def verify_e2e(
    encoder: CLIPEncoder,
    retriever: CLIPRetriever,
    dataset: BaseClaimDataset,
    top_k: int,
    embedding_index: EmbeddingIndex = None,
) -> dict:
    """Run end-to-end verification with retrieved evidence.

    Returns:
        Feature extraction results dict.
    """
    logger.info(f"Running E2E verification (top-{top_k} retrieved evidence)...")
    evidence_map = build_evidence_map_retrieved(retriever, dataset, top_k)
    return extract_features(encoder, dataset, evidence_map, embedding_index=embedding_index)
