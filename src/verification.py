import logging

import torch
import numpy as np

from .models.clip_encoder import CLIPEncoder
from .retrieval import CLIPRetriever, _progress_bar
from .data.base_dataset import BaseClaimDataset

logger = logging.getLogger(__name__)


def extract_features(
    encoder: CLIPEncoder,
    dataset: BaseClaimDataset,
    evidence_map: dict[str, list[tuple[str, float]]],
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

    Returns:
        Dict with:
            features: np.array [N, 6] — hand-crafted features
            embeddings: np.array [N, 1536] — concat text+image embeddings
            labels: np.array [N] — ground truth labels
            claim_ids: list[str]
            misinfo_types: list[str]
    """
    all_features = []
    all_embeddings = []
    all_labels = []
    all_claim_ids = []
    all_misinfo_types = []

    n = len(dataset)
    log_every = max(1, n // 20)
    for i in range(n):
        item = dataset[i]
        claim_id = item["claim_id"]
        claim_text = item["claim_text"]

        evidence = evidence_map.get(claim_id, [])
        if not evidence:
            # No evidence found — use zero features
            all_features.append(np.zeros(6))
            all_embeddings.append(np.zeros(1536))
            all_labels.append(item["label"])
            all_claim_ids.append(claim_id)
            all_misinfo_types.append(item["misinfo_type"])
            continue

        # Get evidence images
        evidence_ids = [eid for eid, _ in evidence]
        evidence_images = [dataset.get_image(eid) for eid in evidence_ids]

        # Encode
        text_emb = encoder.encode_texts([claim_text])         # [1, D]
        image_embs = encoder.encode_images(evidence_images)   # [K, D]

        # Compute similarities
        sims = encoder.compute_similarity(text_emb.squeeze(0), image_embs)  # [K]
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

        # Embedding features: concat text + mean image
        mean_img_emb = image_embs.mean(dim=0)  # [D]
        concat_emb = torch.cat([text_emb.squeeze(0), mean_img_emb], dim=0)  # [2D]
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
    evidence_map = {}
    n = len(dataset)
    log_every = max(1, n // 20)
    for i in range(n):
        item = dataset[i]
        results = retriever.retrieve(item["claim_text"], top_k=top_k)
        evidence_map[item["claim_id"]] = results
        if (i + 1) % log_every == 0 or (i + 1) == n:
            logger.info(_progress_bar(i + 1, n, "Evidence retrieval"))
    return evidence_map


def verify_oracle(
    encoder: CLIPEncoder,
    dataset: BaseClaimDataset,
) -> dict:
    """Run verification with oracle (gold) evidence.

    Returns:
        Feature extraction results dict.
    """
    logger.info("Running oracle verification (gold evidence)...")
    evidence_map = build_evidence_map_oracle(dataset)
    return extract_features(encoder, dataset, evidence_map)


def verify_e2e(
    encoder: CLIPEncoder,
    retriever: CLIPRetriever,
    dataset: BaseClaimDataset,
    top_k: int,
) -> dict:
    """Run end-to-end verification with retrieved evidence.

    Returns:
        Feature extraction results dict.
    """
    logger.info(f"Running E2E verification (top-{top_k} retrieved evidence)...")
    evidence_map = build_evidence_map_retrieved(retriever, dataset, top_k)
    return extract_features(encoder, dataset, evidence_map)
