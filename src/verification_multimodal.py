"""Multimodal verification: combine visual (CLIP) and text (NLI) features.

Supports four evaluation settings based on oracle/E2E for each modality.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def extract_multimodal_features(
    visual_features: np.ndarray,
    text_features: np.ndarray,
) -> np.ndarray:
    """Concatenate visual and text features.

    Args:
        visual_features: [N, V] — e.g. 6-dim CLIP similarity features.
        text_features: [N, T] — e.g. 6-dim NLI aggregated features.

    Returns:
        [N, V+T] concatenated features.
    """
    assert len(visual_features) == len(text_features), (
        f"Feature count mismatch: visual={len(visual_features)}, text={len(text_features)}"
    )
    return np.concatenate([visual_features, text_features], axis=1)


def align_features_by_claim_id(
    visual_data: dict,
    text_data: dict,
) -> dict:
    """Align visual and text features by claim_id ordering.

    Both dicts should have keys: features, labels, claim_ids, misinfo_types.
    Returns aligned arrays (only claims present in both).

    Returns:
        Dict with:
            visual_features, text_features, labels, claim_ids, misinfo_types
    """
    # Build lookup from text data
    text_lookup = {}
    for i, cid in enumerate(text_data["claim_ids"]):
        text_lookup[cid] = i

    aligned_visual = []
    aligned_text = []
    aligned_labels = []
    aligned_ids = []
    aligned_types = []

    for i, cid in enumerate(visual_data["claim_ids"]):
        if cid in text_lookup:
            j = text_lookup[cid]
            aligned_visual.append(visual_data["features"][i])
            aligned_text.append(text_data["features"][j])
            aligned_labels.append(visual_data["labels"][i])
            aligned_ids.append(cid)
            aligned_types.append(visual_data["misinfo_types"][i])

    n_dropped = len(visual_data["claim_ids"]) - len(aligned_ids)
    if n_dropped > 0:
        logger.warning("Dropped %d claims during alignment (missing from one modality)", n_dropped)

    return {
        "visual_features": np.array(aligned_visual),
        "text_features": np.array(aligned_text),
        "features": np.concatenate([np.array(aligned_visual), np.array(aligned_text)], axis=1),
        "labels": np.array(aligned_labels),
        "claim_ids": aligned_ids,
        "misinfo_types": aligned_types,
    }


MULTIMODAL_SETTINGS = [
    ("visual_oracle+text_oracle", "oracle", "oracle"),
    ("visual_e2e+text_oracle", "e2e", "oracle"),
    ("visual_oracle+text_e2e", "oracle", "e2e"),
    ("visual_e2e+text_e2e", "e2e", "e2e"),
]


def get_setting_label(visual_mode: str, text_mode: str) -> str:
    """Get a human-readable label for a multimodal setting."""
    return f"visual_{visual_mode}+text_{text_mode}"
