import logging

import numpy as np
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class ThresholdClassifier:
    """Threshold-based verdict classifier using CLIP similarity scores.

    Maps similarity scores to 3-class verdicts using two thresholds:
        - score >= high_thresh → True (0)
        - low_thresh <= score < high_thresh → Unverifiable (2)
        - score < low_thresh → False (1)
    """

    def __init__(self, cfg=None):
        self.low_thresh = 0.3
        self.high_thresh = 0.6
        if cfg:
            self.search_range = list(cfg.verification.threshold.search_range)
            self.search_steps = cfg.verification.threshold.search_steps
        else:
            self.search_range = [0.1, 0.95]
            self.search_steps = 50

    def predict(self, similarities: np.ndarray) -> np.ndarray:
        """Predict verdicts from similarity scores.

        Args:
            similarities: Array of shape [N] with max cosine similarity per claim.

        Returns:
            Array of shape [N] with predicted labels (0, 1, or 2).
        """
        preds = np.ones(len(similarities), dtype=int)  # default: False (1)
        preds[similarities >= self.high_thresh] = 0     # True
        preds[(similarities >= self.low_thresh) & (similarities < self.high_thresh)] = 2  # Unverifiable
        return preds

    def fit(self, similarities: np.ndarray, labels: np.ndarray) -> dict:
        """Grid-search for optimal thresholds on validation set.

        Args:
            similarities: Array of shape [N] with max cosine similarity per claim.
            labels: Array of shape [N] with ground truth labels.

        Returns:
            Dict with best thresholds and validation F1 score.
        """
        thresholds = np.linspace(self.search_range[0], self.search_range[1], self.search_steps)
        best_f1 = -1.0
        best_low, best_high = self.low_thresh, self.high_thresh

        for low in thresholds:
            for high in thresholds:
                if high <= low:
                    continue
                preds = np.ones(len(similarities), dtype=int)
                preds[similarities >= high] = 0
                preds[(similarities >= low) & (similarities < high)] = 2

                f1 = f1_score(labels, preds, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_low, best_high = low, high

        self.low_thresh = best_low
        self.high_thresh = best_high

        logger.info(
            f"Threshold classifier fitted: low={best_low:.3f}, high={best_high:.3f}, "
            f"val macro-F1={best_f1:.4f}"
        )
        return {
            "low_thresh": float(best_low),
            "high_thresh": float(best_high),
            "val_macro_f1": float(best_f1),
        }
