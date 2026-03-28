"""NLI-based text verification.

Uses a DeBERTa NLI model to score (claim, evidence_text) pairs and
extract aggregated features for verdict classification.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class NLIVerifier:
    """Score claim-evidence pairs with an NLI model and extract features."""

    # Output label order from the NLI model: [entailment, neutral, contradiction]
    # (Some models use [contradiction, neutral, entailment] — we detect at init)
    NLI_LABELS = ["entailment", "neutral", "contradiction"]

    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 32,
                 max_seq_length: int = 512):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        logger.info("Loading NLI model: %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Detect label ordering from model config
        id2label = self.model.config.id2label
        self._ent_idx = None
        self._con_idx = None
        self._neu_idx = None
        for idx, label in id2label.items():
            label_lower = label.lower()
            if "entail" in label_lower:
                self._ent_idx = int(idx)
            elif "contra" in label_lower:
                self._con_idx = int(idx)
            elif "neutr" in label_lower:
                self._neu_idx = int(idx)

        if self._ent_idx is None:
            self._ent_idx, self._neu_idx, self._con_idx = 0, 1, 2
            logger.warning("Could not detect NLI label order, using default [ent=0, neu=1, con=2]")

        logger.info("NLI label indices: entailment=%d, neutral=%d, contradiction=%d",
                     self._ent_idx, self._neu_idx, self._con_idx)
        logger.info("NLI model loaded: %d parameters",
                     sum(p.numel() for p in self.model.parameters()))

    @torch.no_grad()
    def score_pairs(self, claims: list[str], evidences: list[str]) -> np.ndarray:
        """Score claim-evidence pairs.

        Args:
            claims: List of N claim strings.
            evidences: List of N evidence strings.

        Returns:
            np.ndarray of shape [N, 3] with columns [entailment, contradiction, neutral] probs.
        """
        assert len(claims) == len(evidences), "claims and evidences must have same length"

        all_probs = []
        n = len(claims)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_claims = claims[start:end]
            batch_evidences = evidences[start:end]

            inputs = self.tokenizer(
                batch_claims,
                batch_evidences,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            # Reorder to [entailment, contradiction, neutral]
            reordered = np.stack([
                probs[:, self._ent_idx],
                probs[:, self._con_idx],
                probs[:, self._neu_idx],
            ], axis=1)
            all_probs.append(reordered)

            if (end) % (self.batch_size * 10) == 0 or end == n:
                logger.info("NLI scoring: %d/%d pairs", end, n)

        return np.concatenate(all_probs, axis=0)  # [N, 3]

    def extract_features(
        self,
        claim_text: str,
        evidence_texts: list[str],
        top_k: int = 5,
    ) -> np.ndarray:
        """Extract aggregated NLI features for one claim over its evidence texts.

        Uses up to top_k evidence texts. Computes 6-dim feature vector:
            [max_entailment, mean_entailment, max_contradiction, mean_contradiction,
             max_neutral, entailment_minus_contradiction]

        Args:
            claim_text: The claim.
            evidence_texts: List of evidence text strings (already retrieved or oracle).
            top_k: Max number of evidence texts to use.

        Returns:
            np.ndarray of shape [6].
        """
        if not evidence_texts:
            return np.zeros(6)

        texts = evidence_texts[:top_k]
        claims = [claim_text] * len(texts)
        probs = self.score_pairs(claims, texts)  # [K, 3]

        ent = probs[:, 0]  # entailment
        con = probs[:, 1]  # contradiction
        neu = probs[:, 2]  # neutral

        features = np.array([
            float(np.max(ent)),
            float(np.mean(ent)),
            float(np.max(con)),
            float(np.mean(con)),
            float(np.max(neu)),
            float(np.max(ent) - np.max(con)),  # directional signal
        ])
        return features

    def extract_features_batch(
        self,
        dataset,
        evidence_map: dict[str, list[tuple[str, float]]],
        split: str = "test",
        top_k: int = 5,
    ) -> dict:
        """Extract NLI features for all claims in a split.

        Args:
            dataset: WebQADataset instance.
            evidence_map: Dict mapping claim_id -> [(candidate_id, score), ...].
                For oracle mode, candidate_id should match gold text IDs.
            split: Dataset split to process.
            top_k: Max evidence texts per claim.

        Returns:
            Dict with:
                features: np.array [N, 6]
                labels: np.array [N]
                claim_ids: list[str]
                misinfo_types: list[str]
                raw_scores: dict[str, np.ndarray]  — per-claim NLI probabilities
        """
        split_data = dataset.get_split(split)
        n = len(split_data)

        # First, collect all (claim, evidence) pairs for batch NLI scoring
        all_claims = []
        all_evidences = []
        pair_mapping = []  # (index_in_split, pair_index_start, pair_count)

        for i in range(n):
            item = split_data[i]
            claim_id = item["claim_id"]
            claim_text = item["claim_text"]

            # Get evidence texts
            evidence_ids = [eid for eid, _ in evidence_map.get(claim_id, [])[:top_k]]
            candidates = item["text_candidates"]

            # Build candidate ID -> text mapping
            cand_map = {}
            for cand in candidates:
                cid = dataset.get_candidate_id(cand)
                cand_map[cid] = dataset.get_candidate_text(cand)

            evidence_texts = []
            for eid in evidence_ids:
                text = cand_map.get(eid, "")
                if text.strip():
                    evidence_texts.append(text)

            start_idx = len(all_claims)
            for et in evidence_texts[:top_k]:
                all_claims.append(claim_text)
                all_evidences.append(et)
            pair_mapping.append((i, start_idx, len(evidence_texts[:top_k])))

        # Batch NLI scoring
        if all_claims:
            logger.info("Scoring %d claim-evidence pairs with NLI...", len(all_claims))
            all_probs = self.score_pairs(all_claims, all_evidences)
        else:
            all_probs = np.zeros((0, 3))

        # Aggregate features per claim
        all_features = []
        all_labels = []
        all_claim_ids = []
        all_misinfo_types = []
        raw_scores = {}

        for i, start, count in pair_mapping:
            item = split_data[i]

            if count > 0:
                probs = all_probs[start:start + count]
                ent = probs[:, 0]
                con = probs[:, 1]
                neu = probs[:, 2]
                features = np.array([
                    float(np.max(ent)),
                    float(np.mean(ent)),
                    float(np.max(con)),
                    float(np.mean(con)),
                    float(np.max(neu)),
                    float(np.max(ent) - np.max(con)),
                ])
                raw_scores[item["claim_id"]] = probs
            else:
                features = np.zeros(6)

            all_features.append(features)
            all_labels.append(item["label"])
            all_claim_ids.append(item["claim_id"])
            all_misinfo_types.append(item["misinfo_type"])

        return {
            "features": np.array(all_features),
            "labels": np.array(all_labels),
            "claim_ids": all_claim_ids,
            "misinfo_types": all_misinfo_types,
            "raw_scores": raw_scores,
        }


def build_text_evidence_map_oracle(dataset, split: str = "test") -> dict[str, list[tuple[str, float]]]:
    """Build evidence map using gold text IDs (oracle mode).

    Returns:
        Dict mapping claim_id -> [(text_id, 1.0), ...].
    """
    split_data = dataset.get_split(split)
    evidence_map = {}
    for i in range(len(split_data)):
        item = split_data[i]
        evidence_map[item["claim_id"]] = [(tid, 1.0) for tid in item["gold_text_ids"]]
    return evidence_map


def build_text_evidence_map_bm25(
    bm25_retriever,
    dataset,
    split: str = "test",
    top_k: int = 5,
) -> dict[str, list[tuple[str, float]]]:
    """Build evidence map using BM25 retrieved texts (E2E mode).

    Returns:
        Dict mapping claim_id -> [(candidate_id, bm25_score), ...].
    """
    return bm25_retriever.retrieve_all(dataset, split=split, top_k=top_k)
