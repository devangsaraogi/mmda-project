"""WebQA-Adv dataset loader.

Loads claims from a JSONL file and images from a TSV + lineidx pair.
Images are read on demand (not held in memory) for compatibility with
multi-worker DataLoaders.
"""

import base64
import json
import logging
from io import BytesIO

import numpy as np
from PIL import Image

from .base_dataset import BaseClaimDataset

logger = logging.getLogger(__name__)

_LABEL_MAP = {"TRUE": 0, "FALSE": 1, "UNVERIFIABLE": 2}


class WebQADataset(BaseClaimDataset):
    """Loader for the WebQA-Adv adversarial dataset.

    Data files
    ----------
    - ``webqa_adv.jsonl``  — one JSON object per line with claim info
    - ``imgs.tsv``         — ``<numeric_id>\\t<base64_bytes>\\n`` per image
    - ``imgs.lineidx``     — byte offset of each TSV line, one per line
    """

    def __init__(self, cfg):
        data_cfg = cfg.data

        # ------------------------------------------------------------------
        # 1. Load claims from JSONL
        # ------------------------------------------------------------------
        jsonl_path = data_cfg.webqa_jsonl
        logger.info("Loading claims from %s …", jsonl_path)
        self._claims = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self._claims.append(json.loads(line))
        logger.info("Loaded %d claims.", len(self._claims))

        # ------------------------------------------------------------------
        # 2. Load byte offsets from lineidx
        # ------------------------------------------------------------------
        lineidx_path = data_cfg.webqa_images_lineidx
        logger.info("Loading image index from %s …", lineidx_path)
        with open(lineidx_path, "r", encoding="utf-8") as f:
            self._byte_offsets = [int(offset.strip()) for offset in f]
        logger.info("Loaded %d byte offsets.", len(self._byte_offsets))

        # ------------------------------------------------------------------
        # 3. Build TSV index: {numeric_id_str: line_number}
        # ------------------------------------------------------------------
        self._tsv_path = data_cfg.webqa_images_tsv
        logger.info("Building TSV index from %s …", self._tsv_path)
        self._id_to_line = {}
        with open(self._tsv_path, "rb") as tsv:
            for line_num, offset in enumerate(self._byte_offsets):
                tsv.seek(offset)
                # Read just enough to grab the numeric ID (before the tab)
                chunk = tsv.read(64)  # IDs are short; 64 bytes is plenty
                tab_pos = chunk.find(b"\t")
                if tab_pos == -1:
                    continue
                numeric_id = chunk[:tab_pos].decode("ascii")
                self._id_to_line[numeric_id] = line_num
        self._all_image_ids = list(self._id_to_line.keys())
        logger.info("Indexed %d images in TSV.", len(self._all_image_ids))

        # ------------------------------------------------------------------
        # 4. Validate gold image IDs and build pipeline records
        # ------------------------------------------------------------------
        missing_total = 0
        for claim in self._claims:
            gold = claim.get("gold", {})
            raw_ids = gold.get("image_ids", [])
            valid_ids = [str(iid) for iid in raw_ids if str(iid) in self._id_to_line]
            if len(valid_ids) < len(raw_ids):
                missing_total += len(raw_ids) - len(valid_ids)
            claim["_valid_gold_ids"] = valid_ids
        if missing_total:
            logger.warning(
                "%d gold image ID(s) not found in TSV and were dropped.", missing_total
            )

        # ------------------------------------------------------------------
        # 5. Compute splits via seeded shuffle
        # ------------------------------------------------------------------
        n = len(self._claims)
        indices = list(range(n))
        np.random.RandomState(cfg.seed).shuffle(indices)

        train_end = int(n * data_cfg.train_ratio)
        val_end = train_end + int(n * data_cfg.val_ratio)

        self._split_indices = {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:],
        }
        self._active_indices = indices  # default: all data
        logger.info(
            "Splits — train: %d, val: %d, test: %d",
            len(self._split_indices["train"]),
            len(self._split_indices["val"]),
            len(self._split_indices["test"]),
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._active_indices)

    def __getitem__(self, idx: int) -> dict:
        claim = self._claims[self._active_indices[idx]]
        return {
            "claim_id": claim["id"],
            "claim_text": claim["claim"],
            "gold_image_ids": claim["_valid_gold_ids"],
            "label": _LABEL_MAP[claim["label"]],
            "misinfo_type": claim.get("manipulation_type", "true").lower(),
        }

    def get_image(self, image_id: str) -> Image.Image:
        """Load a single image from the TSV by its numeric ID string."""
        line_num = self._id_to_line[image_id]
        offset = self._byte_offsets[line_num]
        with open(self._tsv_path, "rb") as tsv:
            tsv.seek(offset)
            raw_line = tsv.readline()
        parts = raw_line.split(b"\t", 1)
        img_bytes = base64.b64decode(parts[1])
        return Image.open(BytesIO(img_bytes)).convert("RGB")

    def get_all_image_ids(self) -> list[str]:
        """Return ALL image IDs from the TSV (full retrieval pool)."""
        return self._all_image_ids

    def get_split(self, split: str) -> "WebQADataset":
        if split not in self._split_indices:
            raise ValueError(f"Unknown split: {split!r}. Use 'train', 'val', or 'test'.")
        copy = WebQADataset.__new__(WebQADataset)
        copy._claims = self._claims
        copy._byte_offsets = self._byte_offsets
        copy._id_to_line = self._id_to_line
        copy._all_image_ids = self._all_image_ids
        copy._tsv_path = self._tsv_path
        copy._split_indices = self._split_indices
        copy._active_indices = self._split_indices[split]
        return copy
