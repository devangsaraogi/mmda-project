"""SigLIP 2 encoder — drop-in alternative to the CLIP encoder.

Uses ``transformers.AutoModel`` with the ``google/siglip2-base-patch16-256``
checkpoint (~86M params). Returns L2-normalised embeddings, same
contract as ``CLIPEncoder``. Used to answer the "is it the encoder, or
the candidate-pool scope?" question as an ablation.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch
from PIL import Image

try:
    from transformers import AutoModel, AutoProcessor
except Exception as exc:  # pragma: no cover
    AutoModel = None
    AutoProcessor = None
    _import_err = exc
else:
    _import_err = None

logger = logging.getLogger(__name__)


class SigLIPEncoder:
    """Drop-in SigLIP 2 encoder exposing the same methods as CLIPEncoder.

    The contract is deliberately narrow:
        .encode_images(pil_images, batch_size=...)  -> [N, D] float32 torch
        .encode_texts(strings)                      -> [N, D] float32 torch
        .embed_dim: int

    Both outputs are L2-normalised so cosine similarity reduces to dot
    product (same as the existing CLIP path).
    """

    DEFAULT_MODEL = "google/siglip2-base-patch16-256"

    def __init__(self, cfg, model_name: Optional[str] = None):
        if _import_err is not None:
            raise RuntimeError(
                f"transformers is required for SigLIPEncoder: {_import_err}"
            )
        self.model_name = model_name or self.DEFAULT_MODEL
        device_cfg = getattr(getattr(cfg, "siglip", object()), "device", None) \
            or getattr(cfg.clip, "device", "cuda")
        self.device = torch.device(device_cfg if torch.cuda.is_available() else "cpu")

        logger.info("Loading SigLIP model %s on %s", self.model_name, self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Inspect the embedding dim from a tiny dry-run.
        with torch.no_grad():
            dummy = self.processor(text=["dim probe"], return_tensors="pt",
                                   padding="max_length", truncation=True)
            dummy = {k: v.to(self.device) for k, v in dummy.items()}
            text_emb = self.model.get_text_features(**dummy)
        self.embed_dim = int(text_emb.shape[-1])
        logger.info("SigLIP model loaded. Embedding dim: %d", self.embed_dim)

    @torch.no_grad()
    def encode_images(
        self,
        images: List[Image.Image],
        batch_size: int = 64,
    ) -> torch.Tensor:
        if not images:
            return torch.empty(0, self.embed_dim)
        outs = []
        for i in range(0, len(images), batch_size):
            chunk = images[i:i + batch_size]
            batch = self.processor(images=chunk, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            emb = self.model.get_image_features(**batch)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            outs.append(emb.cpu().float())
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 256,
    ) -> torch.Tensor:
        if not texts:
            return torch.empty(0, self.embed_dim)
        outs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            batch = self.processor(
                text=chunk,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}
            emb = self.model.get_text_features(**batch)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            outs.append(emb.cpu().float())
        return torch.cat(outs, dim=0)
