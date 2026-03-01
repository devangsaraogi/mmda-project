import logging

import torch
import torch.nn.functional as F
import open_clip
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """Frozen CLIP ViT-L/14 wrapper for encoding images and text."""

    def __init__(self, cfg):
        self.device = torch.device(cfg.clip.device if torch.cuda.is_available() or cfg.clip.device == "cpu" else "cpu")
        self.batch_size = cfg.clip.batch_size
        self.embed_dim = cfg.clip.embed_dim

        logger.info(f"Loading CLIP model {cfg.clip.model_name} ({cfg.clip.pretrained}) on {self.device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            cfg.clip.model_name, pretrained=cfg.clip.pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(cfg.clip.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        logger.info(f"CLIP model loaded. Embedding dim: {self.embed_dim}")

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image], batch_size: int = None) -> torch.Tensor:
        """Encode PIL images to L2-normalized embeddings.

        Args:
            images: List of PIL images.
            batch_size: Override default batch size.

        Returns:
            Tensor of shape [N, embed_dim], L2-normalized.
        """
        bs = batch_size or self.batch_size
        all_embeds = []

        for i in range(0, len(images), bs):
            batch = images[i : i + bs]
            tensors = torch.stack([self.preprocess(img) for img in batch]).to(self.device)
            embeds = self.model.encode_image(tensors)
            embeds = F.normalize(embeds, dim=-1)
            all_embeds.append(embeds.cpu())

        return torch.cat(all_embeds, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = None) -> torch.Tensor:
        """Encode text strings to L2-normalized embeddings.

        Args:
            texts: List of text strings.
            batch_size: Override default batch size.

        Returns:
            Tensor of shape [N, embed_dim], L2-normalized.
        """
        bs = batch_size or self.batch_size
        all_embeds = []

        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            tokens = self.tokenizer(batch).to(self.device)
            embeds = self.model.encode_text(tokens)
            embeds = F.normalize(embeds, dim=-1)
            all_embeds.append(embeds.cpu())

        return torch.cat(all_embeds, dim=0)

    def compute_similarity(self, text_emb: torch.Tensor, image_embs: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between one text embedding and multiple image embeddings.

        Args:
            text_emb: Shape [embed_dim] or [1, embed_dim].
            image_embs: Shape [N, embed_dim].

        Returns:
            Tensor of shape [N] with cosine similarities.
        """
        if text_emb.dim() == 1:
            text_emb = text_emb.unsqueeze(0)
        # Both are already L2-normalized, so dot product = cosine similarity
        return (text_emb @ image_embs.T).squeeze(0)
