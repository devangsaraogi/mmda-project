"""Cross-attention and confidence-weighted fusion classifiers.

Companions to ``src/models/fusion_classifier.py`` (which holds the
concat-MLP and gated variants we used at midterm). These add two modern
fusion strategies we can compare against in the final paper:

  - ``CrossAttentionFusion``: project each modality to a common hidden
    dim, then treat them as two tokens of a mini-transformer with a
    single cross-attention layer. Output of the [CLS]-like pooled
    representation goes into the verdict head. Inspired by HAMMER
    (Shao et al. 2023 / DGM4) and the cross-modal attention line of
    work in multimodal misinformation detection.

  - ``ConfidenceWeightedFusion``: given pre-computed unimodal logit
    probabilities (from each modality's own MLP), weight each modality
    by 1 - entropy(p_modality) normalised per sample. A cheap,
    principled late-fusion baseline. Requires the caller to pass the
    unimodal probability vectors alongside the feature vectors.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Single cross-attention layer over two modality tokens."""

    def __init__(
        self,
        visual_dim: int = 6,
        text_dim: int = 6,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim

        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        # Learned modality token embeddings (so the attention knows which
        # token is which; small but helps on tiny 2-token sequences).
        self.mod_embed = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        v = self.visual_proj(visual_features)  # [B, H]
        t = self.text_proj(text_features)      # [B, H]
        tokens = torch.stack([v, t], dim=1)    # [B, 2, H]
        tokens = tokens + self.mod_embed.unsqueeze(0)  # add modality positional

        attn_out, _ = self.attn(tokens, tokens, tokens)  # [B, 2, H]
        x = self.norm1(tokens + attn_out)
        x = self.norm2(x + self.ff(x))

        pooled = x.mean(dim=1)  # [B, H]  (2-token mean, cheap and works)
        return self.classifier(pooled)


class ConfidenceWeightedFusion(nn.Module):
    """Late fusion weighted by each modality's per-sample confidence.

    Given unimodal probability vectors ``p_v`` and ``p_t`` (shape
    ``[B, C]``), the fused logit is a convex combination weighted by
    ``1 - H(p)`` (normalised), where ``H`` is the Shannon entropy of
    the unimodal prediction. A tiny MLP on the fused representation
    produces the final logits, so the model can still learn corrective
    transformations beyond the raw weighted-average.

    ``forward`` signature diverges from the other fusion classes: it
    takes the unimodal *probabilities*, not raw feature vectors. The
    caller is expected to pre-compute them from each modality's MLP.
    """

    def __init__(
        self,
        num_classes: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        entropy_eps: float = 1e-8,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.entropy_eps = entropy_eps

        self.head = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _entropy(self, p: torch.Tensor) -> torch.Tensor:
        # Shannon entropy per sample; shape [B].
        return -(p * (p + self.entropy_eps).log()).sum(dim=-1)

    def forward(self, p_visual: torch.Tensor, p_text: torch.Tensor) -> torch.Tensor:
        # Confidence = 1 - normalised entropy. Higher => more confident.
        max_ent = torch.log(torch.tensor(float(self.num_classes),
                                          device=p_visual.device))
        c_v = 1.0 - self._entropy(p_visual) / max_ent   # [B]
        c_t = 1.0 - self._entropy(p_text) / max_ent     # [B]
        # Normalise so weights sum to 1 per sample.
        w_sum = (c_v + c_t).clamp(min=self.entropy_eps)
        w_v = (c_v / w_sum).unsqueeze(-1)  # [B, 1]
        w_t = (c_t / w_sum).unsqueeze(-1)

        fused = w_v * p_visual + w_t * p_text  # [B, C]
        return self.head(fused)
