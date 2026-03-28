"""Fusion classifiers for combining visual + text modality features."""

import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    """Score-level fusion: concatenate visual + text feature vectors and classify.

    Default input: 6 CLIP visual features + 6 NLI text features = 12-dim.
    """

    def __init__(self, visual_dim=6, text_dim=6, hidden_dims=None,
                 num_classes=3, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        input_dim = visual_dim + text_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.network = nn.Sequential(*layers)
        self.visual_dim = visual_dim
        self.text_dim = text_dim

    def forward(self, visual_features, text_features):
        """Forward pass.

        Args:
            visual_features: [B, visual_dim]
            text_features: [B, text_dim]

        Returns:
            Logits [B, num_classes].
        """
        x = torch.cat([visual_features, text_features], dim=1)
        return self.network(x)


class GatedFusionClassifier(nn.Module):
    """Gated fusion: learned per-sample gates weight visual vs text features.

    gate = sigmoid(W_g * [visual; text] + b_g)
    fused = gate * visual + (1 - gate) * text
    output = classifier(fused)
    """

    def __init__(self, visual_dim=6, text_dim=6, hidden_dim=256,
                 num_classes=3, dropout=0.3):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim

        # Project both modalities to same dimension
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
        )

        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(visual_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, visual_features, text_features):
        """Forward pass with gated fusion.

        Args:
            visual_features: [B, visual_dim]
            text_features: [B, text_dim]

        Returns:
            Logits [B, num_classes].
        """
        v = self.visual_proj(visual_features)  # [B, H]
        t = self.text_proj(text_features)      # [B, H]

        concat = torch.cat([visual_features, text_features], dim=1)  # [B, V+T]
        g = self.gate(concat)  # [B, H]

        fused = g * v + (1 - g) * t  # [B, H]
        return self.classifier(fused)
