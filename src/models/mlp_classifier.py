import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """Trainable MLP classifier on CLIP features for verdict prediction.

    Supports two input modes:
        - "features": 6-dim hand-crafted similarity features
          (max_sim, mean_sim, min_sim, std_sim, top1_sim, top3_mean_sim)
        - "embeddings": 1536-dim concatenated text + image embeddings (768 + 768)
    """

    INPUT_DIMS = {"features": 6, "embeddings": 1536}

    def __init__(self, cfg):
        super().__init__()
        mlp_cfg = cfg.verification.mlp
        self.input_mode = mlp_cfg.input_mode
        input_dim = self.INPUT_DIMS[self.input_mode]
        hidden_dims = list(mlp_cfg.hidden_dims)
        dropout = mlp_cfg.dropout
        num_classes = mlp_cfg.num_classes

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, input_dim].

        Returns:
            Logits of shape [B, num_classes].
        """
        return self.network(x)
