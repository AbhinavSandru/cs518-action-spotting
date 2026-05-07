import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionSpotter(nn.Module):
    def __init__(
        self,
        input_dim=2048,
        num_classes=17,
        d_model=256,
        nhead=4,
        num_layers=4,
        dropout=0.3,
    ):
        """
        Transformer encoder for action spotting on precomputed ResNet-152 features.

        Args:
            input_dim:   feature dimension from ResNet-152 (512)
            num_classes: number of action classes (17 for SoccerNet-v2)
            d_model:     internal transformer dimension
            nhead:       number of attention heads
            num_layers:  number of transformer encoder layers
            dropout:     dropout probability after attention layers
        """
        super().__init__()

        # Project input features to transformer dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding (learned)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Per-frame classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, window_size, input_dim)
        Returns:
            logits: (batch, window_size, num_classes)
        """
        B, T, _ = x.shape

        x = F.normalize(x, dim=-1)                  # L2 normalize input features
        x = self.input_proj(x)                      # (B, T, d_model)
        x = x + self.pos_embedding[:, :T, :]        # add positional encoding
        x = self.transformer(x)                     # (B, T, d_model)
        logits = self.classifier(x)                 # (B, T, num_classes)
        return logits
