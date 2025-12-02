# v30_model.py
#
# V30 Transformer-based sequence model for supervised trading decision learning.
#
# Input:
#   x: [batch_size, seq_len, feature_dim] float32
#
# Outputs:
#   logits_action: [batch_size, num_actions]
#   logits_risk:   [batch_size, num_risks]
#
# This model is designed to consume the datasets built by v30_dataset_builder.py:
#   X: [N, seq_len, feature_dim]
#   y_action: [N]
#   y_risk: [N]

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import math


@dataclass
class V30TransformerConfig:
    feature_dim: int = 9
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 128
    dropout: float = 0.1
    num_actions: int = 4
    num_risks: int = 3
    max_seq_len: int = 512


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T]


class V30TransformerModel(nn.Module):
    def __init__(self, config: V30TransformerConfig):
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.feature_dim, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        self.pos_encoder = PositionalEncoding(
            d_model=config.d_model,
            max_len=config.max_seq_len,
        )

        self.norm = nn.LayerNorm(config.d_model)

        self.action_head = nn.Linear(config.d_model, config.num_actions)
        self.risk_head = nn.Linear(config.d_model, config.num_risks)

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, F]
        """
        h = self.input_proj(x)         # [B, T, d_model]
        h = self.pos_encoder(h)        # add positional encoding
        h = self.transformer(h)        # [B, T, d_model]
        h = self.norm(h)

        # Use last token representation as pooled context
        ctx = h[:, -1, :]              # [B, d_model]

        logits_action = self.action_head(ctx)
        logits_risk = self.risk_head(ctx)
        return logits_action, logits_risk


def build_v30_transformer(
    feature_dim: int = 9,
    num_actions: int = 4,
    num_risks: int = 3,
    seq_len: int = 128,
) -> V30TransformerModel:
    cfg = V30TransformerConfig(
        feature_dim=feature_dim,
        num_actions=num_actions,
        num_risks=num_risks,
        max_seq_len=max(seq_len, 512),
    )
    return V30TransformerModel(cfg)


if __name__ == "__main__":
    # simple smoke test
    model = build_v30_transformer(feature_dim=9, num_actions=4, num_risks=3, seq_len=128)
    x = torch.randn(2, 128, 9)
    logits_action, logits_risk = model(x)
    print("logits_action:", logits_action.shape)
    print("logits_risk:", logits_risk.shape)
