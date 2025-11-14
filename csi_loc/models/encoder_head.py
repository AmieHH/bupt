from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .mae import PositionalEncoding, TransformerEncoder


class CSILocEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(d_model, nhead, num_layers, dim_feedforward, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D_in]
        x = self.embed(x)
        x = self.pos_enc(x)
        h = self.encoder(x)  # [B, L, D]
        return h


class RegressionHead(nn.Module):
    def __init__(self, d_model: int, aggregation: Literal["mean", "last"] = "mean") -> None:
        super().__init__()
        self.aggregation = aggregation
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 2),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, L, D]
        if self.aggregation == "mean":
            pooled = h.mean(dim=1)
        else:
            pooled = h[:, -1, :]
        return self.mlp(pooled)  # [B, 2]
