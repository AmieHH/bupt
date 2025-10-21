from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def random_mask(L: int, mask_ratio: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (keep_indices, mask_indices) as boolean masks of shape [L]."""
    num_mask = int(L * mask_ratio)
    perm = torch.randperm(L, device=device)
    mask_idx = perm[:num_mask]
    keep_idx = perm[num_mask:]
    mask = torch.zeros(L, dtype=torch.bool, device=device)
    mask[mask_idx] = True
    keep = ~mask
    return keep, mask


class MaskedAutoencoder1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        decoder_depth: int = 2,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(d_model, nhead, num_layers, dim_feedforward, dropout)
        self.mask_token = nn.Parameter(torch.zeros(d_model))

        # simple decoder
        dec_layers = []
        for _ in range(decoder_depth):
            dec_layers.append(nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ))
        self.decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, norm_first=True
        ), num_layers=decoder_depth)
        self.proj = nn.Linear(d_model, input_dim)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B, L, D_in] -> recon: [B, L, D_in], mask: [B, L]"""
        B, L, _ = x.shape
        device = x.device
        x_emb = self.embed(x)
        x_emb = self.pos_enc(x_emb)

        # same mask across batch
        keep, mask = random_mask(L, mask_ratio, device)
        keep_idx = keep.nonzero(as_tuple=False).squeeze(1)
        mask_idx = mask.nonzero(as_tuple=False).squeeze(1)

        visible = x_emb[:, keep_idx, :]
        enc = self.encoder(visible)

        # reconstruct full sequence ordering
        out_tokens = torch.zeros(B, L, enc.size(-1), device=device)
        out_tokens[:, keep_idx, :] = enc
        out_tokens[:, mask_idx, :] = self.mask_token.view(1, 1, -1)
        out_tokens = self.pos_enc(out_tokens)
        dec = self.decoder(out_tokens)
        recon = self.proj(dec)
        return recon, mask.unsqueeze(0).expand(B, -1)
