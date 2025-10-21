from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mse_loss(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE only over masked positions.
    recon/target: [B, L, D], mask: [B, L] (True = masked)
    """
    diff = recon - target
    diff = diff.pow(2).mean(dim=-1)  # [B, L]
    loss = (diff * mask.float()).sum() / (mask.float().sum().clamp_min(1.0))
    return loss


class Huber2DLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(pred, target, delta=self.delta)


def trajectory_smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    """
    Encourage smooth trajectory by penalizing second differences.
    pred: [B, L, 2] or [B, 2] (if [B, 2], returns 0)
    """
    if pred.dim() == 2:
        return pred.new_tensor(0.0)
    if pred.size(1) < 3:
        return pred.new_tensor(0.0)
    d1 = pred[:, 1:, :] - pred[:, :-1, :]
    d2 = d1[:, 1:, :] - d1[:, :-1, :]
    return (d2.pow(2).sum(dim=-1).mean())


def trajectory_consistency_loss(pred: torch.Tensor, weak: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Encourage predicted displacements to match weak-label displacements.
    pred, weak: [B, L, 2]
    """
    if weak is None or pred.dim() != 3 or weak.dim() != 3:
        return pred.new_tensor(0.0)
    if pred.size(1) < 2:
        return pred.new_tensor(0.0)
    dp = pred[:, 1:, :] - pred[:, :-1, :]
    dw = weak[:, 1:, :] - weak[:, :-1, :]
    return F.smooth_l1_loss(dp, dw)
