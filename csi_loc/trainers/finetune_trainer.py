from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from csi_loc.losses.losses import Huber2DLoss, trajectory_consistency_loss, trajectory_smoothness_loss
from csi_loc.utils.train_utils import AverageMeter, to_device


class FinetuneTrainer:
    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        lambda_consistency: float = 1.0,
        lambda_smooth: float = 0.1,
        use_sequence_output: bool = False,
    ) -> None:
        self.encoder = encoder
        self.head = head
        self.optimizer = optimizer
        self.device = device
        self.lambda_consistency = lambda_consistency
        self.lambda_smooth = lambda_smooth
        self.use_sequence_output = use_sequence_output
        self.reg_loss = Huber2DLoss(delta=1.0)

    def _forward(self, csi: torch.Tensor) -> torch.Tensor:
        h = self.encoder(csi)
        out = self.head(h)
        return out

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.encoder.train(); self.head.train()
        loss_meter = AverageMeter()
        for batch in tqdm(loader, desc="finetune", leave=False):
            batch = to_device(batch, self.device)
            csi = batch["csi"].float()
            posw: Optional[torch.Tensor] = batch.get("pos_weak")
            pos: Optional[torch.Tensor] = batch.get("pos")

            pred = self._forward(csi)
            if pred.dim() == 2 and pos is not None and pos.dim() == 3:
                # pool to last position label
                pos = pos[:, -1, :]
                if posw is not None and posw.dim() == 3:
                    posw = posw[:, -1, :]

            loss = torch.tensor(0.0, device=self.device)
            if pos is not None:
                loss = loss + self.reg_loss(pred, pos)

            # encourage temporal smoothness / consistency when sequence available
            if posw is not None and posw.dim() == 3:
                # reconstruct sequence predictions if head pools to a single vector
                if pred.dim() == 2:
                    pred_seq = pred.unsqueeze(1).repeat(1, csi.size(1), 1)
                else:
                    pred_seq = pred
                loss = loss + self.lambda_consistency * trajectory_consistency_loss(pred_seq, posw)
                loss = loss + self.lambda_smooth * trajectory_smoothness_loss(pred_seq)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.head.parameters()), 1.0)
            self.optimizer.step()

            loss_meter.update(loss.item(), n=csi.size(0))
        return {"loss": loss_meter.avg}

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.encoder.eval(); self.head.eval()
        loss_meter = AverageMeter()
        for batch in tqdm(loader, desc="finetune-val", leave=False):
            batch = to_device(batch, self.device)
            csi = batch["csi"].float()
            posw: Optional[torch.Tensor] = batch.get("pos_weak")
            pos: Optional[torch.Tensor] = batch.get("pos")
            pred = self._forward(csi)
            if pred.dim() == 2 and pos is not None and pos.dim() == 3:
                pos = pos[:, -1, :]
                if posw is not None and posw.dim() == 3:
                    posw = posw[:, -1, :]
            loss = torch.tensor(0.0, device=self.device)
            if pos is not None:
                loss = loss + self.reg_loss(pred, pos)
            if posw is not None and posw.dim() == 3:
                if pred.dim() == 2:
                    pred_seq = pred.unsqueeze(1).repeat(1, csi.size(1), 1)
                else:
                    pred_seq = pred
                loss = loss + self.lambda_consistency * trajectory_consistency_loss(pred_seq, posw)
                loss = loss + self.lambda_smooth * trajectory_smoothness_loss(pred_seq)
            loss_meter.update(loss.item(), n=csi.size(0))
        return {"loss": loss_meter.avg}
