from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from csi_loc.losses.losses import masked_mse_loss
from csi_loc.utils.train_utils import AverageMeter, to_device


class PretrainTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        mask_ratio: float,
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.mask_ratio = mask_ratio
        self.device = device

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        loss_meter = AverageMeter()
        for batch in tqdm(loader, desc="pretrain", leave=False):
            batch = to_device(batch, self.device)
            csi = batch["csi"].float()
            recon, mask = self.model(csi, self.mask_ratio)
            loss = masked_mse_loss(recon, csi, mask)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            loss_meter.update(loss.item(), n=csi.size(0))
        return {"loss": loss_meter.avg}

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        loss_meter = AverageMeter()
        for batch in tqdm(loader, desc="pretrain-val", leave=False):
            batch = to_device(batch, self.device)
            csi = batch["csi"].float()
            recon, mask = self.model(csi, self.mask_ratio)
            loss = masked_mse_loss(recon, csi, mask)
            loss_meter.update(loss.item(), n=csi.size(0))
        return {"loss": loss_meter.avg}
