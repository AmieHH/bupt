from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Windowing:
    window_size: int
    stride: int


class CSIDataset(Dataset):
    """
    Generic CSI dataset loading .npz files with keys:
      - 'csi': float32 array of shape [T, D]
      - 'pos': float32 array of shape [T, 2] (optional)
      - 'pos_weak': float32 array of shape [T, 2] (optional)
    Returns windowed samples with length 'window_size'.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        windowing: Windowing,
        use_pos: bool = False,
        drop_last_incomplete: bool = True,
    ) -> None:
        super().__init__()
        assert split in {"train", "val", "test"}
        self.root_dir = os.path.join(root_dir, split)
        self.window_size = windowing.window_size
        self.stride = windowing.stride
        self.use_pos = use_pos
        self.drop_last_incomplete = drop_last_incomplete

        files = sorted(glob.glob(os.path.join(self.root_dir, "*.npz")))
        if not files:
            raise FileNotFoundError(f"No .npz files in {self.root_dir}")
        self.index: List[Tuple[str, int]] = []  # (file_path, start_idx)

        for fp in files:
            with np.load(fp) as data:
                csi = data["csi"].astype(np.float32)
                T = csi.shape[0]
            last_start = T - self.window_size
            if last_start < 0:
                continue
            starts = list(range(0, last_start + 1, self.stride))
            if not self.drop_last_incomplete and (starts[-1] != last_start):
                starts.append(last_start)
            for s in starts:
                self.index.append((fp, s))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fp, s = self.index[idx]
        with np.load(fp) as data:
            csi = data["csi"][s : s + self.window_size].astype(np.float32)
            item: Dict[str, torch.Tensor] = {
                "csi": torch.from_numpy(csi),  # [L, D]
            }
            if self.use_pos and ("pos" in data):
                pos = data["pos"][s : s + self.window_size].astype(np.float32)
                item["pos"] = torch.from_numpy(pos)
            if self.use_pos and ("pos_weak" in data):
                posw = data["pos_weak"][s : s + self.window_size].astype(np.float32)
                item["pos_weak"] = torch.from_numpy(posw)
        return item
