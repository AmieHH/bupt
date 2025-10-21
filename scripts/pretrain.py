import argparse
import os

import torch
from torch.utils.data import DataLoader

from csi_loc.data.dataset import CSIDataset, Windowing
from csi_loc.models.mae import MaskedAutoencoder1D
from csi_loc.trainers.pretrain_trainer import PretrainTrainer
from csi_loc.utils.config import load_config, ensure_dir
from csi_loc.utils.train_utils import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = cfg["data_root"]
    # ensure numeric types (yaml may parse as strings in some environments)
    window_size = int(cfg["window_size"])
    stride = int(cfg["stride"])
    batch_size = int(cfg["batch_size"])
    lr = float(cfg["lr"])
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    mask_ratio = float(cfg["mask_ratio"])
    epochs = int(cfg["epochs"])

    win = Windowing(window_size, stride) 

    train_set = CSIDataset(data_root, "train", win, use_pos=False)
    val_set = CSIDataset(data_root, "val", win, use_pos=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    sample = next(iter(train_loader))
    input_dim = sample["csi"].shape[-1]

    model = MaskedAutoencoder1D(
        input_dim=input_dim,
        d_model=int(cfg["model"]["d_model"]),
        nhead=int(cfg["model"]["nhead"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dim_feedforward=int(cfg["model"]["dim_feedforward"]),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        decoder_depth=int(cfg["model"].get("decoder_depth", 2)),
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    trainer = PretrainTrainer(model, optim, mask_ratio, device)

    out_dir = cfg.get("out_dir", "outputs/pretrain")
    ensure_dir(out_dir)

    best = 1e9
    for epoch in range(1, epochs + 1):
        tr = trainer.train_epoch(train_loader)
        va = trainer.eval_epoch(val_loader)
        print({"epoch": epoch, "train": tr, "val": va})
        if va["loss"] < best:
            best = va["loss"]
            torch.save({"state_dict": model.state_dict(), "cfg": cfg}, os.path.join(out_dir, "best.pt"))


if __name__ == "__main__":
    main()
