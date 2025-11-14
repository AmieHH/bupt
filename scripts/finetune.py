import argparse
import os

import torch
from torch.utils.data import DataLoader

from csi_loc.data.dataset import CSIDataset, Windowing
from csi_loc.models.encoder_head import CSILocEncoder, RegressionHead
from csi_loc.models.mae import MaskedAutoencoder1D
from csi_loc.trainers.finetune_trainer import FinetuneTrainer
from csi_loc.utils.config import load_config, ensure_dir
from csi_loc.utils.train_utils import seed_everything


def load_pretrained(encoder: CSILocEncoder, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    enc_prefix = "encoder."
    mapping = {}
    for k, v in state.items():
        # Map MAE encoder to encoder.embed/pos_enc/encoder
        if k.startswith("embed."):
            mapping["embed." + k[len("embed."):]] = v
        elif k.startswith("pos_enc."):
            mapping["pos_enc." + k[len("pos_enc."):]] = v
        elif k.startswith("encoder."):
            mapping["encoder." + k[len("encoder."):]] = v
    missing, unexpected = encoder.load_state_dict(mapping, strict=False)
    print("Loaded pretrained. Missing:", missing, "Unexpected:", unexpected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--pretrained", type=str, required=False, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = cfg["data_root"]
    # ensure numeric types
    window_size = int(cfg["window_size"])
    stride = int(cfg["stride"])
    batch_size = int(cfg["batch_size"])
    d_model = int(cfg["model"]["d_model"])
    nhead = int(cfg["model"]["nhead"])
    num_layers = int(cfg["model"]["num_layers"])
    dim_feedforward = int(cfg["model"]["dim_feedforward"])
    dropout = float(cfg["model"].get("dropout", 0.1))
    lr = float(cfg["lr"])
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    lambda_consistency = float(cfg.get("lambda_consistency", 1.0))
    lambda_smooth = float(cfg.get("lambda_smooth", 0.1))
    epochs = int(cfg["epochs"])

    win = Windowing(window_size, stride) 

    train_set = CSIDataset(data_root, "train", win, use_pos=True)
    val_set = CSIDataset(data_root, "val", win, use_pos=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    sample = next(iter(train_loader))
    input_dim = sample["csi"].shape[-1]

    encoder = CSILocEncoder(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)

    if args.pretrained:
        load_pretrained(encoder, args.pretrained)

    head = RegressionHead(d_model, aggregation=cfg["head"].get("aggregation", "mean")).to(device)

    optim = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=lr, weight_decay=weight_decay)

    trainer = FinetuneTrainer(
        encoder,
        head,
        optim,
        device,
        lambda_consistency=lambda_consistency,
        lambda_smooth=lambda_smooth,
    )

    out_dir = cfg.get("out_dir", "outputs/finetune")
    ensure_dir(out_dir)

    best = 1e9
    for epoch in range(1, epochs + 1):
        tr = trainer.train_epoch(train_loader)
        va = trainer.eval_epoch(val_loader)
        print({"epoch": epoch, "train": tr, "val": va})
        if va["loss"] < best:
            best = va["loss"]
            torch.save({
                "encoder": encoder.state_dict(),
                "head": head.state_dict(),
                "cfg": cfg
            }, os.path.join(out_dir, "best.pt"))


if __name__ == "__main__":
    main()
