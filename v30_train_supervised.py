# v30_train_supervised.py
#
# Supervised training script for V30 Transformer model.
# Trains on teacher-generated sequence datasets (.npz) from v30_dataset_builder.py.
#
# Usage (example):
#   python v30_train_supervised.py --symbol BTCUSDT --timeframe 1h \
#       --data-dir datasets --model-dir models_v30 --epochs 30 --batch-size 128

from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from v30_model import build_v30_transformer


class V30SequenceDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X = data["X"].astype("float32")        # [N, T, F]
        self.y_action = data["y_action"].astype("int64")
        self.y_risk = data["y_risk"].astype("int64")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.X[idx],             # [T, F]
            self.y_action[idx],      # int
            self.y_risk[idx],        # int
        )


def load_datasets(
    symbol: str,
    timeframe: str,
    data_dir: str,
) -> Tuple[V30SequenceDataset, V30SequenceDataset, V30SequenceDataset]:
    base = f"v30_{symbol}_{timeframe}"
    train_path = os.path.join(data_dir, base + "_train.npz")
    valid_path = os.path.join(data_dir, base + "_valid.npz")
    test_path = os.path.join(data_dir, base + "_test.npz")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(valid_path):
        raise FileNotFoundError(f"Valid file not found: {valid_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_ds = V30SequenceDataset(train_path)
    valid_ds = V30SequenceDataset(valid_path)
    test_ds = V30SequenceDataset(test_path)
    return train_ds, valid_ds, test_ds


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Return (action_accuracy, risk_accuracy) on given loader."""
    model.eval()
    correct_action = 0
    correct_risk = 0
    total = 0

    with torch.no_grad():
        for X, y_action, y_risk in loader:
            X = X.to(device)
            y_action = y_action.to(device)
            y_risk = y_risk.to(device)

            logits_action, logits_risk = model(X)
            pred_action = logits_action.argmax(dim=-1)
            pred_risk = logits_risk.argmax(dim=-1)

            correct_action += (pred_action == y_action).sum().item()
            correct_risk += (pred_risk == y_risk).sum().item()
            total += X.size(0)

    acc_action = correct_action / max(total, 1)
    acc_risk = correct_risk / max(total, 1)
    return acc_action, acc_risk


def train(
    symbol: str,
    timeframe: str = "1h",
    data_dir: str = "datasets",
    model_dir: str = "models_v30",
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    lambda_risk: float = 0.3,
    device_str: str = "auto",
) -> str:
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    os.makedirs(model_dir, exist_ok=True)

    train_ds, valid_ds, test_ds = load_datasets(symbol, timeframe, data_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    seq_len = train_ds.X.shape[1]
    feature_dim = train_ds.X.shape[2]

    model = build_v30_transformer(
        feature_dim=feature_dim,
        num_actions=int(train_ds.y_action.max()) + 1,
        num_risks=int(train_ds.y_risk.max()) + 1,
        seq_len=seq_len,
    ).to(device)

    print(f"[V30 Train] symbol={symbol}, timeframe={timeframe}, device={device}")
    print(f"[V30 Train] seq_len={seq_len}, feature_dim={feature_dim}")
    print(f"[V30 Train] actions={int(train_ds.y_action.max()) + 1}, risks={int(train_ds.y_risk.max()) + 1}")

    criterion_action = nn.CrossEntropyLoss()
    criterion_risk = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_valid_acc = 0.0
    best_path = os.path.join(model_dir, f"v30_transformer_{symbol}_{timeframe}_best.pt")
    epochs_no_improve = 0
    early_stop_patience = 8

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for X, y_action, y_risk in train_loader:
            X = X.to(device)
            y_action = y_action.to(device)
            y_risk = y_risk.to(device)

            logits_action, logits_risk = model(X)
            loss_action = criterion_action(logits_action, y_action)
            loss_risk = criterion_risk(logits_risk, y_risk)
            loss = loss_action + lambda_risk * loss_risk

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * X.size(0)

        epoch_loss /= max(len(train_ds), 1)
        acc_action_valid, acc_risk_valid = evaluate_model(model, valid_loader, device)
        valid_metric = acc_action_valid

        scheduler.step(valid_metric)

        dt = time.time() - t0
        print(
            f"[Epoch {epoch:02d}] loss={epoch_loss:.4f} "
            f"valid_acc_action={acc_action_valid:.4f} valid_acc_risk={acc_risk_valid:.4f} "
            f"time={dt:.1f}s"
        )

        if valid_metric > best_valid_acc:
            best_valid_acc = valid_metric
            epochs_no_improve = 0
            torch.save(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "seq_len": seq_len,
                    "feature_dim": feature_dim,
                    "model_state_dict": model.state_dict(),
                },
                best_path,
            )
            print(f"[V30 Train] New best model saved to {best_path}, valid_acc_action={best_valid_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"[V30 Train] No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= early_stop_patience:
            print("[V30 Train] Early stopping triggered.")
            break

    # Load best and evaluate on test set
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[V30 Train] Loaded best model from {best_path} for final test evaluation.")

    acc_action_test, acc_risk_test = evaluate_model(model, test_loader, device)
    print(f"[V30 Test] acc_action={acc_action_test:.4f}, acc_risk={acc_risk_test:.4f}")

    return best_path


def parse_args():
    p = argparse.ArgumentParser(description="V30 Transformer supervised training")
    p.add_argument("--symbol", type=str, default="BTCUSDT", help="symbol, e.g., BTCUSDT")
    p.add_argument("--timeframe", type=str, default="1h", help="timeframe, e.g., 1h")
    p.add_argument("--data-dir", type=str, default="datasets", help="directory containing v30_*.npz files")
    p.add_argument("--model-dir", type=str, default="models_v30", help="directory to save trained models")
    p.add_argument("--epochs", type=int, default=30, help="max training epochs")
    p.add_argument("--batch-size", type=int, default=128, help="batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("--lambda-risk", type=float, default=0.3, help="weight for risk classification loss")
    p.add_argument("--device", type=str, default="auto", help="device: auto/cuda/cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    best = train(
        symbol=args.symbol.upper(),
        timeframe=args.timeframe,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_risk=args.lambda_risk,
        device_str=args.device,
    )
    print(f"[V30 Train] Done. Best model: {best}")
