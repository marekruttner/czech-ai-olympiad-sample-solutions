import argparse
import csv
import os
import random
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


IOAI_REPO = "https://github.com/IOAI-official/IOAI-2025.git"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RadarTrainDataset(Dataset):
    def __init__(self, folder: Path):
        self.folder = folder
        self.files = sorted(folder.glob("*.mat.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .mat.pt files found in {folder}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.files[idx]
        tensor = torch.load(f, map_location="cpu")
        if tensor.shape != (7, 50, 181):
            raise ValueError(f"Unexpected tensor shape {tensor.shape} in {f}")
        x = tensor[:6].float()
        y = tensor[6].long() + 1  # [-1,3] -> [0,4]
        return x, y


class RadarInferenceDataset(Dataset):
    def __init__(self, folder: Path):
        self.folder = folder
        self.files = sorted(folder.glob("*.mat.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .mat.pt files found in {folder}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        f = self.files[idx]
        tensor = torch.load(f, map_location="cpu")
        if tensor.shape == (7, 50, 181):
            x = tensor[:6].float()
        elif tensor.shape == (6, 50, 181):
            x = tensor.float()
        else:
            raise ValueError(f"Unexpected tensor shape {tensor.shape} in {f}")
        return x, f.name


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 5, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class BatchResult:
    loss: float
    correct_bg: int
    total_bg: int
    correct_obj: int
    total_obj: int


def ioai_score_from_counts(correct_bg: int, total_bg: int, correct_obj: int, total_obj: int) -> float:
    numerator = correct_bg * 1 + correct_obj * 50
    denominator = total_bg * 1 + total_obj * 50
    return float(numerator / denominator) if denominator > 0 else 0.0


def evaluate_batch(logits: torch.Tensor, labels: torch.Tensor, loss_fn: nn.Module) -> BatchResult:
    loss = loss_fn(logits, labels)
    preds = torch.argmax(logits, dim=1)

    bg_mask = labels == 0
    obj_mask = labels != 0

    correct_bg = int((preds[bg_mask] == labels[bg_mask]).sum().item())
    total_bg = int(bg_mask.sum().item())
    correct_obj = int((preds[obj_mask] == labels[obj_mask]).sum().item())
    total_obj = int(obj_mask.sum().item())

    return BatchResult(
        loss=float(loss.item()),
        correct_bg=correct_bg,
        total_bg=total_bg,
        correct_obj=correct_obj,
        total_obj=total_obj,
    )


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    cb = tb = co = to = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            result = evaluate_batch(logits, y, loss_fn)
            loss = loss_fn(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += result.loss
        cb += result.correct_bg
        tb += result.total_bg
        co += result.correct_obj
        to += result.total_obj

    avg_loss = total_loss / len(loader)
    score = ioai_score_from_counts(cb, tb, co, to)
    return avg_loss, score


def find_data_dirs(root: Path) -> Tuple[Path, Path, Path]:
    radar_root = root / "Individual-Contest" / "Radar"
    train = radar_root / "training_set"

    val_candidates = [
        radar_root / "validation_set",
        radar_root / "val_set",
        radar_root / "Solution" / "validation_set",
    ]
    test_candidates = [
        radar_root / "test_set",
        radar_root / "testing_set",
        radar_root / "Solution" / "test_set",
    ]

    val = next((p for p in val_candidates if p.exists()), None)
    test = next((p for p in test_candidates if p.exists()), None)

    if not train.exists():
        raise FileNotFoundError(f"Training directory not found: {train}")
    if val is None:
        raise FileNotFoundError("Validation directory not found (expected validation_set or val_set)")
    if test is None:
        raise FileNotFoundError("Test directory not found (expected test_set or testing_set)")

    return train, val, test


def clone_ioai_repo(target_dir: Path) -> None:
    if target_dir.exists() and (target_dir / ".git").exists():
        print(f"Dataset repo already exists at {target_dir}")
        return

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", IOAI_REPO, str(target_dir)
    ], check=True)
    subprocess.run(["git", "-C", str(target_dir), "sparse-checkout", "set", "Individual-Contest/Radar"], check=True)


def write_submission(model: nn.Module, dataset: RadarInferenceDataset, device: torch.device, out_csv: Path) -> None:
    model.eval()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["filename"] + [f"pixel_{i}" for i in range(50 * 181)]
        writer.writerow(header)

        for x, fname in dataset:
            with torch.no_grad():
                logits = model(x.unsqueeze(0).to(device))
                pred = torch.argmax(logits, dim=1).squeeze(0).cpu() - 1  # [0,4] -> [-1,3]
            row = [fname] + pred.flatten().tolist()
            writer.writerow(row)


def zip_submissions(val_csv: Path, test_csv: Path, out_zip: Path) -> None:
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(val_csv, arcname=val_csv.name)
        zf.write(test_csv, arcname=test_csv.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline IOAI 2025 Radar segmentation training")
    parser.add_argument("--data-dir", type=Path, default=Path("data/ioai-2025"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true", help="Clone IOAI repo and fetch Radar dataset")
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.download:
        clone_ioai_repo(args.data_dir)

    train_dir, val_dir, test_dir = find_data_dirs(args.data_dir)
    print(f"Train: {train_dir}\nVal: {val_dir}\nTest: {test_dir}")

    train_ds = RadarTrainDataset(train_dir)
    val_ds = RadarInferenceDataset(val_dir)
    test_ds = RadarInferenceDataset(test_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_score = run_epoch(model, train_loader, loss_fn, device, optimizer)
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} train_score={train_score:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "baseline_cnn.pt")

    val_csv = args.output_dir / "submission_val.csv"
    test_csv = args.output_dir / "submission_test.csv"
    zip_path = args.output_dir / "submission.zip"

    write_submission(model, val_ds, device, val_csv)
    write_submission(model, test_ds, device, test_csv)
    zip_submissions(val_csv, test_csv, zip_path)

    print(f"Saved model: {args.output_dir / 'baseline_cnn.pt'}")
    print(f"Saved submissions: {val_csv}, {test_csv}")
    print(f"Saved archive: {zip_path}")


if __name__ == "__main__":
    main()
