from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import functional as TF

from melanoma.config import DEFAULT_IMG_SIZE


def load_rows(csv_path: Path) -> list[tuple[str, int]]:
    """Return (image_id, label) with label 1=malignant, 0=benign.

    Second column may be ``benign`` / ``malignant`` (training CSV) or ``0`` / ``1`` / ``0.0`` / ``1.0``
    (e.g. Part 3B test ground truth: 0=benign, 1=malignant).
    """
    rows: list[tuple[str, int]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for parts in reader:
            if not parts or len(parts) < 2:
                continue
            image_id = parts[0].strip()
            raw = parts[1].strip().replace("\r", "")
            low = raw.lower()
            if low in ("benign", "malignant"):
                y = 1 if low == "malignant" else 0
            else:
                try:
                    v = float(raw)
                except ValueError:
                    continue
                if v == 0.0:
                    y = 0
                elif v == 1.0:
                    y = 1
                else:
                    continue
            rows.append((image_id, y))
    return rows


def stratified_split(
    rows: list[tuple[str, int]],
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Index split stratified by label; guarantees at least one sample per class in val when possible."""
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for i, (_, y) in enumerate(rows):
        by_label[y].append(i)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for y, idxs in by_label.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_val = int(round(n * val_ratio))
        n_val = max(1, min(n - 1, n_val)) if n > 1 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


class LesionImageDataset(torch.utils.data.Dataset):
    """Loads ISIC_*.jpg from image_dir; rows from load_rows()."""

    def __init__(
        self,
        rows: list[tuple[str, int]],
        image_dir: Path,
        train: bool,
        image_size: int = DEFAULT_IMG_SIZE,
    ) -> None:
        self.rows = rows
        self.image_dir = Path(image_dir)
        self.train = train
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # x: CHW uint8
        if not self.train:
            return x
        if random.random() < 0.5:
            x = TF.hflip(x)
        if random.random() < 0.5:
            x = TF.vflip(x)
        angle = random.uniform(-25.0, 25.0)
        x = TF.rotate(x, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        return x

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id, y = self.rows[i]
        path = self.image_dir / f"{image_id}.jpg"
        x = read_image(str(path))
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        x = self._augment(x)
        x = TF.resize(x, [self.image_size, self.image_size], antialias=True)
        x = x.float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = (x - mean) / std
        return x, torch.tensor(y, dtype=torch.float32)


def make_weighted_sampler(indices: list[int], rows: list[tuple[str, int]]) -> WeightedRandomSampler:
    labels = [rows[i][1] for i in indices]
    n0 = max(1, labels.count(0))
    n1 = max(1, labels.count(1))
    w0 = 1.0 / n0
    w1 = 1.0 / n1
    weights = [w1 if rows[i][1] == 1 else w0 for i in indices]
    return WeightedRandomSampler(weights, num_samples=len(indices), replacement=True)


def make_loaders(
    image_dir: Path,
    label_csv: Path,
    val_ratio: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, list[tuple[str, int]]]:
    rows = load_rows(label_csv)
    train_idx, val_idx = stratified_split(rows, val_ratio=val_ratio, seed=seed)
    full_train = LesionImageDataset(rows, image_dir, train=True)
    full_eval = LesionImageDataset(rows, image_dir, train=False)
    train_set = Subset(full_train, train_idx)
    val_set = Subset(full_eval, val_idx)

    if use_weighted_sampler:
        sampler = make_weighted_sampler(train_idx, rows)
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, rows
