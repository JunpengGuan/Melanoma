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


def _parse_label(raw: str, *, row_num: int, csv_path: Path) -> int:
    """
    Strict label parsing:
      0 = benign
      1 = melanoma

    Accepts:
      benign / melanoma
      malignant (treated as melanoma for backward compatibility)
      0 / 1 / 0.0 / 1.0
    """
    low = raw.strip().replace("\r", "").lower()

    if low == "benign":
        return 0
    if low in ("melanoma", "malignant"):
        return 1
    if low in ("0", "0.0"):
        return 0
    if low in ("1", "1.0"):
        return 1

    raise ValueError(
        f"Invalid label '{raw}' at row {row_num} in {csv_path}. "
        "Expected benign / melanoma / malignant / 0 / 1."
    )


def load_rows(csv_path: Path) -> list[tuple[str, int]]:
    """Return (image_id, label) with label 1=melanoma, 0=benign.

    Accepts string labels:
      benign / melanoma
    and, for backward compatibility:
      malignant -> melanoma

    Also accepts numeric labels:
      0 / 1 / 0.0 / 1.0
    """
    rows: list[tuple[str, int]] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row_num, parts in enumerate(reader, start=1):
            if not parts:
                continue
            if len(parts) < 2:
                raise ValueError(f"Row {row_num} in {csv_path} has fewer than 2 columns.")

            image_id = parts[0].strip()
            raw = parts[1].strip()

            if not image_id:
                raise ValueError(f"Empty image_id at row {row_num} in {csv_path}.")

            # Skip header row explicitly
            if row_num == 1:
                first = image_id.lower()
                second = raw.lower()
                if first in ("image", "image_id", "isic_id") or second in (
                    "label",
                    "class",
                    "target",
                    "diagnosis",
                    "groundtruth",
                    "ground_truth",
                ):
                    continue

            y = _parse_label(raw, row_num=row_num, csv_path=csv_path)
            rows.append((image_id, y))

    if not rows:
        raise ValueError(f"No valid rows found in {csv_path}.")

    return rows


def filter_existing_rows(rows: list[tuple[str, int]], image_dir: Path) -> list[tuple[str, int]]:
    """Keep only rows whose JPG exists."""
    image_dir = Path(image_dir)
    out: list[tuple[str, int]] = []
    for image_id, y in rows:
        if (image_dir / f"{image_id}.jpg").is_file():
            out.append((image_id, y))
    return out


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

    for _, idxs in by_label.items():
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
    rows = filter_existing_rows(rows, image_dir)

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
