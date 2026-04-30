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


def parse_label(raw):
    raw = raw.strip().lower()

    if raw == "benign":
        return 0
    if raw == "melanoma" or raw == "malignant":
        return 1
    if raw == "0" or raw == "0.0":
        return 0
    if raw == "1" or raw == "1.0":
        return 1

    raise ValueError("Bad label: " + raw)


def load_rows(csv_path):
    rows = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        for i, parts in enumerate(reader):
            if len(parts) < 2:
                continue

            image_id = parts[0].strip()
            label = parts[1].strip()

            if i == 0:
                if image_id.lower() in ["image", "image_id", "isic_id"]:
                    continue
                if label.lower() in ["label", "class", "target", "diagnosis", "melanoma"]:
                    continue

            rows.append((image_id, parse_label(label)))

    return rows


def filter_existing_rows(rows, image_dir, mask_dir):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    new_rows = []

    for image_id, label in rows:
        img_path = image_dir / (image_id + ".jpg")
        mask_path = mask_dir / (image_id + "_segmentation.png")

        if img_path.exists() and mask_path.exists():
            new_rows.append((image_id, label))

    return new_rows


def stratified_split(rows, val_ratio, seed):
    random.seed(seed)

    by_label = defaultdict(list)

    for i, row in enumerate(rows):
        label = row[1]
        by_label[label].append(i)

    train_idx = []
    val_idx = []

    for label in by_label:
        idxs = by_label[label]
        random.shuffle(idxs)

        n_val = int(round(len(idxs) * val_ratio))

        if len(idxs) > 1:
            n_val = max(1, min(len(idxs) - 1, n_val))
        else:
            n_val = 0

        val_idx += idxs[:n_val]
        train_idx += idxs[n_val:]

    random.shuffle(train_idx)
    random.shuffle(val_idx)

    return train_idx, val_idx


class LesionImageMaskConcatDataset(torch.utils.data.Dataset):
    def __init__(self, rows, image_dir, mask_dir, train=True, image_size=DEFAULT_IMG_SIZE):
        self.rows = rows
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.train = train
        self.image_size = image_size

    def __len__(self):
        return len(self.rows)

    def augment(self, image, mask):
        if not self.train:
            return image, mask

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        angle = random.uniform(-25, 25)
        image = TF.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)

        return image, mask

    def __getitem__(self, index):
        image_id, label = self.rows[index]

        img_path = self.image_dir / (image_id + ".jpg")
        mask_path = self.mask_dir / (image_id + "_segmentation.png")

        image = read_image(str(img_path))
        mask = read_image(str(mask_path))

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if mask.shape[0] > 1:
            mask = mask[:1]

        image, mask = self.augment(image, mask)

        image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=transforms.InterpolationMode.NEAREST,
            antialias=False,
        )

        image = image.float() / 255.0
        mask = mask.float() / 255.0
        mask = (mask > 0.5).float()

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        x = torch.cat([image, mask], dim=0)

        return x, torch.tensor(label, dtype=torch.float32)


def make_weighted_sampler(indices, rows):
    labels = [rows[i][1] for i in indices]

    n0 = max(1, labels.count(0))
    n1 = max(1, labels.count(1))

    weights = []

    for i in indices:
        if rows[i][1] == 1:
            weights.append(1.0 / n1)
        else:
            weights.append(1.0 / n0)

    return WeightedRandomSampler(weights, num_samples=len(indices), replacement=True)


def make_loaders(image_dir, mask_dir, label_csv, val_ratio, batch_size, num_workers, seed, use_weighted_sampler=True):
    rows = load_rows(label_csv)
    rows = filter_existing_rows(rows, image_dir, mask_dir)

    if len(rows) == 0:
        raise ValueError("No image/mask pairs found. Check paths.")

    train_idx, val_idx = stratified_split(rows, val_ratio, seed)

    train_data = LesionImageMaskConcatDataset(rows, image_dir, mask_dir, train=True)
    val_data = LesionImageMaskConcatDataset(rows, image_dir, mask_dir, train=False)

    train_set = Subset(train_data, train_idx)
    val_set = Subset(val_data, val_idx)

    if use_weighted_sampler:
        sampler = make_weighted_sampler(train_idx, rows)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, rows