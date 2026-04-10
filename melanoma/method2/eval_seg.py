"""Report validation Dice (soft) and IoU (binary @ 0.5) for a saved U-Net checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from melanoma.config import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_LABEL_CSV,
    DEFAULT_SEG_MASK_DIR,
    DEFAULT_UNET_CKPT,
)
from melanoma.method1.data import load_rows, stratified_split
from melanoma.method2.data_seg import LesionSegDataset, filter_rows_with_masks
from melanoma.method2.seg_metrics import mean_dice_soft, mean_iou_binary
from melanoma.method2.unet import build_unet


def main() -> None:
    p = argparse.ArgumentParser(description="U-Net segmentation metrics on stratified val split.")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_UNET_CKPT)
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    p.add_argument("--label-csv", type=Path, default=DEFAULT_LABEL_CSV)
    p.add_argument("--mask-dir", type=Path, default=DEFAULT_SEG_MASK_DIR)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--iou-thresh", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows(args.label_csv)
    rows = filter_rows_with_masks(rows, args.image_dir, args.mask_dir)
    _, val_idx = stratified_split(rows, val_ratio=args.val_ratio, seed=args.seed)
    ds = LesionSegDataset(rows, args.image_dir, args.mask_dir)
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_unet().to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=True)

    dice = mean_dice_soft(model, val_loader, device)
    iou = mean_iou_binary(model, val_loader, device, thresh=args.iou_thresh)
    print(f"val_samples: {len(val_idx)}  dice_soft: {dice:.4f}  iou@{args.iou_thresh}: {iou:.4f}")


if __name__ == "__main__":
    main()
