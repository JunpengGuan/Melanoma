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


def segmentation_val_metrics(
    *,
    checkpoint: Path,
    image_dir: Path,
    label_csv: Path,
    mask_dir: Path,
    val_ratio: float = 0.15,
    seed: int = 42,
    batch_size: int = 8,
    num_workers: int = 2,
    iou_thresh: float = 0.5,
) -> dict:
    """Same stratified val split as train_seg; returns dice (soft) and IoU @ threshold."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows(label_csv)
    rows = filter_rows_with_masks(rows, image_dir, mask_dir)
    _, val_idx = stratified_split(rows, val_ratio=val_ratio, seed=seed)
    ds = LesionSegDataset(rows, image_dir, mask_dir)
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = build_unet().to(device)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=True)

    dice = mean_dice_soft(model, val_loader, device)
    iou = mean_iou_binary(model, val_loader, device, thresh=iou_thresh)
    return {
        "val_samples": len(val_idx),
        "dice_soft": float(dice),
        "iou": float(iou),
        "iou_thresh": float(iou_thresh),
        "device": str(device),
    }


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
    p.add_argument("--output", type=Path, default=None, help="Optional path to append metrics as text.")
    args = p.parse_args()

    m = segmentation_val_metrics(
        checkpoint=args.checkpoint,
        image_dir=args.image_dir,
        label_csv=args.label_csv,
        mask_dir=args.mask_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        iou_thresh=args.iou_thresh,
    )
    line = (
        f"val_samples: {m['val_samples']}  dice_soft: {m['dice_soft']:.4f}  "
        f"iou@{m['iou_thresh']}: {m['iou']:.4f}"
    )
    print(line)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
