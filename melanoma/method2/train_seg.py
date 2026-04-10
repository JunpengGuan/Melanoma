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
from melanoma.method2.losses import SegmentationLoss
from melanoma.method2.seg_metrics import mean_dice_soft, mean_iou_binary
from melanoma.method2.unet import build_unet
from melanoma.train_report import merge_train_report


def main() -> None:
    p = argparse.ArgumentParser(description="Method 2 stage 1: train U-Net (CNN) lesion segmentation.")
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    p.add_argument("--label-csv", type=Path, default=DEFAULT_LABEL_CSV)
    p.add_argument("--mask-dir", type=Path, default=DEFAULT_SEG_MASK_DIR)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_UNET_CKPT)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = load_rows(args.label_csv)
    rows = filter_rows_with_masks(rows, args.image_dir, args.mask_dir)
    if len(rows) < 10:
        raise SystemExit("Too few samples with image+mask; check paths.")

    train_idx, val_idx = stratified_split(rows, val_ratio=args.val_ratio, seed=args.seed)
    ds = LesionSegDataset(rows, args.image_dir, args.mask_dir)
    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_unet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = SegmentationLoss(bce_weight=0.5)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    tr = vd = vi = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logit = model(x)
            loss = crit(logit, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        tr = running / max(1, n)
        vd = mean_dice_soft(model, val_loader, device)
        vi = mean_iou_binary(model, val_loader, device, thresh=0.5)
        print(f"epoch {epoch:03d}  train_loss={tr:.4f}  val_dice={vd:.4f}  val_iou={vi:.4f}")

    torch.save({"model": model.state_dict()}, args.checkpoint)
    print(f"saved {args.checkpoint}")

    train_eval_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    train_dice = float(mean_dice_soft(model, train_eval_loader, device))
    train_iou = float(mean_iou_binary(model, train_eval_loader, device, thresh=0.5))
    merge_train_report(
        "method2_segmentation",
        {
            "epochs": args.epochs,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "device": str(device),
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "train_loss_last_epoch": tr,
            "train_dice_soft": train_dice,
            "train_iou_0.5": train_iou,
            "val_dice_soft": float(vd),
            "val_iou_0.5": float(vi),
            "checkpoint": str(args.checkpoint),
        },
    )


if __name__ == "__main__":
    main()
