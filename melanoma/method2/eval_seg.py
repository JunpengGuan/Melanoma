from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from melanoma.config import METHOD2_CONFIG_YAML
from melanoma.method1.data import load_rows, stratified_split
from melanoma.method2.data_seg import LesionSegDataset, filter_rows_with_masks
from melanoma.method2.seg_metrics import mean_dice_soft, mean_iou_binary
from melanoma.method2.unet import build_unet
from melanoma.yaml_config import load_yaml_section, resolve_path


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


def segmentation_dataset_metrics(
    *,
    checkpoint: Path,
    image_dir: Path,
    label_csv: Path,
    mask_dir: Path,
    batch_size: int = 8,
    num_workers: int = 2,
    iou_thresh: float = 0.5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows(label_csv)
    rows = filter_rows_with_masks(rows, image_dir, mask_dir)
    ds = LesionSegDataset(rows, image_dir, mask_dir)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = build_unet().to(device)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=True)

    dice = mean_dice_soft(model, loader, device)
    iou = mean_iou_binary(model, loader, device, thresh=iou_thresh)
    return {
        "val_samples": len(rows),
        "dice_soft": float(dice),
        "iou": float(iou),
        "iou_thresh": float(iou_thresh),
        "device": str(device),
    }


def main():
    cfg = load_yaml_section(METHOD2_CONFIG_YAML, "eval_seg")
    output_path = resolve_path(cfg.get("output"))
    val_image_dir = resolve_path(cfg.get("val_image_dir"))
    val_label_csv = resolve_path(cfg.get("val_label_csv"))
    val_mask_dir = resolve_path(cfg.get("val_mask_dir"))

    if val_image_dir is not None and val_label_csv is not None and val_mask_dir is not None:
        m = segmentation_dataset_metrics(
            checkpoint=resolve_path(cfg["checkpoint"]),
            image_dir=val_image_dir,
            label_csv=val_label_csv,
            mask_dir=val_mask_dir,
            batch_size=int(cfg.get("batch_size", 8)),
            num_workers=int(cfg.get("num_workers", 2)),
            iou_thresh=float(cfg.get("iou_thresh", 0.5)),
        )
    else:
        m = segmentation_val_metrics(
            checkpoint=resolve_path(cfg["checkpoint"]),
            image_dir=resolve_path(cfg["image_dir"]),
            label_csv=resolve_path(cfg["label_csv"]),
            mask_dir=resolve_path(cfg["mask_dir"]),
            val_ratio=float(cfg.get("val_ratio", 0.15)),
            seed=int(cfg.get("seed", 42)),
            batch_size=int(cfg.get("batch_size", 8)),
            num_workers=int(cfg.get("num_workers", 2)),
            iou_thresh=float(cfg.get("iou_thresh", 0.5)),
        )
    line = (
        f"val_samples: {m['val_samples']}  dice_soft: {m['dice_soft']:.4f}  "
        f"iou@{m['iou_thresh']}: {m['iou']:.4f}"
    )
    print(line)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
