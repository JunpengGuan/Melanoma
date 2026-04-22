from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset

from melanoma.config import METHOD2_CONFIG_YAML
from melanoma.method1.data import load_rows, stratified_split
from melanoma.method2.data_seg import LesionSegDataset, filter_rows_with_masks
from melanoma.method2.losses import SegmentationLoss
from melanoma.method2.seg_metrics import mean_dice_soft, mean_iou_binary
from melanoma.method2.unet import build_unet
from melanoma.train_report import merge_train_report
from melanoma.yaml_config import load_yaml_section, resolve_path


def main() -> None:
    cfg = load_yaml_section(METHOD2_CONFIG_YAML, "train_seg")
    image_dir = resolve_path(cfg["image_dir"])
    label_csv = resolve_path(cfg["label_csv"])
    mask_dir = resolve_path(cfg["mask_dir"])
    val_image_dir = resolve_path(cfg.get("val_image_dir"))
    val_label_csv = resolve_path(cfg.get("val_label_csv"))
    val_mask_dir = resolve_path(cfg.get("val_mask_dir"))
    epochs = int(cfg.get("epochs", 30))
    batch_size = int(cfg.get("batch_size", 8))
    lr = float(cfg.get("lr", 1e-4))
    val_ratio = float(cfg.get("val_ratio", 0.15))
    num_workers = int(cfg.get("num_workers", 2))
    seed = int(cfg.get("seed", 42))
    checkpoint = resolve_path(cfg["checkpoint"])

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_rows = load_rows(label_csv)
    train_rows = filter_rows_with_masks(train_rows, image_dir, mask_dir)
    if len(train_rows) < 10:
        raise SystemExit("Too few samples with image+mask; check paths.")

    if val_image_dir is not None and val_label_csv is not None and val_mask_dir is not None:
        val_rows = load_rows(val_label_csv)
        val_rows = filter_rows_with_masks(val_rows, val_image_dir, val_mask_dir)
        train_idx = list(range(len(train_rows)))
        val_idx = list(range(len(val_rows)))
        train_ds = LesionSegDataset(train_rows, image_dir, mask_dir)
        val_ds = LesionSegDataset(val_rows, val_image_dir, val_mask_dir)
    else:
        train_idx, val_idx = stratified_split(train_rows, val_ratio=val_ratio, seed=seed)
        train_ds = LesionSegDataset(train_rows, image_dir, mask_dir)
        val_ds = train_ds

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = build_unet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = SegmentationLoss(bce_weight=0.5)

    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    tr = vd = vi = 0.0
    for epoch in range(1, epochs + 1):
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

    torch.save({"model": model.state_dict()}, checkpoint)
    print(f"saved {checkpoint}")

    train_eval_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_dice = float(mean_dice_soft(model, train_eval_loader, device))
    train_iou = float(mean_iou_binary(model, train_eval_loader, device, thresh=0.5))
    merge_train_report(
        "method2_segmentation",
        {
            "config_yaml": str(METHOD2_CONFIG_YAML),
            "epochs": epochs,
            "val_ratio": val_ratio,
            "seed": seed,
            "device": str(device),
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "train_image_dir": str(image_dir),
            "val_image_dir": str(val_image_dir or image_dir),
            "train_loss_last_epoch": tr,
            "train_dice_soft": train_dice,
            "train_iou_0.5": train_iou,
            "val_dice_soft": float(vd),
            "val_iou_0.5": float(vi),
            "checkpoint": str(checkpoint),
        },
    )


if __name__ == "__main__":
    main()
