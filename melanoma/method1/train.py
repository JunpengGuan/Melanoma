from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from melanoma.config import DEFAULT_IMAGE_DIR, DEFAULT_LABEL_CSV, PROJECT_ROOT
from melanoma.method1.data import LesionImageDataset, load_rows, make_loaders, stratified_split
from melanoma.method1.models import build_classifier
from melanoma.train_report import merge_train_report


@torch.no_grad()
def eval_epoch(model: nn.Module, loader, device: torch.device, criterion) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).view(-1, 1)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()
        correct += (pred.eq(y).all(dim=1)).sum().item()
        n += x.size(0)
    return total_loss / max(1, n), correct / max(1, n)


@torch.no_grad()
def collect_probs(model: nn.Module, loader, device: torch.device) -> tuple[list[float], list[int]]:
    model.eval()
    probs: list[float] = []
    labels: list[int] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).view(-1, 1)
        logits = model(x)
        p = torch.sigmoid(logits).squeeze(-1).cpu()
        probs.extend(p.tolist())
        labels.extend(y.int().view(-1).tolist())
    return probs, labels


def main() -> None:
    p = argparse.ArgumentParser(description="Method 1: train end-to-end lesion classifier.")
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    p.add_argument("--label-csv", type=Path, default=DEFAULT_LABEL_CSV)
    p.add_argument("--backbone", type=str, default="efficientnet_b0")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-weighted-sampler", action="store_true")
    p.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = load_rows(args.label_csv)
    n_pos = sum(1 for _, y in rows if y == 1)
    n_neg = len(rows) - n_pos
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device)

    train_loader, val_loader, rows = make_loaders(
        args.image_dir,
        args.label_csv,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_weighted_sampler=not args.no_weighted_sampler,
    )

    model = build_classifier(args.backbone, pretrained=not args.no_pretrained)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1, 1)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        tr_loss = running / max(1, seen)
        va_loss, va_acc = eval_epoch(model, val_loader, device, criterion)
        print(f"epoch {epoch:03d}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_acc={va_acc:.4f}")

    ckpt = args.checkpoint_dir / f"{args.backbone}_last.pt"
    torch.save({"model": model.state_dict(), "backbone": args.backbone}, ckpt)
    print(f"saved {ckpt}")

    train_idx, val_idx = stratified_split(rows, val_ratio=args.val_ratio, seed=args.seed)
    full_eval = LesionImageDataset(rows, args.image_dir, train=False)
    train_eval_loader = DataLoader(
        Subset(full_eval, train_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    tr_loss, tr_acc = eval_epoch(model, train_eval_loader, device, criterion)
    va_loss, va_acc = eval_epoch(model, val_loader, device, criterion)
    v_probs, v_labels = collect_probs(model, val_loader, device)
    val_auc: float | None
    try:
        from sklearn.metrics import roc_auc_score

        val_auc = (
            float(roc_auc_score(v_labels, v_probs))
            if len(set(v_labels)) >= 2
            else float("nan")
        )
    except ImportError:
        val_auc = None

    report = {
        "backbone": args.backbone,
        "epochs": args.epochs,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "device": str(device),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "train_loss": tr_loss,
        "train_acc": tr_acc,
        "val_loss": va_loss,
        "val_acc": va_acc,
        "val_auc_roc": val_auc,
        "checkpoint": str(ckpt),
    }
    merge_train_report("method1", report)


if __name__ == "__main__":
    main()
