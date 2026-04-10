from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from melanoma.config import DEFAULT_IMAGE_DIR, DEFAULT_LABEL_CSV, PROJECT_ROOT
from melanoma.method1.data import load_rows, make_loaders
from melanoma.method1.models import build_classifier


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

    train_loader, val_loader, _ = make_loaders(
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


if __name__ == "__main__":
    main()
