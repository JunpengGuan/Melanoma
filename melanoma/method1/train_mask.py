import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from melanoma.config import PROJECT_ROOT
from melanoma.method1.data_mask import LesionImageMaskDataset
from melanoma.method1.data_mask import load_rows, make_loaders, stratified_split
from melanoma.method1.models import build_classifier
from melanoma.train_report import merge_train_report


@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).view(-1, 1)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)

        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()

        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def collect_probs(model, loader, device):
    model.eval()

    probs = []
    labels = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze(-1).cpu()

        probs += prob.tolist()
        labels += y.int().tolist()

    return probs, labels


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--label-csv", type=Path, required=True)

    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = load_rows(args.label_csv)
    n_pos = sum(1 for _, y in rows if y == 1)
    n_neg = len(rows) - n_pos
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device)

    train_loader, val_loader, rows = make_loaders(
        args.image_dir,
        args.mask_dir,
        args.label_csv,
        args.val_ratio,
        args.batch_size,
        args.num_workers,
        args.seed,
        use_weighted_sampler=not args.no_weighted_sampler,
    )

    model = build_classifier(args.backbone, pretrained=not args.no_pretrained)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).view(-1, 1)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_cuda):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            total += x.size(0)

        train_loss = running_loss / max(1, total)
        val_loss, val_acc = eval_epoch(model, val_loader, device, criterion)

        print(
            f"epoch {epoch:03d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}",
            flush=True,
        )

    ckpt = args.checkpoint_dir / (args.backbone + "_mask_last.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "backbone": args.backbone,
            "with_mask": True,
        },
        ckpt,
    )

    print("saved", ckpt, flush=True)

    train_idx, val_idx = stratified_split(rows, args.val_ratio, args.seed)

    eval_data = LesionImageMaskDataset(
        rows,
        args.image_dir,
        args.mask_dir,
        train=False,
    )

    train_eval_loader = DataLoader(
        Subset(eval_data, train_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    train_loss, train_acc = eval_epoch(model, train_eval_loader, device, criterion)
    val_loss, val_acc = eval_epoch(model, val_loader, device, criterion)

    val_probs, val_labels = collect_probs(model, val_loader, device)

    try:
        from sklearn.metrics import roc_auc_score

        if len(set(val_labels)) >= 2:
            val_auc = float(roc_auc_score(val_labels, val_probs))
        else:
            val_auc = float("nan")
    except ImportError:
        val_auc = None

    report = {
        "backbone": args.backbone,
        "with_mask": True,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_auc_roc": val_auc,
        "checkpoint": str(ckpt),
    }

    merge_train_report("method1_with_mask", report)


if __name__ == "__main__":
    main()