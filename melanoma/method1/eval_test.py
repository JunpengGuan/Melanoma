from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from melanoma.config import (
    DEFAULT_CHECKPOINT,
    DEFAULT_TEST_IMAGE_DIR,
    DEFAULT_TEST_LABEL_CSV,
)
from melanoma.method1.data import LesionImageDataset, load_rows
from melanoma.method1.models import build_classifier


def filter_existing_rows(rows: list[tuple[str, int]], image_dir: Path) -> list[tuple[str, int]]:
    image_dir = Path(image_dir)
    out: list[tuple[str, int]] = []
    for image_id, y in rows:
        if (image_dir / f"{image_id}.jpg").is_file():
            out.append((image_id, y))
    return out


@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[float], list[int]]:
    model.eval()
    probs: list[float] = []
    labels: list[int] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.sigmoid(logits).squeeze(-1).cpu()
        probs.extend(p.tolist())
        labels.extend(y.int().tolist())
    return probs, labels


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate Method-1 checkpoint on a held-out set (JPG folder + CSV labels).",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"default: {DEFAULT_CHECKPOINT}",
    )
    p.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_TEST_IMAGE_DIR,
        help=f"default: {DEFAULT_TEST_IMAGE_DIR}",
    )
    p.add_argument(
        "--label-csv",
        type=Path,
        default=DEFAULT_TEST_LABEL_CSV,
        help=f"default: {DEFAULT_TEST_LABEL_CSV} (benign/malignant or 0/1)",
    )
    p.add_argument("--backbone", type=str, default=None, help="Override backbone if missing from checkpoint")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    try:
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            roc_auc_score,
        )
    except ImportError as e:
        raise SystemExit(
            "Need scikit-learn for metrics. Run: pip install scikit-learn",
        ) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    backbone = args.backbone or ckpt.get("backbone")
    if not backbone:
        raise SystemExit("Checkpoint has no 'backbone'; pass --backbone efficientnet_b0 (or vit_b_16).")

    rows = load_rows(args.label_csv)
    rows = filter_existing_rows(rows, args.image_dir)
    if not rows:
        raise SystemExit("No labeled images found (check --image-dir and --label-csv, and .jpg names).")

    skipped = len(load_rows(args.label_csv)) - len(rows)
    if skipped:
        print(f"warning: skipped {skipped} rows with missing JPG under {args.image_dir}")

    ds = LesionImageDataset(rows, args.image_dir, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_classifier(str(backbone), pretrained=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)

    probs, y_true = run_eval(model, loader, device)
    y_score = probs
    y_pred = [1 if p >= args.threshold else 0 for p in probs]

    print(f"samples: {len(y_true)}  device: {device}  backbone: {backbone}")
    print(f"accuracy @ {args.threshold}: {accuracy_score(y_true, y_pred):.4f}")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    print(f"sensitivity (malignant recall): {sens:.4f}")
    print(f"specificity: {spec:.4f}")
    print(f"confusion_matrix [tn fp; fn tp]:\n  TN={tn} FP={fp}\n  FN={fn} TP={tp}")

    if len(set(y_true)) < 2:
        print("auc_roc: n/a (only one class in labels)")
    else:
        print(f"auc_roc: {roc_auc_score(y_true, y_score):.4f}")


if __name__ == "__main__":
    main()
