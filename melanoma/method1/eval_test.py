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


def method1_test_metrics(
    *,
    checkpoint: Path,
    image_dir: Path,
    label_csv: Path,
    backbone: str | None,
    batch_size: int,
    num_workers: int,
    threshold: float,
) -> dict:
    """Evaluate Method 1 on a folder + CSV (default: Part3B test). Returns metric dict."""
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    resolved = backbone or ckpt.get("backbone")
    if not resolved:
        raise ValueError("Checkpoint has no 'backbone'; pass backbone=...")

    all_rows = load_rows(label_csv)
    rows = filter_existing_rows(all_rows, image_dir)
    if not rows:
        raise ValueError("No labeled images found (check paths and .jpg names).")
    skipped = len(all_rows) - len(rows)

    ds = LesionImageDataset(rows, image_dir, train=False)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    model = build_classifier(str(resolved), pretrained=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)

    probs, y_true = run_eval(model, loader, device)
    y_pred = [1 if p >= threshold else 0 for p in probs]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    auc = (
        float(roc_auc_score(y_true, probs))
        if len(set(y_true)) >= 2
        else float("nan")
    )
    return {
        "split": "test",
        "samples": len(y_true),
        "skipped_missing_jpg": skipped,
        "device": str(device),
        "backbone": str(resolved),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": sens,
        "specificity": spec,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "auc_roc": auc,
    }


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

    try:
        m = method1_test_metrics(
            checkpoint=args.checkpoint,
            image_dir=args.image_dir,
            label_csv=args.label_csv,
            backbone=args.backbone,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            threshold=args.threshold,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e

    if m["skipped_missing_jpg"]:
        print(f"warning: skipped {m['skipped_missing_jpg']} rows with missing JPG under {args.image_dir}")

    print(f"samples: {m['samples']}  device: {m['device']}  backbone: {m['backbone']}")
    print(f"accuracy @ {m['threshold']}: {m['accuracy']:.4f}")
    print(f"sensitivity (malignant recall): {m['sensitivity']:.4f}")
    print(f"specificity: {m['specificity']:.4f}")
    print(
        f"confusion_matrix [tn fp; fn tp]:\n  TN={m['tn']} FP={m['fp']}\n  FN={m['fn']} TP={m['tp']}",
    )
    auc = m["auc_roc"]
    print(f"auc_roc: {auc:.4f}" if auc == auc else "auc_roc: n/a (only one class in labels)")


if __name__ == "__main__":
    main()
