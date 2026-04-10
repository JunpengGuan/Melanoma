"""Run Method 1 and Method 2 **test set** (Part3B) metrics and save under ``results/``."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from melanoma.config import (
    DEFAULT_CHECKPOINT,
    DEFAULT_METHOD2_LR,
    DEFAULT_METHOD2_XGB,
    DEFAULT_TEST_IMAGE_DIR,
    DEFAULT_TEST_LABEL_CSV,
    DEFAULT_UNET_CKPT,
    PROJECT_ROOT,
)
from melanoma.method1.eval_test import method1_test_metrics
from melanoma.method2.eval_test import method2_test_metrics


def _sanitize(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def main() -> None:
    p = argparse.ArgumentParser(description="Save Part3B test metrics for Method 1 & 2 to results/.")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--checkpoint-m1", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--test-image-dir", type=Path, default=DEFAULT_TEST_IMAGE_DIR)
    p.add_argument("--test-label-csv", type=Path, default=DEFAULT_TEST_LABEL_CSV)
    p.add_argument("--unet-checkpoint", type=Path, default=DEFAULT_UNET_CKPT)
    p.add_argument("--lr-path", type=Path, default=DEFAULT_METHOD2_LR)
    p.add_argument("--xgb-path", type=Path, default=DEFAULT_METHOD2_XGB)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    report: dict = {
        "generated_at_utc": ts,
        "threshold": args.threshold,
        "test_image_dir": str(args.test_image_dir),
        "test_label_csv": str(args.test_label_csv),
    }

    m1 = method1_test_metrics(
        checkpoint=args.checkpoint_m1,
        image_dir=args.test_image_dir,
        label_csv=args.test_label_csv,
        backbone=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
    )
    report["method1"] = m1

    m2_lr = method2_test_metrics(
        unet_checkpoint=args.unet_checkpoint,
        classifier="lr",
        lr_path=args.lr_path,
        xgb_path=args.xgb_path,
        test_image_dir=args.test_image_dir,
        test_label_csv=args.test_label_csv,
        threshold=args.threshold,
    )
    report["method2_lr"] = m2_lr

    m2_xgb = method2_test_metrics(
        unet_checkpoint=args.unet_checkpoint,
        classifier="xgb",
        lr_path=args.lr_path,
        xgb_path=args.xgb_path,
        test_image_dir=args.test_image_dir,
        test_label_csv=args.test_label_csv,
        threshold=args.threshold,
    )
    report["method2_xgb"] = m2_xgb

    lines = [
        f"Test set report (UTC {ts})  Part3B",
        f"threshold={args.threshold}",
        f"images: {args.test_image_dir}",
        f"labels: {args.test_label_csv}",
        "",
        "=== Method 1 ===",
        json.dumps(m1, indent=2),
        "",
        "=== Method 2 — LR ===",
        json.dumps(m2_lr, indent=2),
        "",
        "=== Method 2 — XGB ===",
        json.dumps(m2_xgb, indent=2),
        "",
    ]

    txt_path = out_dir / "test_metrics.txt"
    json_path = out_dir / "test_metrics.json"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize(report), f, indent=2)

    print(f"Wrote {txt_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
