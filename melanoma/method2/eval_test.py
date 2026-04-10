from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import joblib
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from melanoma.config import (
    DEFAULT_METHOD2_LR,
    DEFAULT_METHOD2_XGB,
    DEFAULT_TEST_IMAGE_DIR,
    DEFAULT_TEST_LABEL_CSV,
    DEFAULT_UNET_CKPT,
    SEG_IMG_SIZE,
)
from melanoma.method1.data import load_rows
from melanoma.method2.abcd import extract_abcd
from melanoma.method2.infer import load_rgb_hwc_uint8, predict_mask_bool
from melanoma.method2.unet import build_unet


def filter_existing(rows: list[tuple[str, int]], image_dir: Path) -> list[tuple[str, int]]:
    image_dir = Path(image_dir)
    return [(i, y) for i, y in rows if (image_dir / f"{i}.jpg").is_file()]


def method2_test_metrics(
    *,
    unet_checkpoint: Path,
    classifier: Literal["lr", "xgb"],
    lr_path: Path,
    xgb_path: Path,
    test_image_dir: Path,
    test_label_csv: Path,
    threshold: float,
) -> dict:
    """Method 2 on test folder + CSV: U-Net masks + LR or XGB."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = load_rows(test_label_csv)
    rows = filter_existing(all_rows, test_image_dir)
    if not rows:
        raise ValueError("No labeled test images found.")

    unet = build_unet().to(device)
    ck = torch.load(unet_checkpoint, map_location=device, weights_only=False)
    unet.load_state_dict(ck["model"], strict=True)
    unet.eval()

    if classifier == "lr":
        clf = joblib.load(lr_path)
    else:
        clf = xgb.XGBClassifier()
        clf.load_model(str(xgb_path))

    probs: list[float] = []
    y_true: list[int] = []
    for image_id, y in rows:
        img_path = test_image_dir / f"{image_id}.jpg"
        rgb = load_rgb_hwc_uint8(img_path, SEG_IMG_SIZE)
        mbool = predict_mask_bool(unet, img_path, device, SEG_IMG_SIZE)
        feat = extract_abcd(rgb, mbool).reshape(1, -1)
        probs.append(float(clf.predict_proba(feat)[0, 1]))
        y_true.append(int(y))

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
        "classifier": classifier,
        "samples": len(y_true),
        "skipped_missing_jpg": len(all_rows) - len(rows),
        "device": str(device),
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
    p = argparse.ArgumentParser(description="Method 2: evaluate U-Net mask + ABCD + tabular classifier on test CSV.")
    p.add_argument("--unet-checkpoint", type=Path, default=DEFAULT_UNET_CKPT)
    p.add_argument("--classifier", choices=("lr", "xgb"), default="lr")
    p.add_argument("--lr-path", type=Path, default=DEFAULT_METHOD2_LR)
    p.add_argument("--xgb-path", type=Path, default=DEFAULT_METHOD2_XGB)
    p.add_argument("--test-image-dir", type=Path, default=DEFAULT_TEST_IMAGE_DIR)
    p.add_argument("--test-label-csv", type=Path, default=DEFAULT_TEST_LABEL_CSV)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    try:
        m = method2_test_metrics(
            unet_checkpoint=args.unet_checkpoint,
            classifier=args.classifier,
            lr_path=args.lr_path,
            xgb_path=args.xgb_path,
            test_image_dir=args.test_image_dir,
            test_label_csv=args.test_label_csv,
            threshold=args.threshold,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e

    print(
        f"samples: {m['samples']}  device: {m['device']}  classifier: {m['classifier']}",
    )
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
