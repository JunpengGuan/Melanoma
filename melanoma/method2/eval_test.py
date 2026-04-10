from __future__ import annotations

import argparse
from pathlib import Path

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = filter_existing(load_rows(args.test_label_csv), args.test_image_dir)
    if not rows:
        raise SystemExit("No labeled test images found.")

    unet = build_unet().to(device)
    ck = torch.load(args.unet_checkpoint, map_location=device, weights_only=False)
    unet.load_state_dict(ck["model"], strict=True)
    unet.eval()

    if args.classifier == "lr":
        clf = joblib.load(args.lr_path)
    else:
        clf = xgb.XGBClassifier()
        clf.load_model(str(args.xgb_path))

    probs: list[float] = []
    y_true: list[int] = []
    for image_id, y in rows:
        img_path = args.test_image_dir / f"{image_id}.jpg"
        rgb = load_rgb_hwc_uint8(img_path, SEG_IMG_SIZE)
        mbool = predict_mask_bool(unet, img_path, device, SEG_IMG_SIZE)
        feat = extract_abcd(rgb, mbool).reshape(1, -1)
        p_mal = float(clf.predict_proba(feat)[0, 1])
        probs.append(p_mal)
        y_true.append(int(y))

    y_score = probs
    y_pred = [1 if p >= args.threshold else 0 for p in probs]

    print(f"samples: {len(y_true)}  device: {device}  classifier: {args.classifier}")
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
