from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from melanoma.config import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_LABEL_CSV,
    DEFAULT_SEG_MASK_DIR,
    DEFAULT_METHOD2_LR,
    DEFAULT_METHOD2_XGB,
    DEFAULT_UNET_CKPT,
    SEG_IMG_SIZE,
)
from melanoma.method1.data import load_rows, stratified_split
from melanoma.method2.abcd import extract_abcd
from melanoma.method2.data_seg import filter_rows_with_masks, load_binary_mask, mask_path_for_id
from melanoma.method2.infer import load_rgb_hwc_uint8, predict_mask_bool
from melanoma.method2.unet import build_unet


def gt_mask_bool(mask_dir: Path, image_id: str, size: int) -> np.ndarray:
    path = mask_path_for_id(mask_dir, image_id)
    t = load_binary_mask(path, (size, size))
    return t.squeeze(0).numpy().astype(bool)


def main() -> None:
    p = argparse.ArgumentParser(description="Method 2 stage 3: train LR / XGBoost on ABCD features.")
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    p.add_argument("--label-csv", type=Path, default=DEFAULT_LABEL_CSV)
    p.add_argument("--mask-dir", type=Path, default=DEFAULT_SEG_MASK_DIR)
    p.add_argument("--unet-checkpoint", type=Path, default=DEFAULT_UNET_CKPT)
    p.add_argument("--use-gt-mask", action="store_true", help="Use GT masks (upper bound); ignore U-Net.")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr-out", type=Path, default=DEFAULT_METHOD2_LR)
    p.add_argument("--xgb-out", type=Path, default=DEFAULT_METHOD2_XGB)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows(args.label_csv)
    rows = filter_rows_with_masks(rows, args.image_dir, args.mask_dir)
    train_idx, _ = stratified_split(rows, val_ratio=args.val_ratio, seed=args.seed)

    unet = None
    if not args.use_gt_mask:
        if not args.unet_checkpoint.is_file():
            raise SystemExit(f"Missing U-Net checkpoint {args.unet_checkpoint}; train with train_seg.py first.")
        unet = build_unet().to(device)
        ck = torch.load(args.unet_checkpoint, map_location=device, weights_only=False)
        unet.load_state_dict(ck["model"], strict=True)
        unet.eval()

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for i in train_idx:
        image_id, y = rows[i]
        img_path = args.image_dir / f"{image_id}.jpg"
        rgb = load_rgb_hwc_uint8(img_path, SEG_IMG_SIZE)
        if args.use_gt_mask:
            mbool = gt_mask_bool(args.mask_dir, image_id, SEG_IMG_SIZE)
        else:
            mbool = predict_mask_bool(unet, img_path, device, SEG_IMG_SIZE)
        feat = extract_abcd(rgb, mbool)
        X_list.append(feat)
        y_list.append(y)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    n_pos = max(1, int(y.sum()))
    n_neg = max(1, int(len(y) - y.sum()))

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    args.lr_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.lr_out)
    print(f"saved {args.lr_out}")

    clf_xgb = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=args.seed,
        scale_pos_weight=n_neg / n_pos,
    )
    clf_xgb.fit(X, y)
    clf_xgb.save_model(str(args.xgb_out))
    print(f"saved {args.xgb_out}")


if __name__ == "__main__":
    main()
