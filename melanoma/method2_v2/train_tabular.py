from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from melanoma.classification_metrics import binary_metrics, threshold_sweep_summary
from melanoma.config import (
    METHOD2_CONFIG_YAML,
    SEG_IMG_SIZE,
)
from melanoma.method1.data import load_rows, stratified_split
from melanoma.method2_v2.abcd import FEATURE_NAMES, extract_abcd
from melanoma.method2_v2.data_seg import filter_rows_with_masks, load_binary_mask, mask_path_for_id
from melanoma.method2_v2.infer import load_rgb_hwc_uint8, predict_mask_bool
from melanoma.method2_v2.unet import build_unet
from melanoma.train_report import merge_train_report
from melanoma.yaml_config import load_yaml_section, resolve_path


def _filter_rows_with_images(rows: list[tuple[str, int]], image_dir: Path) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for image_id, y in rows:
        if (image_dir / f"{image_id}.jpg").is_file():
            out.append((image_id, y))
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Method 2 ABCD tabular classifiers.")
    parser.add_argument(
        "--mask-source",
        choices=["pred", "gt"],
        default=None,
        help="Override train_tabular.use_gt_mask: pred=U-Net masks, gt=ground-truth masks.",
    )
    parser.add_argument(
        "--feature-set",
        default=None,
        help="ABCD feature subset, e.g. A, B, C, D, AB, ABC, ABCD. Default: ABCD.",
    )
    parser.add_argument("--tag", default=None, help="Run tag used in checkpoint/result filenames.")
    parser.add_argument("--report-section", default=None, help="Section name in results/train_report.*")
    parser.add_argument("--lr-out", default=None, help="Output path for Logistic Regression checkpoint.")
    parser.add_argument("--xgb-out", default=None, help="Output path for XGBoost checkpoint.")
    parser.add_argument("--feature-table-out", default=None, help="Output CSV path for train features.")
    parser.add_argument("--val-feature-table-out", default=None, help="Output CSV path for val features.")
    parser.add_argument("--val-predictions-out", default=None, help="Output CSV path for val probabilities.")
    parser.add_argument("--val-threshold", type=float, default=None, help="Fixed threshold for reported metrics.")
    parser.add_argument("--image-dir", default=None, help="Override shared.image_dir.")
    parser.add_argument("--label-csv", default=None, help="Override shared.label_csv.")
    parser.add_argument("--mask-dir", default=None, help="Override shared.mask_dir.")
    parser.add_argument("--val-image-dir", default=None, help="Override shared.val_image_dir.")
    parser.add_argument("--val-label-csv", default=None, help="Override shared.val_label_csv.")
    parser.add_argument("--val-mask-dir", default=None, help="Override shared.val_mask_dir.")
    parser.add_argument("--unet-checkpoint", default=None, help="Override shared.unet_checkpoint.")
    parser.add_argument("--val-ratio", type=float, default=None, help="Override shared.val_ratio.")
    parser.add_argument("--seed", type=int, default=None, help="Override shared.seed.")
    return parser.parse_args(argv)


def _feature_indices(feature_set: str) -> tuple[list[int], list[str], str]:
    spec = str(feature_set or "ABCD").upper().replace("+", "")
    if spec == "ALL":
        spec = "ABCD"
    allowed = set("ABCD")
    requested = set(spec)
    if not spec or requested - allowed:
        raise SystemExit(
            f"Unsupported feature set '{feature_set}'. Use combinations of A/B/C/D, e.g. A, BC, ABCD.",
        )
    ordered = "".join(ch for ch in "ABCD" if ch in requested)
    idx = [i for i, name in enumerate(FEATURE_NAMES) if len(name) > 1 and name[1] == "_" and name[0] in requested]
    names = [FEATURE_NAMES[i] for i in idx]
    if not idx:
        raise SystemExit(f"Feature set '{feature_set}' selected no features.")
    return idx, names, ordered


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = dict(cfg)
    if args.mask_source is not None:
        cfg["use_gt_mask"] = args.mask_source == "gt"
    for arg_name, cfg_name in [
        ("feature_set", "feature_set"),
        ("tag", "tag"),
        ("report_section", "report_section"),
        ("lr_out", "lr_out"),
        ("xgb_out", "xgb_out"),
        ("feature_table_out", "feature_table_out"),
        ("val_feature_table_out", "val_feature_table_out"),
        ("val_predictions_out", "val_predictions_out"),
        ("image_dir", "image_dir"),
        ("label_csv", "label_csv"),
        ("mask_dir", "mask_dir"),
        ("val_image_dir", "val_image_dir"),
        ("val_label_csv", "val_label_csv"),
        ("val_mask_dir", "val_mask_dir"),
        ("unet_checkpoint", "unet_checkpoint"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            cfg[cfg_name] = value
    if args.val_threshold is not None:
        cfg["val_threshold"] = args.val_threshold
    if args.val_ratio is not None:
        cfg["val_ratio"] = args.val_ratio
    if args.seed is not None:
        cfg["seed"] = args.seed
    return cfg


def gt_mask_bool(mask_dir: Path, image_id: str, size: int) -> np.ndarray:
    path = mask_path_for_id(mask_dir, image_id)
    t = load_binary_mask(path, (size, size))
    return t.squeeze(0).numpy().astype(bool)


def _model_report(y_true: list[int], probs: list[float], threshold: float) -> dict:
    fixed = binary_metrics(y_true, probs, threshold)
    fixed["threshold_sweep"] = threshold_sweep_summary(y_true, probs)
    return fixed


def _write_feature_table(
    path,
    image_ids,
    X,
    y,
    feature_names,
):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", *feature_names, "label"])
        for image_id, feats, label in zip(image_ids, X, y, strict=True):
            writer.writerow([image_id, *[float(v) for v in feats], int(label)])
    print(f"saved {path}")


def _write_prediction_table(
    path,
    image_ids,
    y,
    probs_lr,
    probs_xgb,
):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "label", "lr_prob", "xgb_prob"])
        for row in zip(image_ids, y, probs_lr, probs_xgb, strict=True):
            image_id, label, lr_prob, xgb_prob = row
            writer.writerow([image_id, int(label), float(lr_prob), float(xgb_prob)])
    print(f"saved {path}")


def main(argv: list[str] | None = None):
    args = _parse_args(argv)
    cfg = load_yaml_section(METHOD2_CONFIG_YAML, "train_tabular")
    cfg = _apply_cli_overrides(cfg, args)
    image_dir = resolve_path(cfg["image_dir"])
    label_csv = resolve_path(cfg["label_csv"])
    mask_dir = resolve_path(cfg["mask_dir"])
    val_image_dir = resolve_path(cfg.get("val_image_dir"))
    val_label_csv = resolve_path(cfg.get("val_label_csv"))
    val_mask_dir = resolve_path(cfg.get("val_mask_dir"))
    unet_checkpoint = resolve_path(cfg["unet_checkpoint"])
    use_gt_mask = bool(cfg.get("use_gt_mask", False))
    val_ratio = float(cfg.get("val_ratio", 0.15))
    seed = int(cfg.get("seed", 42))
    val_threshold = float(cfg.get("val_threshold", 0.5))
    mask_tag = "gt_mask" if use_gt_mask else "pred_mask"
    feature_idx, feature_names, feature_set = _feature_indices(str(cfg.get("feature_set") or "ABCD"))
    run_tag = str(cfg.get("tag") or (mask_tag if feature_set == "ABCD" else f"{mask_tag}_{feature_set.lower()}"))
    lr_out = resolve_path(cfg.get("lr_out") or f"checkpoints/method2_{run_tag}_lr.joblib")
    xgb_out = resolve_path(cfg.get("xgb_out") or f"checkpoints/method2_{run_tag}_xgb.json")
    feature_table_out = resolve_path(
        cfg.get("feature_table_out") or f"results/method2_{run_tag}_train_features.csv",
    )
    val_feature_table_out = resolve_path(
        cfg.get("val_feature_table_out") or f"results/method2_{run_tag}_val_features.csv",
    )
    val_predictions_out = resolve_path(
        cfg.get("val_predictions_out") or f"results/method2_{run_tag}_val_predictions.csv",
    )
    report_section = cfg.get("report_section")
    if not report_section:
        report_section = f"method2_tabular_{run_tag}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = load_rows(label_csv)
    if use_gt_mask:
        train_rows = filter_rows_with_masks(train_rows, image_dir, mask_dir)
    else:
        train_rows = _filter_rows_with_images(train_rows, image_dir)
    has_external_val = (
        val_image_dir is not None
        and val_label_csv is not None
        and (not use_gt_mask or val_mask_dir is not None)
    )
    if has_external_val:
        val_rows = load_rows(val_label_csv)
        if use_gt_mask:
            val_rows = filter_rows_with_masks(val_rows, val_image_dir, val_mask_dir)
        else:
            val_rows = _filter_rows_with_images(val_rows, val_image_dir)
        train_idx = list(range(len(train_rows)))
        val_idx = list(range(len(val_rows)))
    else:
        train_idx, val_idx = stratified_split(train_rows, val_ratio=val_ratio, seed=seed)
        val_rows = train_rows
        val_image_dir = image_dir
        val_mask_dir = mask_dir

    unet = None
    if not use_gt_mask:
        if not unet_checkpoint.is_file():
            raise SystemExit(f"Missing U-Net checkpoint {unet_checkpoint}; train with train_seg.py first.")
        unet = build_unet().to(device)
        ck = torch.load(unet_checkpoint, map_location=device, weights_only=False)
        unet.load_state_dict(ck["model"], strict=True)
        unet.eval()

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    train_image_ids: list[str] = []
    for i in train_idx:
        image_id, y = train_rows[i]
        img_path = image_dir / f"{image_id}.jpg"
        rgb = load_rgb_hwc_uint8(img_path, SEG_IMG_SIZE)
        if use_gt_mask:
            mbool = gt_mask_bool(mask_dir, image_id, SEG_IMG_SIZE)
        else:
            mbool = predict_mask_bool(unet, img_path, device, SEG_IMG_SIZE)
        feat = extract_abcd(rgb, mbool)
        X_list.append(feat)
        y_list.append(y)
        train_image_ids.append(image_id)

    X_all = np.stack(X_list, axis=0)
    X = X_all[:, feature_idx]
    y = np.array(y_list, dtype=np.int64)
    _write_feature_table(feature_table_out, train_image_ids, X, y_list, feature_names)
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
    lr_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, lr_out)
    print(f"saved {lr_out}")

    try:
        import xgboost as xgb
    except ImportError as exc:
        raise SystemExit("Need xgboost for Method 2 tabular training. Run: pip install xgboost") from exc

    clf_xgb = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=seed,
        scale_pos_weight=n_neg / n_pos,
    )
    clf_xgb.fit(X, y)
    clf_xgb.save_model(str(xgb_out))
    print(f"saved {xgb_out}")

    probs_lr: list[float] = []
    probs_xgb: list[float] = []
    y_val: list[int] = []
    val_image_ids: list[str] = []
    X_val_list: list[np.ndarray] = []
    for i in val_idx:
        image_id, y = val_rows[i]
        img_path = val_image_dir / f"{image_id}.jpg"
        rgb = load_rgb_hwc_uint8(img_path, SEG_IMG_SIZE)
        if use_gt_mask:
            mbool = gt_mask_bool(val_mask_dir, image_id, SEG_IMG_SIZE)
        else:
            mbool = predict_mask_bool(unet, img_path, device, SEG_IMG_SIZE)
        feat_all = extract_abcd(rgb, mbool)
        feat = feat_all[feature_idx]
        X_val_list.append(feat)
        feat = feat.reshape(1, -1)
        probs_lr.append(float(pipe.predict_proba(feat)[0, 1]))
        probs_xgb.append(float(clf_xgb.predict_proba(feat)[0, 1]))
        y_val.append(int(y))
        val_image_ids.append(image_id)

    X_val = np.stack(X_val_list, axis=0)
    _write_feature_table(val_feature_table_out, val_image_ids, X_val, y_val, feature_names)
    _write_prediction_table(val_predictions_out, val_image_ids, y_val, probs_lr, probs_xgb)

    thr = val_threshold
    merge_train_report(
        str(report_section),
        {
            "config_yaml": str(METHOD2_CONFIG_YAML),
            "val_ratio": val_ratio,
            "seed": seed,
            "use_gt_mask": use_gt_mask,
            "mask_source": "ground_truth" if use_gt_mask else "unet_prediction",
            "feature_set": feature_set,
            "run_tag": run_tag,
            "device": str(device),
            "train_samples": len(train_idx),
            "val_samples": len(y_val),
            "train_image_dir": str(image_dir),
            "val_image_dir": str(val_image_dir),
            "feature_names": feature_names,
            "lr": _model_report(y_val, probs_lr, thr),
            "xgb": _model_report(y_val, probs_xgb, thr),
            "lr_checkpoint": str(lr_out),
            "xgb_checkpoint": str(xgb_out),
            "feature_table": str(feature_table_out) if feature_table_out else None,
            "val_feature_table": str(val_feature_table_out) if val_feature_table_out else None,
            "val_predictions": str(val_predictions_out) if val_predictions_out else None,
        },
    )


if __name__ == "__main__":
    main()
