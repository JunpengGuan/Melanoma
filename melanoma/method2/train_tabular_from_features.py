from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from melanoma.method2.train_tabular import (
    _feature_indices,
    _model_report,
    _write_prediction_table,
)
from melanoma.train_report import merge_train_report
from melanoma.yaml_config import resolve_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Method 2 tabular classifiers from cached ABCD feature CSVs.",
    )
    parser.add_argument("--train-feature-table", required=True)
    parser.add_argument("--val-feature-table", required=True)
    parser.add_argument("--feature-set", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--mask-source", required=True, choices=["pred_mask", "gt_mask"])
    parser.add_argument("--lr-out", required=True)
    parser.add_argument("--xgb-out", required=True)
    parser.add_argument("--val-predictions-out", required=True)
    parser.add_argument("--report-section", default=None)
    parser.add_argument("--val-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _read_feature_table(path: Path, feature_names: list[str]) -> tuple[list[str], np.ndarray, list[int]]:
    image_ids: list[str] = []
    labels: list[int] = []
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"Empty feature table: {path}")
        missing = [name for name in feature_names if name not in reader.fieldnames]
        if missing:
            raise SystemExit(f"{path} is missing feature columns: {missing}")
        for row in reader:
            image_ids.append(str(row["image_id"]))
            labels.append(int(float(row["label"])))
            rows.append([float(row[name]) for name in feature_names])
    if not rows:
        raise SystemExit(f"No rows in feature table: {path}")
    return image_ids, np.asarray(rows, dtype=np.float32), labels


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _, feature_names, normalized_feature_set = _feature_indices(args.feature_set)

    train_feature_table = resolve_path(args.train_feature_table)
    val_feature_table = resolve_path(args.val_feature_table)
    lr_out = resolve_path(args.lr_out)
    xgb_out = resolve_path(args.xgb_out)
    val_predictions_out = resolve_path(args.val_predictions_out)

    train_ids, X, y_list = _read_feature_table(train_feature_table, feature_names)
    val_ids, X_val, y_val = _read_feature_table(val_feature_table, feature_names)
    y = np.asarray(y_list, dtype=np.int64)
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
        ],
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
        random_state=args.seed,
        scale_pos_weight=n_neg / n_pos,
    )
    clf_xgb.fit(X, y)
    xgb_out.parent.mkdir(parents=True, exist_ok=True)
    clf_xgb.save_model(str(xgb_out))
    print(f"saved {xgb_out}")

    probs_lr = [float(p) for p in pipe.predict_proba(X_val)[:, 1]]
    probs_xgb = [float(p) for p in clf_xgb.predict_proba(X_val)[:, 1]]
    _write_prediction_table(val_predictions_out, val_ids, y_val, probs_lr, probs_xgb)

    report_section = args.report_section or f"method2_tabular_{args.tag}"
    merge_train_report(
        report_section,
        {
            "source": "cached_feature_table",
            "mask_source": args.mask_source,
            "feature_set": normalized_feature_set,
            "run_tag": args.tag,
            "seed": args.seed,
            "val_threshold": args.val_threshold,
            "train_samples": len(train_ids),
            "val_samples": len(val_ids),
            "feature_names": feature_names,
            "train_feature_table": str(train_feature_table),
            "val_feature_table": str(val_feature_table),
            "val_predictions": str(val_predictions_out),
            "lr": _model_report(y_val, probs_lr, args.val_threshold),
            "xgb": _model_report(y_val, probs_xgb, args.val_threshold),
            "lr_checkpoint": str(lr_out),
            "xgb_checkpoint": str(xgb_out),
        },
    )


if __name__ == "__main__":
    main()
