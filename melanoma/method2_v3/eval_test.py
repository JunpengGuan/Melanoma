from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import torch

from melanoma.classification_metrics import binary_metrics
from melanoma.config import METHOD2_CONFIG_YAML, SEG_IMG_SIZE
from melanoma.method1.data import load_rows
from melanoma.method2.abcd import extract_abcd
from melanoma.method2.infer import load_rgb_hwc_uint8, predict_mask_bool
from melanoma.method2.unet import build_unet
from melanoma.yaml_config import load_yaml_section, resolve_path


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
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise SystemExit("Need xgboost to evaluate the XGB Method 2 classifier. Run: pip install xgboost") from exc
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

    metrics = binary_metrics(y_true, probs, threshold)
    return {
        "split": "test",
        "classifier": classifier,
        "samples": len(y_true),
        "skipped_missing_jpg": len(all_rows) - len(rows),
        "device": str(device),
        **metrics,
    }


def main() -> None:
    cfg = load_yaml_section(METHOD2_CONFIG_YAML, "eval_test")
    try:
        m = method2_test_metrics(
            unet_checkpoint=resolve_path(cfg["unet_checkpoint"]),
            classifier=str(cfg.get("classifier", "lr")),
            lr_path=resolve_path(cfg["lr_path"]),
            xgb_path=resolve_path(cfg["xgb_path"]),
            test_image_dir=resolve_path(cfg["test_image_dir"]),
            test_label_csv=resolve_path(cfg["test_label_csv"]),
            threshold=float(cfg.get("threshold", 0.5)),
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e

    print(
        f"samples: {m['samples']}  device: {m['device']}  classifier: {m['classifier']}",
    )
    print(f"accuracy @ {m['threshold']}: {m['accuracy']:.4f}")
    print(f"balanced accuracy: {m['balanced_accuracy']:.4f}")
    print(f"sensitivity (melanoma recall): {m['sensitivity']:.4f}")
    print(f"specificity: {m['specificity']:.4f}")
    print(f"f1: {m['f1']:.4f}")
    print(
        f"confusion_matrix [tn fp; fn tp]:\n  TN={m['tn']} FP={m['fp']}\n  FN={m['fn']} TP={m['tp']}",
    )
    auc = m["auc_roc"]
    print(f"auc_roc: {auc:.4f}" if auc == auc else "auc_roc: n/a (only one class in labels)")
    auc_pr = m["auc_pr"]
    print(f"auc_pr: {auc_pr:.4f}" if auc_pr == auc_pr else "auc_pr: n/a (only one class in labels)")


if __name__ == "__main__":
    main()
