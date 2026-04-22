from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from melanoma.config import (
    METHOD1_CONFIG_YAML,
    METHOD2_CONFIG_YAML,
    PROJECT_ROOT,
)
from melanoma.method1.eval_test import method1_test_metrics
from melanoma.method2.eval_test import method2_test_metrics
from melanoma.yaml_config import load_yaml_section, resolve_path


def _sanitize(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def main() -> None:
    m1_cfg = load_yaml_section(METHOD1_CONFIG_YAML, "eval_test")
    m2_cfg = load_yaml_section(METHOD2_CONFIG_YAML, "eval_test")
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    report: dict = {
        "generated_at_utc": ts,
        "method1_config_yaml": str(METHOD1_CONFIG_YAML),
        "method2_config_yaml": str(METHOD2_CONFIG_YAML),
        "method1_threshold": float(m1_cfg.get("threshold", 0.5)),
        "method2_threshold": float(m2_cfg.get("threshold", 0.5)),
        "method1_test_image_dir": str(resolve_path(m1_cfg["image_dir"])),
        "method1_test_label_csv": str(resolve_path(m1_cfg["label_csv"])),
        "method2_test_image_dir": str(resolve_path(m2_cfg["test_image_dir"])),
        "method2_test_label_csv": str(resolve_path(m2_cfg["test_label_csv"])),
    }

    m1 = method1_test_metrics(
        checkpoint=resolve_path(m1_cfg["checkpoint"]),
        image_dir=resolve_path(m1_cfg["image_dir"]),
        label_csv=resolve_path(m1_cfg["label_csv"]),
        backbone=m1_cfg.get("backbone"),
        batch_size=int(m1_cfg.get("batch_size", 32)),
        num_workers=int(m1_cfg.get("num_workers", 2)),
        threshold=float(m1_cfg.get("threshold", 0.5)),
    )
    report["method1"] = m1

    m2_lr = method2_test_metrics(
        unet_checkpoint=resolve_path(m2_cfg["unet_checkpoint"]),
        classifier="lr",
        lr_path=resolve_path(m2_cfg["lr_path"]),
        xgb_path=resolve_path(m2_cfg["xgb_path"]),
        test_image_dir=resolve_path(m2_cfg["test_image_dir"]),
        test_label_csv=resolve_path(m2_cfg["test_label_csv"]),
        threshold=float(m2_cfg.get("threshold", 0.5)),
    )
    report["method2_lr"] = m2_lr

    m2_xgb = method2_test_metrics(
        unet_checkpoint=resolve_path(m2_cfg["unet_checkpoint"]),
        classifier="xgb",
        lr_path=resolve_path(m2_cfg["lr_path"]),
        xgb_path=resolve_path(m2_cfg["xgb_path"]),
        test_image_dir=resolve_path(m2_cfg["test_image_dir"]),
        test_label_csv=resolve_path(m2_cfg["test_label_csv"]),
        threshold=float(m2_cfg.get("threshold", 0.5)),
    )
    report["method2_xgb"] = m2_xgb

    lines = [
        f"Test set report (UTC {ts})  ISIC2017 Test",
        f"method1 threshold={m1_cfg.get('threshold', 0.5)}",
        f"method2 threshold={m2_cfg.get('threshold', 0.5)}",
        f"method1 images: {resolve_path(m1_cfg['image_dir'])}",
        f"method1 labels: {resolve_path(m1_cfg['label_csv'])}",
        f"method2 images: {resolve_path(m2_cfg['test_image_dir'])}",
        f"method2 labels: {resolve_path(m2_cfg['test_label_csv'])}",
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
