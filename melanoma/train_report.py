"""Merge training-time metrics into ``results/train_report.{json,txt}``.

Each training script updates its own section without erasing others."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from melanoma.config import PROJECT_ROOT


def _json_safe(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def merge_train_report(section: str, data: dict, *, out_dir: Path | None = None) -> None:
    """Write ``data`` under key ``section``; refresh ``train_report.txt`` from full JSON."""
    out_dir = out_dir or (PROJECT_ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "train_report.json"

    existing: dict = {}
    if json_path.is_file():
        try:
            existing = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    existing[section] = _json_safe(data)
    existing["updated_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    safe = _json_safe(existing)
    json_path.write_text(json.dumps(safe, indent=2), encoding="utf-8")

    lines = [
        f"Train report  (last update UTC {safe['updated_at_utc']})",
        "",
    ]
    for key in sorted(k for k in safe if k != "updated_at_utc"):
        lines.append(f"=== {key} ===")
        lines.append(json.dumps(safe[key], indent=2))
        lines.append("")
    (out_dir / "train_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"updated train report: {json_path}")
