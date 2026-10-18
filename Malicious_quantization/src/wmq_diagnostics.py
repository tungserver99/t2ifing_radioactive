"""Append-only metric warnings; quality failures do not imply execution failures."""
import json
import math
from pathlib import Path
import time


def quality_warning(root, stage, message, *, metrics=None, thresholds=None, action="continue", **context):
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: finite(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite(item) for item in value]
        return value
    entry = finite({"time_unix": time.time(), "level": "warning", "stage": stage,
                    "message": message, "action": action, "metrics": metrics,
                    "thresholds": thresholds, **context})
    path = Path(root) / "quality_warnings.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"WARNING [{stage}] {message}; action={action}; details={path}", flush=True)
    return entry
