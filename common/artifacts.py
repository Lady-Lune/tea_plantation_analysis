"""Artifact paths and stage completion manifests."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def manifest_dir(data_dir: Path) -> Path:
    """Return the directory containing stage manifests."""
    path = data_dir / "manifests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_manifest(data_dir: Path, stage: str, details: dict[str, Any]) -> Path:
    """Write an atomic completion manifest for a successful stage."""
    target = manifest_dir(data_dir) / f"{stage}.json"
    payload = {
        "stage": stage,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary.replace(target)
    return target


def require_artifacts(paths: list[Path], stage: str) -> None:
    """Fail before a stage runs when an upstream artifact is absent."""
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{stage} requires missing artifacts: {', '.join(missing)}")
