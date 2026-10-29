"""Coordinate-reference-system guards used at stage boundaries."""

from typing import Any

from config import METRIC_CRS, STORAGE_CRS


def require_crs(frame: Any, name: str) -> None:
    """Raise when a geospatial frame has no CRS."""
    if getattr(frame, "crs", None) is None:
        raise ValueError(f"{name} has no CRS.")


def metric_centroids(frame: Any) -> Any:
    """Return centroids calculated after projection to the metric CRS."""
    require_crs(frame, "input geodata")
    return frame.to_crs(METRIC_CRS).geometry.centroid


def normalize_storage_crs(frame: Any) -> Any:
    """Return a frame in the canonical storage CRS."""
    require_crs(frame, "input geodata")
    return frame.to_crs(STORAGE_CRS)
