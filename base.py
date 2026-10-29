"""Project entry point for local data discovery and readiness inspection.

This module is intentionally scaffolding only.  It defines the inspection
workflow that will later load every local source, summarize its usefulness,
and hand validated data to the T1-T5 pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class LocalDataSource:
    """Describe one expected local source and its role in the project."""

    name: str
    path: Path
    expected_format: str
    role: str
    required: bool


def define_local_data_sources(project_root: Path = PROJECT_ROOT) -> list[LocalDataSource]:
    """Return the complete inventory of raw and generated local data sources."""
    raw = project_root / "data" / "raw"
    return [
        LocalDataSource("health_facilities", project_root / "health_facilities.geojson", "GeoJSON", "health supply", False),
        LocalDataSource("education_facilities", project_root / "education_facilities.geojson", "GeoJSON", "education cross-check", False),
        LocalDataSource("populated_places", project_root / "populated_places.geojson", "GeoJSON", "settlement validation", False),
        LocalDataSource("roads", project_root / "heigit_LK_planet_roadsurface_lines.geojson", "GeoJSON", "routing fallback", False),
        LocalDataSource("raw_health_facilities", raw / "hdx_health_facilities.geojson", "GeoJSON", "health supply", False),
        LocalDataSource("raw_education_facilities", raw / "hdx_education_facilities.geojson", "GeoJSON", "education cross-check", False),
        LocalDataSource("raw_populated_places", raw / "hdx_populated_places.geojson", "GeoJSON", "settlement validation", False),
        LocalDataSource("raw_roads", raw / "hdx_roads.geojson", "GeoJSON", "routing cross-check", False),
        LocalDataSource("gn_divisions", project_root / "data" / "sri_lanka.gpkg", "GeoPackage", "demand units", False),
        LocalDataSource("accessibility_scores", project_root / "data" / "accessibility_scores.csv", "CSV", "analysis output", False),
    ]


def discover_local_files(project_root: Path = PROJECT_ROOT) -> list[Path]:
    """Find local files without downloading, transforming, or mutating them."""
    ignored = {".venv", ".git", "__pycache__"}
    return sorted(
        p for p in project_root.rglob("*")
        if p.is_file() and not any(part in ignored for part in p.parts)
    )


def load_local_source(source: LocalDataSource) -> Any:
    """Load one source using its format-specific reader.

    Future implementation: use GeoPandas for vector data, a GPX reader for
    trails, and the lk-schools loader for the school table.  Preserve source
    metadata and do not silently skip malformed or missing required data.
    """
    if not source.path.exists():
        return None
    if source.expected_format == "GeoJSON":
        import geopandas as gpd
        return gpd.read_file(source.path)
    if source.expected_format == "GeoPackage":
        import geopandas as gpd
        return gpd.read_file(source.path, layer=source.name)
    if source.expected_format == "CSV":
        import pandas as pd
        return pd.read_csv(source.path)
    raise ValueError(f"Unsupported source format: {source.expected_format}")


def inspect_source_data(source: LocalDataSource, data: Any) -> dict[str, Any]:
    """Produce a human-readable readiness summary for one loaded source.

    The summary should include existence, row/feature count, columns,
    geometry types, CRS, bounds, null counts, duplicate counts, and validity
    checks where those concepts apply.
    """
    summary: dict[str, Any] = {
        "name": source.name, "path": str(source.path), "exists": source.path.exists(),
        "required": source.required, "role": source.role,
    }
    if data is None:
        summary["usable"] = False
        return summary
    summary.update({"usable": True, "rows": len(data), "columns": list(data.columns)})
    if hasattr(data, "geometry"):
        summary["crs"] = str(data.crs)
        summary["geometry_types"] = data.geom_type.value_counts(dropna=False).to_dict()
        summary["bounds"] = tuple(float(v) for v in data.total_bounds)
        summary["invalid_geometries"] = int((~data.geometry.is_valid).sum())
        summary["null_geometries"] = int(data.geometry.isna().sum())
    summary["null_counts"] = data.isna().sum().loc[lambda s: s.gt(0)].to_dict()
    summary["duplicate_rows"] = int(data.duplicated().sum())
    return summary


def build_data_inventory(project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    """Load and inspect every local source needed by the pipeline."""
    inventory = []
    for source in define_local_data_sources(project_root):
        data = load_local_source(source)
        inventory.append(inspect_source_data(source, data))
    return inventory


def display_data_inventory(inventory: list[dict[str, Any]]) -> None:
    """Display all source summaries so data usefulness can be reviewed."""
    for item in inventory:
        state = "READY" if item.get("usable") else "MISSING"
        print(f"{state:7} {item['name']:<28} {item['path']}")
        for key in ("rows", "crs", "geometry_types", "bounds", "invalid_geometries", "duplicate_rows"):
            if key in item:
                print(f"         {key}: {item[key]}")


def validate_inventory(inventory: list[dict[str, Any]]) -> None:
    """Fail explicitly when required sources are absent or unusable."""
    missing = [i["name"] for i in inventory if i["required"] and not i.get("usable")]
    if missing:
        raise FileNotFoundError("Required local sources missing or unreadable: " + ", ".join(missing))


def main() -> None:
    """Run the local-data inspection workflow only.

    Intended connection:
    define_local_data_sources -> build_data_inventory ->
    validate_inventory -> display_data_inventory.
    """
    inventory = build_data_inventory()
    validate_inventory(inventory)
    display_data_inventory(inventory)


if __name__ == "__main__":
    main()
