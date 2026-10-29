"""T1: acquire, normalize, validate, and persist source geodata.

This stage feeds T2 with a common CRS and named GeoPackage layers.
"""

from pathlib import Path
from typing import Any
import re
from dataclasses import asdict

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, Point
from shapely.validation import make_valid

from config import (
    METRIC_CRS,
    STORAGE_CRS,
    STUDY_AREA_BUFFER_M,
    arcgis_layers,
    health_tiers,
    school_tiers,
)
from common.artifacts import write_manifest
from common.validation import validate_geodata


def test_arcgis_access(service_url: str, layer_id: int) -> bool:
    """Confirm public query access with a one-feature request."""
    response = requests.get(
        f"{service_url}/{layer_id}/query",
        params={"where": "1=1", "outFields": "*", "resultRecordCount": 1, "f": "json"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return bool(payload.get("features"))


def download_arcgis_layer(service_url: str, layer_id: int) -> Any:
    """Paginate all ArcGIS features and return a WGS84 GeoDataFrame."""
    url = f"{service_url}/{layer_id}/query"
    features: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = requests.get(url, params={
            "where": "1=1", "outFields": "*", "returnGeometry": "true",
            "outSR": "4326", "f": "geojson", "resultOffset": offset,
            "resultRecordCount": 2_000,
        }, timeout=120)
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(payload["error"])
        page = payload.get("features", [])
        if not page:
            break
        features.extend(page)
        if len(page) < 2_000:
            break
        offset += len(page)
    return gpd.GeoDataFrame.from_features(features, crs=STORAGE_CRS) if features else gpd.GeoDataFrame(geometry=[], crs=STORAGE_CRS)


def load_hdx_layers(raw_dir: Path) -> dict[str, Any]:
    """Load local HDX health, education, populated-place, and road layers."""
    names = {
        "health": ["hdx_health_facilities.geojson", "health_facilities.geojson"],
        "education": ["hdx_education_facilities.geojson", "education_facilities.geojson"],
        "populated_places": ["hdx_populated_places.geojson", "populated_places.geojson"],
        "roads": ["hdx_roads.geojson", "heigit_LK_planet_roadsurface_lines.geojson"],
    }
    result = {}
    root = raw_dir.parent.parent if raw_dir.name == "raw" else raw_dir
    for name, candidates in names.items():
        path = next((raw_dir / p for p in candidates if (raw_dir / p).exists()), None)
        if path is None:
            path = next((root / p for p in candidates if (root / p).exists()), None)
        if path is not None:
            frame = gpd.read_file(path)
            result[name] = frame.to_crs(STORAGE_CRS) if frame.crs else frame.set_crs(STORAGE_CRS)
    return result


def load_and_tier_lk_schools(education_fallback: Any = None) -> Any:
    """Load school tiers and coordinates from lk-schools or HDX geometry."""
    try:
        import lk_schools
    except ImportError as exc:
        raise RuntimeError("Install lk-schools or provide a prepared schools table.") from exc
    schools = pd.DataFrame.from_records(
        asdict(school) for school in lk_schools.get_all_schools()
    )
    if "moe_type" not in schools:
        raise ValueError("lk-schools records do not contain moe_type.")
    schools["tier"] = schools["moe_type"].astype(str).str.strip()
    schools = schools[schools["tier"].isin({t.code for t in school_tiers()})].copy()
    if education_fallback is None or education_fallback.empty:
        raise RuntimeError(
            "lk-schools provides tiers but no coordinates; an education GeoJSON "
            "coordinate source is required."
        )
    result = education_fallback.copy()
    result = result[result.geometry.notna() & ~result.geometry.is_empty].copy()
    result = result.to_crs(STORAGE_CRS) if result.crs else result.set_crs(STORAGE_CRS)
    if result.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).any():
        result = result[result.geometry.geom_type == "Point"].copy()
    if result.empty:
        raise RuntimeError("The education coordinate source contains no point geometries.")
    result["facility_id"] = result.index.astype(str)
    names = {
        str(value).strip().casefold(): tier
        for value, tier in zip(schools["name"], schools["tier"])
        if pd.notna(value)
    }
    name_column = next((column for column in ("name", "name_en", "name_latin") if column in result), None)
    result["tier"] = (
        result[name_column].map(lambda value: names.get(str(value).strip().casefold(), "Type3"))
        if name_column
        else "Type3"
    )
    return result


def deduplicate_health_facilities(health: Any) -> Any:
    """Remove duplicate health records before supply counts are calculated."""
    result = health.copy()
    text_cols = [c for c in ("name", "amenity", "healthcare", "healthcare_speciality", "facility_type") if c in result]
    text = result[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    def tier(value: str) -> str:
        if re.search(r"teaching|medical faculty", value): return "teaching_hospital"
        if re.search(r"base hospital", value): return "base_hospital"
        if re.search(r"divisional|district hospital", value): return "divisional_hospital"
        if re.search(r"hospital|clinic|phc|health centre|health center|dispensary", value): return "phc"
        return "estate_dispensary"
    result["tier"] = text.map(tier)
    names = next((c for c in ("name", "name_en", "id") if c in result), None)
    result["_dedup_name"] = result[names].fillna("").astype(str) if names else ""
    projected = result.to_crs(METRIC_CRS)
    representative = projected.geometry.centroid
    result["_x"] = representative.x.round(1)
    result["_y"] = representative.y.round(1)
    result = result.drop_duplicates(["_dedup_name", "_x", "_y"]).drop(columns=["_dedup_name", "_x", "_y"])
    result["facility_id"] = result.index.astype(str)
    return result


def load_pekoe_trail(path: Path) -> Any:
    """Read GPX tracks, preserve vertex order, and calculate metric lengths."""
    trail = gpd.read_file(path, layer="tracks" if path.suffix.lower() == ".gpx" else None)
    trail = trail.to_crs(STORAGE_CRS) if trail.crs else trail.set_crs(STORAGE_CRS)
    trail = trail[trail.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    trail["track_id"] = trail.index.astype(str)
    trail["length_m"] = trail.to_crs(METRIC_CRS).geometry.length
    return trail


def define_study_area(tea_districts: Any) -> Any:
    """Union tea districts and apply the configured outward metric buffer."""
    tea = tea_districts.to_crs(METRIC_CRS)
    union = tea.geometry.union_all().buffer(STUDY_AREA_BUFFER_M)
    return gpd.GeoDataFrame({"area_id": ["study_area"]}, geometry=[union], crs=METRIC_CRS).to_crs(STORAGE_CRS)


def audit_geodata(layers: dict[str, Any]) -> dict[str, Any]:
    """Check CRS, extent, geometry validity, nulls, and expected row counts."""
    report = {}
    for name, frame in layers.items():
        report[name] = {
            "rows": len(frame), "crs": str(frame.crs),
            "valid_geometries": bool(frame.geometry.is_valid.all()) if hasattr(frame, "geometry") else None,
            "null_geometry_count": int(frame.geometry.isna().sum()) if hasattr(frame, "geometry") else None,
        }
    return report


def repair_geometries(frame: Any) -> Any:
    """Repair invalid geometries while preserving the source CRS and rows."""
    result = frame.copy()
    invalid = result.geometry.notna() & ~result.geometry.is_valid
    if invalid.any():
        result.loc[invalid, "geometry"] = result.loc[invalid, "geometry"].map(make_valid)
    result = result[~result.geometry.isna() & ~result.geometry.is_empty].copy()
    if not result.geometry.is_valid.all():
        raise ValueError("Geometry repair did not produce a valid layer.")
    return result


def _prepare_for_geopackage(frame: Any) -> Any:
    """Return a GeoPackage-safe copy of a GeoDataFrame.

    OGR reserves ``fid`` for the GeoPackage feature identifier. ArcGIS
    attribute tables commonly contain a field with that name, so preserve it
    under a non-reserved name before writing.
    """
    result = frame.copy()
    renamed: dict[str, str] = {}
    used: set[str] = set()
    for column in result.columns:
        if column == result.geometry.name:
            continue
        candidate = str(column)
        if candidate.lower() == "fid":
            candidate = "source_fid"
        candidate = re.sub(r"[^0-9A-Za-z_]", "_", candidate)
        if not candidate or candidate[0].isdigit():
            candidate = f"field_{candidate}"
        base = candidate[:60]
        candidate = base
        suffix = 1
        while candidate.lower() in used or candidate.lower() == "fid":
            suffix_text = f"_{suffix}"
            candidate = f"{base[:63 - len(suffix_text)]}{suffix_text}"
            suffix += 1
        used.add(candidate.lower())
        if candidate != column:
            renamed[column] = candidate
    return result.rename(columns=renamed)


def write_t1_outputs(layers: dict[str, Any], data_dir: Path) -> None:
    """Persist the T1 GeoPackages and their named layers."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "sri_lanka.gpkg",
        "study_area.gpkg",
        "health_facilities.gpkg",
        "schools_by_tier.gpkg",
    ):
        output = data_dir / filename
        if output.exists():
            output.unlink()
    for name, frame in layers.items():
        if not hasattr(frame, "to_file"):
            continue
        if name in {"gn_divisions", "tea_districts", "study_area"}:
            frame = repair_geometries(frame)
        frame = _prepare_for_geopackage(frame)
        if name in {"gn_divisions", "tea_districts"}:
            path = data_dir / "sri_lanka.gpkg"
            frame.to_file(path, layer=name, driver="GPKG", mode="a" if path.exists() else "w")
        elif name == "study_area":
            frame.to_file(data_dir / "study_area.gpkg", layer="study_area", driver="GPKG")
        elif name == "health":
            frame.to_file(data_dir / "health_facilities.gpkg", layer="health", driver="GPKG")
        elif name == "schools":
            for tier in frame["tier"].dropna().unique():
                frame[frame.tier == tier].to_file(data_dir / "schools_by_tier.gpkg", layer=str(tier), driver="GPKG")
        else:
            frame.to_file(data_dir / f"{name}.gpkg", layer=name, driver="GPKG")


def run(raw_dir: Path, data_dir: Path) -> dict[str, Any]:
    """Orchestrate T1 and return validated layers for the next stage."""
    data_dir.mkdir(parents=True, exist_ok=True)
    sources = load_hdx_layers(raw_dir)
    if "health" in sources:
        sources["health"] = deduplicate_health_facilities(sources["health"])
    services = {c.name: c for c in arcgis_layers()}
    for name, config in services.items():
        if test_arcgis_access(config.service_url, config.layer_id):
            sources[name] = repair_geometries(download_arcgis_layer(config.service_url, config.layer_id))
    if "gn_divisions" not in sources or "tea_districts" not in sources:
        raise RuntimeError("Both ArcGIS layers are required to define the study area.")
    try:
        sources["schools"] = load_and_tier_lk_schools(sources.get("education"))
    except RuntimeError:
        sources["schools"] = sources.get("education", gpd.GeoDataFrame(geometry=[], crs=STORAGE_CRS))
        sources["schools"]["tier"] = "Type3"
    trail_path = next(iter((raw_dir).glob("*.gpx")), None)
    if trail_path is not None:
        sources["pekoe_trail"] = load_pekoe_trail(trail_path)
    sources["study_area"] = define_study_area(sources["tea_districts"])
    sources["audit"] = audit_geodata({k: v for k, v in sources.items() if hasattr(v, "geometry")})
    write_t1_outputs(sources, data_dir)
    write_manifest(
        data_dir,
        "t1",
        {
            "layers": sorted(k for k, v in sources.items() if hasattr(v, "geometry")),
            "rows": {k: len(v) for k, v in sources.items() if hasattr(v, "geometry")},
        },
    )
    return sources
