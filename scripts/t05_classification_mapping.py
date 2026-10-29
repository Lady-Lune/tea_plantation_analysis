"""T5: classify, compare, map, and report accessibility results."""

from pathlib import Path
from typing import Any
import json
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from scipy.stats import spearmanr

from common.artifacts import require_artifacts, write_manifest


def classify_tiers(scores: Any) -> Any:
    """Assign adequate, partial, and severe-desert classes per tier."""
    result = scores.copy()
    sai_columns = [c for c in result.columns if c.startswith("sai_")]
    classes = {}
    for column in sai_columns:
        q25, q75 = result[column].quantile([0.25, 0.75])
        values = result[column]
        labels = pd.Series(pd.NA, index=result.index, dtype="object")
        valid = values.notna()
        if q25 == q75:
            labels.loc[valid & values.lt(q25)] = "Severe desert"
            labels.loc[valid & values.eq(q25)] = "Partial"
            labels.loc[valid & values.gt(q25)] = "Adequate"
        else:
            labels.loc[valid & values.le(q25)] = "Severe desert"
            labels.loc[valid & values.le(q75) & values.gt(q25)] = "Partial"
            labels.loc[valid & values.gt(q75)] = "Adequate"
        result[f"class_{column[4:]}"] = labels
        classes[column] = {"q25": float(q25), "q75": float(q75)}
    if sai_columns:
        result["composite_sai"] = result[sai_columns].min(axis=1)
        result["worst_tier"] = result[[f"class_{c[4:]}" for c in sai_columns]].apply(
            lambda row: "Severe desert" if (row == "Severe desert").any()
            else "Partial" if (row == "Partial").any() else "Adequate", axis=1)
    result.attrs["classification_thresholds"] = classes
    return result


def join_context_indicators(scores: Any, indicators: Any) -> Any:
    """Join poverty, estate-share, literacy, and unemployment indicators."""
    key = next((c for c in ("gnd_code", "GND_CODE", "ADM4_CODE", "code") if c in scores), None)
    indicator_key = next((c for c in ("gnd_code", "GND_CODE", "ADM4_CODE", "code") if c in indicators), None)
    if key is None or indicator_key is None:
        raise ValueError("A shared GND identifier is required to join context indicators.")
    return scores.merge(indicators, left_on=key, right_on=indicator_key, how="left", suffixes=("", "_indicator"))


def calculate_stratified_correlations(scores: Any, indicators: Any) -> Any:
    """Calculate Spearman relationships overall and by plantation intensity."""
    result = {}
    sai_columns = [c for c in scores if c.startswith("sai_")]
    plantation = next((c for c in indicators if "estate" in c.lower() or "plantation" in c.lower()), None)
    groups = {"all": scores.index}
    if plantation:
        groups["plantation_heavy"] = scores.index[indicators[plantation] >= indicators[plantation].median()]
        groups["other"] = scores.index[indicators[plantation] < indicators[plantation].median()]
    for group, indices in groups.items():
        result[group] = {}
        for column in sai_columns:
            indicator = next((c for c in indicators if c != plantation and pd.api.types.is_numeric_dtype(indicators[c])), None)
            if indicator and len(indices) > 2:
                valid = scores.loc[indices, column].notna() & indicators.loc[indices, indicator].notna()
                rho, p_value = spearmanr(scores.loc[indices, column][valid], indicators.loc[indices, indicator][valid])
                result[group][column] = {"indicator": indicator, "rho": float(rho), "p_value": float(p_value)}
    return result


def compare_hdx_benchmark(scores: Any, benchmark: Any) -> dict[str, Any]:
    """Compare ranks with HDX accessibility indicators and report divergence."""
    left = scores.set_index("gnd_code") if "gnd_code" in scores else scores
    right = benchmark.set_index("gnd_code") if "gnd_code" in benchmark else benchmark
    joined = left.join(right, lsuffix="_sai", rsuffix="_hdx").dropna()
    sai = next((c for c in joined if c.startswith("sai_")), None)
    hdx = next((c for c in joined if "access" in c.lower() and c != sai), None)
    if not sai or not hdx or len(joined) < 3:
        return {"rho": None, "n": len(joined)}
    rho, p_value = spearmanr(joined[sai], joined[hdx])
    return {"rho": float(rho), "p_value": float(p_value), "n": len(joined)}


def compare_tin_sensitivity(primary: Any, tin: Any) -> dict[str, Any]:
    """Measure SAI changes and the percentage of GNDs reclassified."""
    columns = [c for c in primary if c.startswith("class_")]
    if not columns:
        return {"reclassified_percent": None, "n": 0}
    joined = primary[["gnd_code", *columns]].merge(tin[["gnd_code", *columns]], on="gnd_code", suffixes=("_primary", "_tin"))
    changed = pd.Series(False, index=joined.index)
    for column in columns:
        changed |= joined[f"{column}_primary"] != joined[f"{column}_tin"]
    return {"reclassified_percent": float(changed.mean() * 100), "n": len(joined)}


def create_static_maps(layers: dict[str, Any], outputs_dir: Path) -> None:
    """Create per-tier and worst-affected PNG maps with documented basemaps."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = outputs_dir / "maps"
    maps_dir.mkdir(exist_ok=True)
    gdf = layers["scores"]
    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    for column in [c for c in gdf if c.startswith("sai_")]:
        fig, axis = plt.subplots(figsize=(10, 8))
        gdf.plot(column=column, ax=axis, legend=True, missing_kwds={"color": "lightgrey"})
        axis.set_title(column)
        axis.set_axis_off()
        fig.savefig(maps_dir / f"{column}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def create_interactive_map(layers: dict[str, Any], output_path: Path) -> None:
    """Create the interactive HTML map with toggleable analysis layers."""
    scores = layers["scores"]
    center = scores.geometry.union_all().centroid if hasattr(scores, "geometry") else None
    location = [center.y, center.x] if center else [7.8, 80.7]
    fmap = folium.Map(location=location, zoom_start=9)
    folium.GeoJson(scores.to_json(), name="GND accessibility").add_to(fmap)
    folium.LayerControl().add_to(fmap)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(output_path)


def write_validation_report(results: dict[str, Any], output_path: Path) -> None:
    """Write benchmark, plantation-gap, and TIN-sensitivity findings."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Validation report", ""]
    for key, value in results.items():
        lines.append(f"## {key}")
        lines.append("```json")
        lines.append(json.dumps(value, indent=2, default=str))
        lines.append("```")
    output_path.write_text("\n".join(lines) + "\n")


def run(data_dir: Path, outputs_dir: Path) -> dict[str, Any]:
    """Orchestrate classification, validation, maps, and final deliverables."""
    score_path = data_dir / "accessibility_scores.csv"
    sri_lanka_path = data_dir / "sri_lanka.gpkg"
    require_artifacts([score_path, sri_lanka_path], "T5")
    gnd = gpd.read_file(sri_lanka_path, layer="gn_divisions")
    scores = pd.read_csv(score_path)
    key = next((c for c in ("gnd_code", "GND_CODE", "ADM4_CODE", "code") if c in gnd), None)
    score_key = next((c for c in ("gnd_code", "GND_CODE", "ADM4_CODE", "code") if c in scores), None)
    if key is None:
        key = "gnd_code"
        gnd[key] = gnd.index.astype(str)
    if score_key is None:
        score_key = key
        scores[score_key] = scores.index.astype(str)
    if score_key != key:
        scores[key] = scores[score_key].astype(str)
    gnd[key] = gnd[key].astype(str)
    scores[key] = scores[key].astype(str)
    scores = scores.drop(columns=["geometry"], errors="ignore")
    classified = classify_tiers(scores)
    result = gpd.GeoDataFrame(gnd.merge(classified, on=key, how="left"), geometry=gnd.geometry.name, crs=gnd.crs)
    create_static_maps({"scores": result}, outputs_dir)
    create_interactive_map({"scores": result}, outputs_dir / "interactive_map.html")
    worst = result.sort_values("composite_sai").head(50) if "composite_sai" in result else result.head(50)
    worst.to_csv(outputs_dir / "worst_affected_communities.csv", index=False)
    write_validation_report({"classification_thresholds": classified.attrs.get("classification_thresholds", {})}, outputs_dir / "validation_report.md")
    write_manifest(data_dir, "t5", {"rows": len(result), "outputs_dir": str(outputs_dir)})
    return result
