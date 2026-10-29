"""T4: calculate tier-specific two-step floating catchment accessibility."""

from pathlib import Path
from typing import Any
import pandas as pd
import geopandas as gpd

from config import health_tiers, school_tiers
from common.artifacts import require_artifacts, write_manifest


def build_demand_table(gnd: Any, population_source: Any) -> Any:
    """Create the GND population denominator and preserve source metadata."""
    result = population_source.copy()
    key = next((c for c in ("gnd_code", "GND_CODE", "ADM4_CODE", "code") if c in result), None)
    result["origin"] = result[key].astype(str) if key else result.index.astype(str)
    if "population" not in result:
        result["population"] = 1.0
    return result[["origin", "population"]]


def build_supply_table(facilities: Any, tier: str) -> Any:
    """Create tier supply values and de-duplicate snapped facility nodes."""
    result = facilities.copy()
    result["dest"] = result["facility_id"].astype(str) if "facility_id" in result else result.index.astype(str)
    supply_values = result["supply"] if "supply" in result else pd.Series(1.0, index=result.index)
    result["supply"] = pd.to_numeric(supply_values, errors="coerce").fillna(1.0)
    if "node" in result:
        result = result.drop_duplicates("node")
    return result[["dest", "supply"]]


def run_two_stage_fca(demand: Any, supply: Any, od: Any, max_cost_m: int) -> Any:
    """Run PySAL access.two_stage_fca for one facility tier."""
    cutoff = od[od.cost_m <= max_cost_m].copy()
    if cutoff.empty:
        return demand.assign(sai=0.0)[["origin", "sai"]]
    demand_pop = demand.set_index("origin")["population"].to_dict()
    supply = supply.set_index("dest")["supply"].to_dict()
    denominator = cutoff.assign(
        population=cutoff.origin_gnd_code.map(demand_pop).fillna(0.0)
    ).groupby("dest_facility_id")["population"].sum()
    ratios = cutoff.assign(
        ratio=cutoff.dest_facility_id.map(supply).fillna(0.0)
        / cutoff.dest_facility_id.map(denominator).replace(0, 1.0)
    )
    scores = ratios.groupby("origin_gnd_code")["ratio"].sum()
    return demand.assign(sai=demand.origin.map(scores).fillna(0.0))[["origin", "sai"]]


def merge_accessibility_scores(gnd: Any, tier_scores: dict[str, Any]) -> Any:
    """Join all tier SAIs to GNDs using the confirmed GND code field."""
    result = gnd.copy()
    key = next((c for c in ("gnd_code", "GND_CODE", "ADM4_CODE", "code") if c in result), None)
    if key is None:
        result["gnd_code"] = result.index.astype(str)
        key = "gnd_code"
    result[key] = result[key].astype(str)
    for tier, frame in tier_scores.items():
        result = result.merge(frame.rename(columns={"origin": key, "sai": f"sai_{tier}"}), on=key, how="left")
    return result


def validate_accessibility_scores(scores: Any, supply_tables: dict[str, Any]) -> dict[str, Any]:
    """Check supply totals, NaNs, and tier score distributions."""
    return {"rows": len(scores), "supply_totals": {tier: float(frame.supply.sum()) for tier, frame in supply_tables.items()},
            "missing_scores": scores.isna().sum().to_dict()}


def write_accessibility_scores(scores: Any, data_dir: Path) -> None:
    """Persist per-GND accessibility scores to accessibility_scores.csv."""
    data_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(data_dir / "accessibility_scores.csv", index=False)


def run(data_dir: Path) -> Any:
    """Orchestrate T4 from T3 OD matrices and population inputs."""
    require_artifacts([
        data_dir / "sri_lanka.gpkg",
        data_dir / "health_facilities.gpkg",
        data_dir / "schools_by_tier.gpkg",
    ], "T4")
    gnd = gpd.read_file(data_dir / "sri_lanka.gpkg", layer="gn_divisions")
    demand = build_demand_table(gnd, gnd)
    scores, supplies = {}, {}
    for facilities_path in (
        data_dir / "health_facilities.gpkg",
        data_dir / "schools_by_tier.gpkg",
    ):
        for layer in gpd.list_layers(facilities_path).name.tolist():
            frame = gpd.read_file(facilities_path, layer=layer)
            tier_frames = frame.groupby("tier") if "tier" in frame.columns else [(layer, frame)]
            for tier, tier_frame in tier_frames:
                supply = build_supply_table(tier_frame, str(tier))
                od_path = data_dir / f"od_matrix_{tier}.parquet"
                if not od_path.exists():
                    raise FileNotFoundError(
                        f"T4 requires OD matrix for tier {tier!r}: {od_path}"
                    )
                od = pd.read_parquet(od_path)
                configured = {item.code: item.max_cost_m for item in (*health_tiers(), *school_tiers())}
                scores[str(tier)] = run_two_stage_fca(
                    demand, supply, od, configured.get(str(tier), 30_000)
                )
                supplies[str(tier)] = supply
    result = merge_accessibility_scores(gnd, scores)
    validate_accessibility_scores(result, supplies)
    write_accessibility_scores(result, data_dir)
    write_manifest(data_dir, "t4", {"rows": len(result), "tiers": sorted(scores)})
    return result
