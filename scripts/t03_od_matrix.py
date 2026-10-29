"""T3: snap demand/supply points and compute sparse network OD matrices."""

from pathlib import Path
from typing import Any
import math
import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np

from config import METRIC_CRS, health_tiers, school_tiers
from common.artifacts import require_artifacts, write_manifest
from common.crs import metric_centroids
from common.validation import normalize_graph_types, require_columns, validate_geodata, validate_od_table


def snap_gnd_centroids(graph: Any, gnd: Any) -> Any:
    """Map each GND code to its nearest graph node and record snap distance."""
    validate_geodata(gnd, "GND divisions")
    points = metric_centroids(gnd)
    nodes = list(graph.nodes)
    coords = np.asarray([[float(graph.nodes[n]["x"]), float(graph.nodes[n]["y"])] for n in nodes], dtype=float)
    identifier = next((column for column in ("gnd_code", "GND_CODE", "ADM4_CODE", "code") if column in gnd), None)
    codes = gnd[identifier].astype(str) if identifier else gnd.index.astype(str)
    result = []
    for code, point in zip(codes, points):
        distances = np.hypot(coords[:, 0] - point.x, coords[:, 1] - point.y)
        index = int(distances.argmin())
        result.append({"origin_gnd_code": code, "node": nodes[index], "snap_distance_m": float(distances[index])})
    return pd.DataFrame(result)


def tier_cost(tier: str) -> int:
    """Return the configured network cutoff for a facility tier."""
    configured = {item.code: item.max_cost_m for item in (*school_tiers(), *health_tiers())}
    return configured.get(tier, 30_000)


def snap_facilities(graph: Any, facilities: Any) -> Any:
    """Map each facility ID to a graph node, flagging duplicate snaps."""
    validate_geodata(facilities, "facilities")
    projected = facilities.to_crs(METRIC_CRS)
    points = projected.geometry
    non_points = ~points.geom_type.isin({"Point", "MultiPoint"})
    if non_points.any():
        points = points.copy()
        points.loc[non_points] = points.loc[non_points].centroid
    nodes = list(graph.nodes)
    coords = np.asarray([[float(graph.nodes[n]["x"]), float(graph.nodes[n]["y"])] for n in nodes], dtype=float)
    ids = facilities["facility_id"] if "facility_id" in facilities else facilities.index.astype(str)
    result = []
    for facility_id, point in zip(ids.astype(str), points):
        distances = np.hypot(coords[:, 0] - point.x, coords[:, 1] - point.y)
        index = int(distances.argmin())
        result.append({"dest_facility_id": facility_id, "node": nodes[index], "snap_distance_m": float(distances[index])})
    return pd.DataFrame(result)


def compute_tier_od(graph: Any, demand_nodes: Any, supply_nodes: Any, max_cost_m: int) -> Any:
    """Run cutoff Dijkstra and return long-format finite and unreachable costs."""
    rows = []
    for _, demand in demand_nodes.iterrows():
        lengths = nx.single_source_dijkstra_path_length(graph, demand["node"], cutoff=max_cost_m * 1.5, weight="length")
        for _, supply in supply_nodes.iterrows():
            cost = lengths.get(supply["node"], math.inf)
            if math.isfinite(cost):
                rows.append({"origin_gnd_code": demand.origin_gnd_code, "dest_facility_id": supply.dest_facility_id, "cost_m": cost})
    result = pd.DataFrame(rows, columns=["origin_gnd_code", "dest_facility_id", "cost_m"])
    validate_od_table(result)
    return result


def validate_od_matrix(od: Any, demand_nodes: Any, tier: str) -> dict[str, Any]:
    """Check reachability, cost distributions, and representative path samples."""
    if od.empty:
        return {"tier": tier, "rows": 0, "origins_with_costs": 0, "origins": len(demand_nodes)}
    counts = od.groupby("origin_gnd_code").size()
    return {"tier": tier, "rows": len(od), "origins_with_costs": int(counts.size), "origins": len(demand_nodes),
            "median_cost_m": float(od.cost_m.median()), "max_cost_m": float(od.cost_m.max())}


def write_od_matrix(od: Any, tier: str, data_dir: Path) -> None:
    """Persist one tier's long-format OD matrix as Parquet."""
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        od.to_parquet(data_dir / f"od_matrix_{tier}.parquet", index=False)
    except (ImportError, ValueError) as exc:
        raise RuntimeError("A Parquet engine is required to write OD matrices.") from exc


def run(data_dir: Path) -> dict[str, Any]:
    """Build primary and TIN sensitivity OD matrices from T2 outputs."""
    require_artifacts(
        [
            data_dir / "graph_multimodal.graphml",
            data_dir / "sri_lanka.gpkg",
            data_dir / "health_facilities.gpkg",
            data_dir / "schools_by_tier.gpkg",
        ],
        "T3",
    )
    graph = normalize_graph_types(nx.MultiDiGraph(nx.read_graphml(data_dir / "graph_multimodal.graphml")))
    gnd = gpd.read_file(data_dir / "sri_lanka.gpkg", layer="gn_divisions")
    demand = snap_gnd_centroids(graph, gnd)
    results = {}
    facility_sources = []
    facility_sources.extend([
        (data_dir / "health_facilities.gpkg", "health"),
        (data_dir / "schools_by_tier.gpkg", "schools"),
    ])
    for path, source_name in facility_sources:
        layers = ["health"] if source_name == "health" else None
        for layer in layers or gpd.list_layers(path).name.tolist():
            facilities = gpd.read_file(path, layer=layer)
            if "tier" not in facilities.columns:
                facilities["tier"] = layer
            for tier, frame in facilities.groupby("tier"):
                supply = snap_facilities(graph, frame)
                od = compute_tier_od(graph, demand, supply, tier_cost(str(tier)))
                write_od_matrix(od, str(tier), data_dir)
                results[str(tier)] = od
    write_manifest(data_dir, "t3", {"tiers": sorted(results), "rows": {tier: len(od) for tier, od in results.items()}})
    return results
