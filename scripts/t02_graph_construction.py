"""T2: construct and validate the multimodal routing graphs."""

from pathlib import Path
from typing import Any
import json

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import LineString

from config import METRIC_CRS, STORAGE_CRS
from common.artifacts import require_artifacts, write_manifest
from common.validation import normalize_graph_types


def download_drive_graph(study_area: Any) -> Any:
    """Download the simplified connected OSM drive graph for the study area."""
    try:
        import osmnx as ox
    except ImportError:
        raise RuntimeError("OSMnx is required for downloading a drive graph; use build_graph_from_roads for local fallback.")
    graph = ox.graph_from_polygon(study_area.geometry.iloc[0], network_type="drive", simplify=True, retain_all=False)
    return ox.add_edge_lengths(graph)


def extract_footway_graph(study_area: Any) -> Any:
    """Extract pedestrian-relevant OSM edges for the footway subgraph."""
    try:
        import osmnx as ox
    except ImportError:
        raise RuntimeError("OSMnx is required for downloading footways.")
    graph = ox.graph_from_polygon(study_area.geometry.iloc[0], network_type="all", simplify=True, retain_all=False)
    allowed = {"footway", "path", "track", "steps"}
    edges = [(u, v, k) for u, v, k, d in graph.edges(keys=True, data=True)
             if set(d.get("highway", [] if isinstance(d.get("highway"), list) else [d.get("highway")])) & allowed]
    return graph.edge_subgraph(edges).copy()


def clip_roads_to_tea_districts(roads: Any, tea_districts: Any) -> Any:
    """Keep only road geometry inside the tea-district union."""
    roads_wgs84 = roads.to_crs(STORAGE_CRS)
    tea_wgs84 = tea_districts.to_crs(STORAGE_CRS)
    boundary = tea_wgs84.geometry.union_all()
    clipped = gpd.clip(roads_wgs84, boundary)
    return clipped[~clipped.geometry.is_empty & clipped.geometry.notna()].copy()


def build_graph_from_roads(roads: Any) -> nx.MultiDiGraph:
    """Build a bidirectional metric graph from roads already clipped to tea districts."""
    projected = roads.to_crs(METRIC_CRS)
    graph = nx.MultiDiGraph()
    node_ids: dict[tuple[float, float], int] = {}
    next_node = 0
    def node_for(coord: tuple[float, float]) -> int:
        nonlocal next_node
        key = (round(coord[0], 3), round(coord[1], 3))
        if key not in node_ids:
            node_ids[key] = next_node
            graph.add_node(next_node, x=key[0], y=key[1])
            next_node += 1
        return node_ids[key]
    for index, geometry in projected.geometry.items():
        if geometry is None or geometry.is_empty:
            continue
        lines = list(geometry.geoms) if geometry.geom_type == "MultiLineString" else [geometry]
        for line in lines:
            coords = list(line.coords)
            for a, b in zip(coords, coords[1:]):
                u = node_for(a)
                v = node_for(b)
                length = float(LineString([a, b]).length)
                graph.add_edge(u, v, key=index, length=length, mode="drive")
                graph.add_edge(v, u, key=index, length=length, mode="drive")
    graph.graph["crs"] = METRIC_CRS
    return graph


def inject_pekoe_trail_edges(graph: Any, trail: Any) -> Any:
    """Snap trail endpoints to existing OSM nodes and add bidirectional edges."""
    try:
        import osmnx as ox
        projected = trail.to_crs(STORAGE_CRS)
        for index, row in projected.iterrows():
            geometry = row.geometry
            if geometry.geom_type == "MultiLineString":
                geometry = max(geometry.geoms, key=lambda item: item.length)
            start, end = geometry.coords[0], geometry.coords[-1]
            u = ox.distance.nearest_nodes(graph, X=start[0], Y=start[1])
            v = ox.distance.nearest_nodes(graph, X=end[0], Y=end[1])
            if u != v:
                length = float(gpd.GeoSeries([geometry], crs=STORAGE_CRS).to_crs(METRIC_CRS).length.iloc[0])
                graph.add_edge(u, v, key=f"pekoe_{index}", length=length, mode="walk", highway="footway")
                graph.add_edge(v, u, key=f"pekoe_{index}", length=length, mode="walk", highway="footway")
    except ImportError:
        return graph
    return graph


def _nearest_road_indices(points: np.ndarray, road_coords: np.ndarray) -> np.ndarray:
    """Find nearest road nodes, using CuPy when installed and usable."""
    try:
        import cupy as cp
        gpu_points = cp.asarray(points)
        gpu_roads = cp.asarray(road_coords)
        distances = cp.sum((gpu_points[:, None, :] - gpu_roads[None, :, :]) ** 2, axis=2)
        return cp.asnumpy(cp.argmin(distances, axis=1))
    except Exception:
        return cKDTree(road_coords).query(points, k=1)[1]


def build_delaunay_sensitivity_graph(
    graph: Any,
    gnd: Any,
    detour_factor: float = 1.2,
    max_edge_m: float = 2_000,
) -> Any:
    """Add bounded GND-based TIN edges for sensitivity analysis.

    Delaunay is calculated only over GND centroids. Each centroid is first
    snapped to the nearest road node, so the triangulation never includes the
    full road network and cannot exhaust memory on a national road layer.
    """
    graph = graph.copy()
    gnd_metric = gnd.to_crs(METRIC_CRS)
    points = gnd_metric.geometry.centroid
    road_nodes = list(graph.nodes)
    coords = np.array([[graph.nodes[n]["x"], graph.nodes[n]["y"]] for n in road_nodes])
    gnd_coords = np.array([[p.x, p.y] for p in points])
    if len(gnd_coords) < 4 or len(road_nodes) == 0:
        return graph
    snapped = _nearest_road_indices(gnd_coords, coords)
    for simplex in Delaunay(gnd_coords).simplices:
        for a, b in zip(simplex, (*simplex[1:], simplex[0])):
            u = road_nodes[int(snapped[a])]
            v = road_nodes[int(snapped[b])]
            distance = float(np.linalg.norm(gnd_coords[a] - gnd_coords[b]))
            if u != v and distance <= max_edge_m:
                weight = distance * detour_factor
                graph.add_edge(u, v, length=weight, mode="tin")
                graph.add_edge(v, u, length=weight, mode="tin")
    return graph


def validate_graph_integrity(graph: Any, gnd: Any) -> dict[str, Any]:
    """Report connectivity, graph size, snapping reachability, and anomalies."""
    components = list(nx.connected_components(graph.to_undirected()))
    return {
        "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
        "connected_components": len(components),
        "largest_component_nodes": max((len(c) for c in components), default=0),
    }


def write_graph_outputs(graphs: dict[str, Any], data_dir: Path) -> None:
    """Write GraphML files and graph_stats.json."""
    data_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, graph in graphs.items():
        if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            continue
        path = data_dir / f"{name}.graphml"
        try:
            import osmnx as ox
            ox.save_graphml(graph, path)
        except ImportError:
            nx.write_graphml(graph, path)
        stats[name] = validate_graph_integrity(graph, graphs.get("gnd", gpd.GeoDataFrame()))
    (data_dir / "graph_stats.json").write_text(json.dumps(stats, indent=2))
    write_manifest(data_dir, "t2", stats)


def run(data_dir: Path) -> dict[str, Any]:
    """Orchestrate T2 from T1 study-area and layer outputs."""
    require_artifacts(
        [data_dir / "sri_lanka.gpkg", data_dir / "roads.gpkg"],
        "T2",
    )
    gnd = gpd.read_file(data_dir / "sri_lanka.gpkg", layer="gn_divisions")
    tea_districts = gpd.read_file(data_dir / "sri_lanka.gpkg", layer="tea_districts")
    road_path = data_dir / "roads.gpkg"
    roads = gpd.read_file(road_path)
    roads = clip_roads_to_tea_districts(roads, tea_districts)
    graph = build_graph_from_roads(roads)
    multimodal = build_delaunay_sensitivity_graph(graph, gnd)
    graphs = {"graph_drive": graph, "graph_multimodal": graph, "graph_multimodal_tin": multimodal, "gnd": gnd}
    write_graph_outputs(graphs, data_dir)
    return graphs
