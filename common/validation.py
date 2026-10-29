"""Explicit validation for data exchanged between pipeline stages."""

from typing import Any

import networkx as nx


def require_columns(frame: Any, columns: set[str], name: str) -> None:
    """Raise with the missing column names when a table schema is incomplete."""
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def validate_geodata(frame: Any, name: str, geometry_types: set[str] | None = None) -> None:
    """Validate CRS, geometry presence, and optional geometry types."""
    if getattr(frame, "crs", None) is None:
        raise ValueError(f"{name} has no CRS.")
    if "geometry" not in frame:
        raise ValueError(f"{name} has no geometry column.")
    if frame.geometry.isna().any():
        raise ValueError(f"{name} contains null geometries.")
    if not frame.geometry.is_valid.all():
        raise ValueError(f"{name} contains invalid geometries.")
    if geometry_types:
        actual = set(frame.geometry.geom_type.dropna().unique())
        if not actual.issubset(geometry_types):
            raise ValueError(f"{name} has unsupported geometry types: {sorted(actual - geometry_types)}")


def normalize_graph_types(graph: nx.Graph) -> nx.Graph:
    """Convert serialized GraphML coordinates and lengths back to numbers."""
    if graph.number_of_nodes() == 0:
        raise ValueError("Routing graph contains no nodes.")
    for node, attrs in graph.nodes(data=True):
        try:
            attrs["x"] = float(attrs["x"])
            attrs["y"] = float(attrs["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Graph node {node!r} has invalid x/y attributes.") from exc
    for u, v, attrs in graph.edges(data=True):
        try:
            attrs["length"] = float(attrs["length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Graph edge {u!r}->{v!r} has invalid length.") from exc
    return graph


def validate_od_table(table: Any) -> None:
    """Validate the long-format OD artifact."""
    require_columns(table, {"origin_gnd_code", "dest_facility_id", "cost_m"}, "OD table")
    if table["cost_m"].isna().any() or (table["cost_m"] < 0).any():
        raise ValueError("OD table contains null or negative costs.")
