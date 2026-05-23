from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import osmnx as ox
import pandas as pd
import numpy as np

from helpers import (
    haversine_km,
    road_adjusted_km,
    ensure_preview_node_ids as helpers_ensure_preview_node_ids,
)
from config import OSM_CACHE_DIR


def build_preview_points(assign_df: pd.DataFrame) -> pd.DataFrame:
    work = ensure_preview_node_ids(assign_df)
    if work.empty:
        return pd.DataFrame(columns=["point_id", "lat", "lon", "kind"])

    depot_lat = float(work.iloc[0]["depot_lat"])
    depot_lon = float(work.iloc[0]["depot_lon"])

    rows: List[Dict[str, Any]] = [
        {"point_id": "DEPOT", "lat": depot_lat, "lon": depot_lon, "kind": "depot"}
    ]

    seen = set()
    for row in work.itertuples(index=False):
        pid = str(row.node_id)
        if pid in seen:
            continue
        seen.add(pid)
        rows.append(
            {
                "point_id": pid,
                "lat": float(row.customer_lat),
                "lon": float(row.customer_lon),
                "kind": "customer",
            }
        )

    return pd.DataFrame(rows)


def ensure_preview_node_ids(assign_df: pd.DataFrame) -> pd.DataFrame:
    return helpers_ensure_preview_node_ids(assign_df)


def load_or_build_osm_preview_graph(points_df: pd.DataFrame):
    if points_df.empty:
        return None

    depot_rows = points_df[points_df["kind"] == "depot"].copy()
    if depot_rows.empty:
        return None

    depot_lat = float(depot_rows.iloc[0]["lat"])
    depot_lon = float(depot_rows.iloc[0]["lon"])

    max_preview_km = 0.0
    for row in points_df.itertuples(index=False):
        d = haversine_km(depot_lat, depot_lon, float(row.lat), float(row.lon))
        if d > max_preview_km:
            max_preview_km = d

    graph_radius_km = min(max(3.0, max_preview_km * 1.15), 20.0)
    graph_radius_m = int(graph_radius_km * 1000.0)

    cache_name = (
        f"osm_point_{depot_lat:.5f}_{depot_lon:.5f}_{graph_radius_m}m.graphml".replace(
            "-", "m"
        )
    )
    cache_path = OSM_CACHE_DIR / cache_name

    if cache_path.exists():
        G = ox.load_graphml(cache_path)
    else:
        G = ox.graph_from_point(
            (depot_lat, depot_lon),
            dist=graph_radius_m,
            network_type="drive",
            simplify=True,
        )
        ox.save_graphml(G, cache_path)

    return ox.project_graph(G)


def snap_preview_points_to_osm(points_df: pd.DataFrame, G_proj) -> pd.DataFrame:
    out = points_df.copy()
    if out.empty or G_proj is None:
        out["osm_node"] = np.nan
        return out

    import geopandas as gpd
    from shapely.geometry import Point

    pts = gpd.GeoDataFrame(
        out.copy(),
        geometry=[Point(lon, lat) for lon, lat in zip(out["lon"], out["lat"])],
        crs="EPSG:4326",
    ).to_crs(G_proj.graph["crs"])

    xs = pts.geometry.x.to_numpy()
    ys = pts.geometry.y.to_numpy()

    out["osm_node"] = ox.distance.nearest_nodes(G_proj, X=xs, Y=ys)
    return out


def build_preview_distance_matrix(
    assign_df: pd.DataFrame,
    osm_threshold_km: float = 14.0,
) -> Dict[str, Dict[str, float]]:
    work = ensure_preview_node_ids(assign_df)
    if work.empty:
        return {}

    depot_lat = float(work.iloc[0]["depot_lat"])
    depot_lon = float(work.iloc[0]["depot_lon"])

    max_spread_km = float(
        work.apply(
            lambda r: haversine_km(
                depot_lat, depot_lon, float(r["customer_lat"]), float(r["customer_lon"])
            ),
            axis=1,
        ).max()
    )

    # If preview is too geographically spread out, skip OSM and use proxy only
    if max_spread_km > osm_threshold_km:
        points_df = build_preview_points(work)
        point_ids = points_df["point_id"].astype(str).tolist()
        matrix: Dict[str, Dict[str, float]] = {k: {} for k in point_ids}

        coord_lookup = {
            str(r["point_id"]): (float(r["lat"]), float(r["lon"]))
            for _, r in points_df.iterrows()
        }

        for i, a in enumerate(point_ids):
            a_lat, a_lon = coord_lookup[a]
            for j, b in enumerate(point_ids):
                if i == j:
                    matrix[a][b] = 0.0
                elif b in matrix and a in matrix[b]:
                    matrix[a][b] = matrix[b][a]
                else:
                    b_lat, b_lon = coord_lookup[b]
                    matrix[a][b] = road_adjusted_km(a_lat, a_lon, b_lat, b_lon)

        return matrix

    points_df = build_preview_points(work)

    try:
        G_proj = load_or_build_osm_preview_graph(points_df)
        snapped = snap_preview_points_to_osm(points_df, G_proj)
    except Exception:
        G_proj = None
        snapped = points_df.copy()
        snapped["osm_node"] = np.nan

    point_ids = snapped["point_id"].astype(str).tolist()
    matrix: Dict[str, Dict[str, float]] = {k: {} for k in point_ids}

    node_lookup = {str(r["point_id"]): r["osm_node"] for _, r in snapped.iterrows()}
    coord_lookup = {
        str(r["point_id"]): (float(r["lat"]), float(r["lon"]))
        for _, r in snapped.iterrows()
    }

    dijkstra_cache: Dict[Any, Dict[Any, float]] = {}

    for a in point_ids:
        a_node = node_lookup.get(a)

        if pd.notna(a_node) and G_proj is not None:
            try:
                dijkstra_cache[a_node] = nx.single_source_dijkstra_path_length(
                    G_proj,
                    source=a_node,
                    weight="length",
                )
            except Exception:
                dijkstra_cache[a_node] = {}

    for i, a in enumerate(point_ids):
        a_lat, a_lon = coord_lookup[a]
        a_node = node_lookup.get(a)

        for j, b in enumerate(point_ids):
            if i == j:
                matrix[a][b] = 0.0
                continue

            if b in matrix and a in matrix[b]:
                matrix[a][b] = matrix[b][a]
                continue

            b_lat, b_lon = coord_lookup[b]
            b_node = node_lookup.get(b)

            dist_km: Optional[float] = None

            if G_proj is not None and pd.notna(a_node) and pd.notna(b_node):
                dist_m = dijkstra_cache.get(a_node, {}).get(b_node)
                if dist_m is not None:
                    dist_km = float(dist_m) / 1000.0

            if dist_km is None:
                dist_km = road_adjusted_km(a_lat, a_lon, b_lat, b_lon)

            matrix[a][b] = dist_km

    return matrix


def load_or_build_osm_preview_graphs(points_df: pd.DataFrame):
    if points_df.empty:
        return None, None

    depot_rows = points_df[points_df["kind"] == "depot"].copy()
    if depot_rows.empty:
        return None, None

    depot_lat = float(depot_rows.iloc[0]["lat"])
    depot_lon = float(depot_rows.iloc[0]["lon"])

    max_preview_km = 0.0
    for row in points_df.itertuples(index=False):
        d = haversine_km(depot_lat, depot_lon, float(row.lat), float(row.lon))
        max_preview_km = max(max_preview_km, d)

    graph_radius_km = min(max(3.0, max_preview_km * 1.15), 20.0)
    graph_radius_m = int(graph_radius_km * 1000.0)

    cache_name = (
        f"osm_point_{depot_lat:.5f}_{depot_lon:.5f}_{graph_radius_m}m.graphml".replace(
            "-", "m"
        )
    )
    cache_path = OSM_CACHE_DIR / cache_name

    if cache_path.exists():
        G_raw = ox.load_graphml(cache_path)
    else:
        G_raw = ox.graph_from_point(
            (depot_lat, depot_lon),
            dist=graph_radius_m,
            network_type="drive",
            simplify=True,
        )
        ox.save_graphml(G_raw, cache_path)

    G_proj = ox.project_graph(G_raw)
    return G_raw, G_proj


def build_snapped_point_lookup(points_df: pd.DataFrame, G_proj) -> Dict[str, Any]:
    snapped = snap_preview_points_to_osm(points_df, G_proj)
    return {str(r["point_id"]): r["osm_node"] for _, r in snapped.iterrows()}


def path_coords_from_osm(
    G_raw,
    node_path: List[Any],
) -> List[Dict[str, float]]:
    coords: List[Dict[str, float]] = []

    for idx, node_id in enumerate(node_path):
        node_data = G_raw.nodes[node_id]
        point = {"lat": float(node_data["y"]), "lon": float(node_data["x"])}

        if idx == 0 or coords[-1] != point:
            coords.append(point)

    return coords


def build_display_leg_path(
    start_point_id: str,
    end_point_id: str,
    coord_lookup: Dict[str, Tuple[float, float]],
    node_lookup: Dict[str, Any],
    G_raw,
) -> List[Dict[str, float]]:
    start_lat, start_lon = coord_lookup[start_point_id]
    end_lat, end_lon = coord_lookup[end_point_id]

    start_node = node_lookup.get(start_point_id)
    end_node = node_lookup.get(end_point_id)

    if G_raw is not None and pd.notna(start_node) and pd.notna(end_node):
        try:
            node_path = nx.shortest_path(
                G_raw,
                source=start_node,
                target=end_node,
                weight="length",
            )
            coords = path_coords_from_osm(G_raw, node_path)
            if len(coords) >= 2:
                return coords
        except Exception:
            pass

    return [
        {"lat": float(start_lat), "lon": float(start_lon)},
        {"lat": float(end_lat), "lon": float(end_lon)},
    ]


def attach_route_display_geometry(
    routes: List[Dict[str, Any]],
    assign_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    work = ensure_preview_node_ids(assign_df)
    if work.empty:
        return routes

    points_df = build_preview_points(work)
    coord_lookup = {
        str(r["point_id"]): (float(r["lat"]), float(r["lon"]))
        for _, r in points_df.iterrows()
    }

    try:
        G_raw, G_proj = load_or_build_osm_preview_graphs(points_df)
        node_lookup = build_snapped_point_lookup(points_df, G_proj)
    except Exception:
        G_raw, G_proj = None, None
        node_lookup = {pid: np.nan for pid in coord_lookup.keys()}

    for route in routes:
        prev_point_id = "DEPOT"

        for stop in route.get("stops", []):
            stop_point_id = (
                str(stop["nodeId"]) if "nodeId" in stop else str(stop.get("nodeId"))
            )
            stop["legPath"] = build_display_leg_path(
                prev_point_id,
                stop_point_id,
                coord_lookup,
                node_lookup,
                G_raw,
            )
            prev_point_id = stop_point_id

        if route.get("stops"):
            route["returnPath"] = build_display_leg_path(
                prev_point_id,
                "DEPOT",
                coord_lookup,
                node_lookup,
                G_raw,
            )
        else:
            route["returnPath"] = []

    return routes
