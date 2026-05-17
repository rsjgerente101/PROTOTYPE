from __future__ import annotations
from pathlib import Path

import networkx as nx
import osmnx as ox
import io
import json
import math
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = FastAPI(title="Delivery Prototype Backend", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASETS: Dict[str, Dict[str, Any]] = {}
RUNS: Dict[str, Dict[str, Any]] = {}

EARTH_RADIUS_KM = 6371.0

OSM_CACHE_DIR = Path("cache/osm")
OSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# OSMnx HTTP + file caching
ox.settings.use_cache = True
ox.settings.log_console = False

RUN_PROFILES: Dict[str, Dict[str, Any]] = {
    "default_balanced": {
        "preview_initial_radius_km": 12.0,
        "preview_max_radius_km": 30.0,
        "preview_local_cap_km": 25.0,
        "preview_osm_threshold_km": 25.0,
        "preview_max_total_stops": 20,
        "enhanced_fairness_weight": 0.60,
        "enhanced_distance_weight": 0.25,
        "enhanced_time_weight": 0.15,
        "enhanced_max_iterations": 20,
        "enhanced_border_fraction": 0.50,
    },
    "amazon_expanded_search": {
        "preview_initial_radius_km": 25.0,
        "preview_max_radius_km": 120.0,
        "preview_local_cap_km": 100.0,
        "preview_osm_threshold_km": 35.0,
        "preview_max_total_stops": 60,
        "enhanced_fairness_weight": 0.35,
        "enhanced_distance_weight": 0.45,
        "enhanced_time_weight": 0.20,
        "enhanced_max_iterations": 20,
        "enhanced_border_fraction": 0.35,
    },
    "zomato_expanded_search": {
        "preview_initial_radius_km": 20.0,
        "preview_max_radius_km": 60.0,
        "preview_local_cap_km": 50.0,
        "preview_osm_threshold_km": 35.0,
        "preview_max_total_stops": 40,
        "enhanced_fairness_weight": 0.60,
        "enhanced_distance_weight": 0.25,
        "enhanced_time_weight": 0.15,
        "enhanced_max_iterations": 20,
        "enhanced_border_fraction": 0.50,
    },
}

DEMO_PREVIEW_DEPOTS: Dict[str, Optional[str]] = {
    "primary_reconstruction": "DEPOT-130",  # Amazon demo depot
    "comparative_template": "DEPOT-153",  # set this to your chosen Zomato depot ID
    "generic_uploaded_dataset": None,
}

MIN_FIXED_DEMO_NODES = 12
MIN_FIXED_DEMO_AGENTS = 6
AMAZON_FIXED_DEMO_NODES = 0
AMAZON_FIXED_DEMO_AGENTS = 0
AMAZON_DEFAULT_REPRESENTATIVES = 6
AMAZON_MAX_CUSTOMERS_PER_REP = 3
AMAZON_MIN_PREVIEW_STOPS = AMAZON_DEFAULT_REPRESENTATIVES * AMAZON_MAX_CUSTOMERS_PER_REP


class FieldMapping(BaseModel):
    depot_id: Optional[str] = None
    depot_lat: str
    depot_lon: str
    customer_id: str
    agent_id: Optional[str] = None
    customer_lat: str
    customer_lon: str
    order_id: Optional[str] = None
    order_date_col: Optional[str] = None
    eta_col: Optional[str] = None
    rating_col: Optional[str] = None
    area_col: Optional[str] = None


class BaselineRequest(BaseModel):
    dataset_id: str
    num_representatives: int = 4
    avg_speed_kmph: float = 40.0
    service_minutes_per_stop: float = 8.0
    seed: int = 42
    run_profile: Optional[str] = "default_balanced"


class EnhancedRequest(BaseModel):
    dataset_id: str
    baseline_run_id: str
    alpha_weight: Optional[float] = None
    beta_weight: Optional[float] = None
    max_iterations: Optional[int] = None
    border_fraction: Optional[float] = None
    run_profile: Optional[str] = None


class AddedCustomerPayload(BaseModel):
    label: str
    lat: float
    lon: float
    address: Optional[str] = None
    assigned_rep: Optional[str] = None
    customer_number: Optional[int] = None


class BaselineAddCustomersRequest(BaseModel):
    baseline_run_id: str
    customers: List[AddedCustomerPayload]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def road_adjusted_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Fast proxy for road-network cost.
    Keeps runtime close to current haversine approach, but inflates distance
    to better approximate road travel than straight-line geometry.
    """
    direct = haversine_km(lat1, lon1, lat2, lon2)
    return direct * 1.25


def build_preview_points(assign_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build unique depot/customer preview points for matrix construction.
    """
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
    """
    Snap preview depot/customers to nearest OSM nodes.
    """
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
    """
    Build preview-only pairwise cost matrix using OSM shortest-path lengths.
    Falls back to road_adjusted_km if graph download/snap/path fails.
    """
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

    print(f"max_spread_km before OSM check: {max_spread_km:.2f}")

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

        print(
            f"Preview spread too large for OSM ({max_spread_km:.2f} km > {osm_threshold_km:.2f} km). Using proxy matrix."
        )
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
    """
    Return both:
    - G_raw: unprojected graph for lat/lon geometry extraction
    - G_proj: projected graph for nearest-node snapping
    """
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
            stop_point_id = str(stop["nodeId"])
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


def preview_matrix_stats(assign_df: pd.DataFrame) -> Dict[str, Any]:
    work = ensure_preview_node_ids(assign_df)
    if work.empty:
        return {
            "previewPoints": 0,
            "matrixPairs": 0,
        }

    unique_points = 1 + int(work["node_id"].nunique())  # depot + customers
    return {
        "previewPoints": unique_points,
        "matrixPairs": unique_points * unique_points,
    }


def filter_df_to_demo_depot(
    df: pd.DataFrame,
    dataset_role: str,
    min_nodes: int = MIN_FIXED_DEMO_NODES,
    min_agents: int = MIN_FIXED_DEMO_AGENTS,
) -> pd.DataFrame:
    demo_depot_id = DEMO_PREVIEW_DEPOTS.get(dataset_role)

    if not demo_depot_id or "depot_id" not in df.columns:
        return df.copy()

    filtered = df[df["depot_id"].astype(str) == str(demo_depot_id)].copy()

    if filtered.empty:
        print(
            f"demo depot override {demo_depot_id} not found; selecting strongest available depot instead"
        )
        fallback_depot_id = choose_best_demo_depot_id(
            df, min_nodes=min_nodes, min_agents=min_agents
        )
        if fallback_depot_id is None:
            return df.copy()
        filtered = df[df["depot_id"].astype(str) == str(fallback_depot_id)].copy()
        print(
            f"using fallback demo depot: {fallback_depot_id} ({len(filtered)} rows before preview trimming)"
        )
        return filtered

    summary = summarize_demo_depot_strength(filtered)
    print(f"configured fixed demo depot: {demo_depot_id} summary: {summary}")

    too_weak = summary["nodes"] < min_nodes or summary["agents"] < min_agents

    if too_weak:
        print(
            f"configured demo depot {demo_depot_id} is too weak "
            f"(needs at least {min_nodes} nodes and {min_agents} agents). "
            f"Selecting strongest available depot instead."
        )
        fallback_depot_id = choose_best_demo_depot_id(
            df, min_nodes=min_nodes, min_agents=min_agents
        )
        if fallback_depot_id and fallback_depot_id != str(demo_depot_id):
            filtered = df[df["depot_id"].astype(str) == str(fallback_depot_id)].copy()
            print(
                f"using stronger fallback demo depot: {fallback_depot_id} ({len(filtered)} rows before preview trimming)"
            )
            return filtered

    print(
        f"using fixed demo depot: {demo_depot_id} ({len(filtered)} rows before preview trimming)"
    )
    return filtered


def summarize_demo_depot_strength(df: pd.DataFrame) -> Dict[str, Any]:
    work = df.copy()
    if work.empty:
        return {
            "rows": 0,
            "nodes": 0,
            "agents": 0,
            "orders": 0,
        }

    nodes = (
        int(work["customer_node_id"].nunique())
        if "customer_node_id" in work.columns
        else int(len(work))
    )
    agents = 0
    if "agent_id" in work.columns:
        agent_series = (
            work["agent_id"]
            .astype(str)
            .replace({"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"})
        )
        agents = int(agent_series[agent_series != "UNKNOWN"].nunique())

    orders = 0
    if "node_order_count" in work.columns:
        orders = int(
            pd.to_numeric(work["node_order_count"], errors="coerce").fillna(1).sum()
        )
    else:
        orders = int(len(work))

    return {
        "rows": int(len(work)),
        "nodes": nodes,
        "agents": agents,
        "orders": orders,
    }


def choose_best_demo_depot_id(
    routing_df: pd.DataFrame,
    min_nodes: int = MIN_FIXED_DEMO_NODES,
    min_agents: int = MIN_FIXED_DEMO_AGENTS,
) -> Optional[str]:
    if routing_df.empty or "depot_id" not in routing_df.columns:
        return None

    best_score = None
    best_depot_id = None

    for depot_id, grp in routing_df.groupby("depot_id"):
        summary = summarize_demo_depot_strength(grp)

        score = (
            0 if summary["nodes"] >= min_nodes else 1,
            0 if summary["agents"] >= min_agents else 1,
            -summary["agents"],
            -summary["nodes"],
            -summary["orders"],
        )

        print(
            "demo depot candidate:",
            {
                "depot_id": str(depot_id),
                **summary,
            },
        )

        if best_score is None or score < best_score:
            best_score = score
            best_depot_id = str(depot_id)

    print("best demo depot selected:", best_depot_id, "score:", best_score)
    return best_depot_id


def ensure_preview_node_ids(assign_df: pd.DataFrame) -> pd.DataFrame:
    work = assign_df.copy().reset_index(drop=True)

    if "customer_node_id" in work.columns:
        work["node_id"] = work["customer_node_id"].astype(str)
    elif "node_id" not in work.columns:
        work["node_id"] = [f"CUST-{i + 1}" for i in range(len(work))]

    return work


def matrix_cost(matrix: Dict[str, Dict[str, float]], a: str, b: str) -> float:
    if a == b:
        return 0.0
    return float(matrix.get(a, {}).get(b, 0.0))


def parse_order_date_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()

    # First try day-first parsing, which matches Zomato-style Order_Date better
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)

    # Fallback: try default parsing for already ISO-like values
    fallback_mask = parsed.isna()
    if fallback_mask.any():
        parsed.loc[fallback_mask] = pd.to_datetime(
            text.loc[fallback_mask], errors="coerce"
        )

    if parsed.notna().any():
        return parsed.dt.normalize()
    return parsed


def get_run_profile(profile_name: Optional[str]) -> Dict[str, Any]:
    key = (profile_name or "default_balanced").strip()
    if key not in RUN_PROFILES:
        key = "default_balanced"
    profile = RUN_PROFILES[key].copy()
    profile["profile_name"] = key
    return profile


def build_routing_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert cleaned order-level rows into routing-node rows.
    One node = one customer_node_id within one depot.
    """
    work = df.copy()

    if "is_routing_eligible" in work.columns:
        work = work[work["is_routing_eligible"].fillna(False).astype(bool)].copy()

    required = [
        "depot_id",
        "depot_lat",
        "depot_lon",
        "customer_node_id",
        "customer_lat",
        "customer_lon",
        "customer_name",
        "observed_eta_min",
        "predicted_eta_min",
        "rating",
        "area",
    ]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise HTTPException(
            status_code=400, detail=f"Missing cleaned dataset columns: {missing}"
        )

    if "agent_id" not in work.columns:
        work["agent_id"] = "UNKNOWN"

    if "order_date" not in work.columns:
        work["order_date"] = pd.NaT

    agg = work.groupby(
        [
            "depot_id",
            "depot_lat",
            "depot_lon",
            "customer_node_id",
            "customer_lat",
            "customer_lon",
        ],
        as_index=False,
    ).agg(
        order_id=("order_id", "first"),
        order_date=("order_date", "first"),
        customer_id=("customer_id", "first"),
        agent_id=("agent_id", "first"),
        customer_name=("customer_name", "first"),
        observed_eta_min=("observed_eta_min", "mean"),
        predicted_eta_min=("predicted_eta_min", "mean"),
        rating=("rating", "mean"),
        area=("area", "first"),
        node_order_count=("node_order_count", "max"),
        direct_depot_customer_km=("direct_depot_customer_km", "mean"),
    )

    agg["order_id"] = agg["order_id"].astype(str)
    agg["customer_id"] = agg["customer_id"].astype(str)
    agg["customer_node_id"] = agg["customer_node_id"].astype(str)
    agg["depot_id"] = agg["depot_id"].astype(str)
    agg["customer_name"] = agg["customer_name"].fillna(agg["customer_node_id"])
    agg["node_name"] = agg["customer_name"]

    return agg.reset_index(drop=True)


def build_amazon_order_routing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Amazon preview routing should preserve order-level rows instead of collapsing
    repeated customer_node_id values into only 12 physical nodes per depot.

    The original customer_node_id is retained as physical_customer_node_id, while
    customer_node_id is made unique per order so each row can appear as a route stop.
    """
    work = df.copy()

    if "is_routing_eligible" in work.columns:
        work = work[work["is_routing_eligible"].fillna(False).astype(bool)].copy()

    required = [
        "depot_id",
        "depot_lat",
        "depot_lon",
        "customer_node_id",
        "customer_lat",
        "customer_lon",
        "customer_name",
        "observed_eta_min",
        "predicted_eta_min",
        "rating",
        "area",
        "order_id",
        "customer_id",
        "agent_id",
    ]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise HTTPException(
            status_code=400, detail=f"Missing Amazon routing columns: {missing}"
        )

    if "node_order_count" not in work.columns:
        work["node_order_count"] = 1
    if "direct_depot_customer_km" not in work.columns:
        work["direct_depot_customer_km"] = work.apply(
            lambda r: haversine_km(
                float(r["depot_lat"]),
                float(r["depot_lon"]),
                float(r["customer_lat"]),
                float(r["customer_lon"]),
            ),
            axis=1,
        )
    if "order_date" not in work.columns:
        work["order_date"] = pd.NaT

    out = work[
        [
            "depot_id",
            "depot_lat",
            "depot_lon",
            "customer_node_id",
            "customer_lat",
            "customer_lon",
            "order_id",
            "order_date",
            "customer_id",
            "agent_id",
            "customer_name",
            "observed_eta_min",
            "predicted_eta_min",
            "rating",
            "area",
            "node_order_count",
            "direct_depot_customer_km",
        ]
    ].copy()

    out["physical_customer_node_id"] = out["customer_node_id"].astype(str)
    out["order_id"] = out["order_id"].astype(str)
    out["customer_id"] = out["customer_id"].astype(str)
    out["agent_id"] = (
        out["agent_id"]
        .astype(str)
        .replace({"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"})
    )
    out["depot_id"] = out["depot_id"].astype(str)

    # Unique route stop ID: avoids the old 12-node Amazon collapse while still
    # showing where repeated orders share the same physical customer node.
    out["customer_node_id"] = (
        out["physical_customer_node_id"].astype(str)
        + "-ORDER-"
        + out["order_id"].astype(str)
    )
    out["node_name"] = out["customer_name"].astype(str)

    return out.reset_index(drop=True)


def read_csv_upload(file: UploadFile) -> pd.DataFrame:
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to read CSV: {exc}"
        ) from exc


def infer_dataset_role(filename: str) -> str:
    name = (filename or "").lower()
    if "amazon" in name:
        return "primary_reconstruction"
    if "zomato" in name:
        return "comparative_template"
    return "generic_uploaded_dataset"


def role_label(role: str) -> str:
    if role == "primary_reconstruction":
        return "Amazon Delivery Dataset (Primary Baseline Reconstruction Source)"
    if role == "comparative_template":
        return "Zomato Delivery Dataset (Comparative Template Dataset)"
    return "Uploaded Delivery Dataset"


def autofill_mapping_from_known_columns(
    df: pd.DataFrame,
    mapping: FieldMapping,
    source_role: str,
) -> FieldMapping:
    """
    Backend safety net so raw uploads normalize consistently even if the frontend
    did not send some optional mapped fields.
    """
    data = mapping.model_dump()

    columns = set(df.columns)

    if not data.get("order_date_col"):
        if "Order_Date" in columns:
            data["order_date_col"] = "Order_Date"
        elif "order_date" in columns:
            data["order_date_col"] = "order_date"

    if source_role == "comparative_template" and not data.get("agent_id"):
        if "Delivery_person_ID" in columns:
            data["agent_id"] = "Delivery_person_ID"
        elif "delivery_person_id" in columns:
            data["agent_id"] = "delivery_person_id"

    return FieldMapping(**data)


def _base_reconstruct_from_mapping(
    df: pd.DataFrame, mapping: FieldMapping
) -> pd.DataFrame:
    """
    Common reconstruction foundation for raw datasets after field mapping.
    Produces a cleaned order-level dataframe that can then be specialized
    for Amazon or Zomato.
    """
    out = pd.DataFrame()

    out["depot_lat"] = pd.to_numeric(df[mapping.depot_lat], errors="coerce")
    out["depot_lon"] = pd.to_numeric(df[mapping.depot_lon], errors="coerce")
    out["customer_lat"] = pd.to_numeric(df[mapping.customer_lat], errors="coerce")
    out["customer_lon"] = pd.to_numeric(df[mapping.customer_lon], errors="coerce")

    out["customer_id"] = df[mapping.customer_id].astype(str)
    out["order_id"] = (
        df[mapping.order_id].astype(str)
        if mapping.order_id and mapping.order_id in df.columns
        else out["customer_id"].astype(str)
    )

    date_col = None
    if mapping.order_date_col and mapping.order_date_col in df.columns:
        date_col = mapping.order_date_col
    elif "Order_Date" in df.columns:
        date_col = "Order_Date"
    elif "order_date" in df.columns:
        date_col = "order_date"

    out["order_date"] = (
        parse_order_date_series(df[date_col]) if date_col is not None else pd.NaT
    )

    out["observed_eta_min"] = (
        pd.to_numeric(df[mapping.eta_col], errors="coerce")
        if mapping.eta_col and mapping.eta_col in df.columns
        else np.nan
    )

    out["rating"] = (
        pd.to_numeric(df[mapping.rating_col], errors="coerce")
        if mapping.rating_col and mapping.rating_col in df.columns
        else np.nan
    )

    out["area"] = (
        df[mapping.area_col].astype(str)
        if mapping.area_col and mapping.area_col in df.columns
        else "UNSPECIFIED"
    )

    # Remove unusable coordinates
    out = out.dropna(
        subset=["depot_lat", "depot_lon", "customer_lat", "customer_lon"]
    ).copy()
    out = out[
        (out["depot_lat"] != 0)
        & (out["depot_lon"] != 0)
        & (out["customer_lat"] != 0)
        & (out["customer_lon"] != 0)
    ].copy()

    if out.empty:
        raise HTTPException(
            status_code=400, detail="No valid rows remain after coordinate filtering."
        )

    # Stable depot_id from unique depot coordinate pairs unless an explicit depot ID was mapped
    if mapping.depot_id and mapping.depot_id in df.columns:
        out["depot_id"] = df.loc[out.index, mapping.depot_id].astype(str)
    else:
        depot_keys = (
            out["depot_lat"].round(6).astype(str)
            + "_"
            + out["depot_lon"].round(6).astype(str)
        )
        depot_codes, _ = pd.factorize(depot_keys)
        out["depot_id"] = pd.Series(depot_codes, index=out.index).map(
            lambda x: f"DEPOT-{x + 1:03d}"
        )

    return out


def reconstruct_raw_amazon_dataset(
    df: pd.DataFrame, mapping: FieldMapping
) -> pd.DataFrame:
    """
    Reconstruct raw Amazon upload into the cleaned route-eligible schema
    aligned with the known-good reconstructed Amazon dataset design.
    """
    out = _base_reconstruct_from_mapping(df, mapping)

    # Synthetic agent identity from depot + raw Amazon Agent_Age
    age_col = "Agent_Age"
    if age_col in df.columns:
        aligned_age = pd.to_numeric(df.loc[out.index, age_col], errors="coerce")
        out["agent_age"] = aligned_age.fillna(-1).astype(int)
        out["agent_id"] = (
            "AGENT-"
            + out["depot_id"].astype(str)
            + "-AGE-"
            + out["agent_age"].astype(str)
        )
    else:
        out["agent_age"] = -1
        out["agent_id"] = "AGENT-" + out["depot_id"].astype(str) + "-AGE-UNKNOWN"

    # Node identity: group destination coordinates within depot
    node_keys = (
        out["depot_id"].astype(str)
        + "_"
        + out["customer_lat"].round(5).astype(str)
        + "_"
        + out["customer_lon"].round(5).astype(str)
    )
    node_codes, _ = pd.factorize(node_keys)
    out["customer_node_id"] = pd.Series(node_codes, index=out.index).map(
        lambda x: f"NODE-{x + 1:04d}"
    )

    # Readable UI names
    node_name_map = {
        node_id: f"Customer {i + 1:04d}"
        for i, node_id in enumerate(
            pd.Series(out["customer_node_id"]).drop_duplicates().tolist()
        )
    }
    out["customer_name"] = out["customer_node_id"].map(node_name_map)

    # Demand / repeated orders per node
    out["node_order_count"] = out.groupby(["depot_id", "customer_node_id"])[
        "order_id"
    ].transform("count")

    # Direct distance
    out["direct_depot_customer_km"] = out.apply(
        lambda r: haversine_km(
            float(r["depot_lat"]),
            float(r["depot_lon"]),
            float(r["customer_lat"]),
            float(r["customer_lon"]),
        ),
        axis=1,
    )

    # Conservative outlier threshold from the old reconstruction guidance
    out["is_distance_outlier"] = out["direct_depot_customer_km"] > 50.0
    out["is_routing_eligible"] = ~out["is_distance_outlier"]

    # Fill weak fields gently
    if out["rating"].notna().any():
        out["rating"] = out["rating"].fillna(out["rating"].median())
    else:
        out["rating"] = 4.0

    out["observed_eta_min"] = out["observed_eta_min"].fillna(
        (out["direct_depot_customer_km"] / 18.0) * 60.0 + 8.0
    )

    final = out[
        [
            "order_id",
            "order_date",
            "customer_id",
            "customer_node_id",
            "depot_id",
            "agent_id",
            "agent_age",
            "depot_lat",
            "depot_lon",
            "customer_lat",
            "customer_lon",
            "customer_name",
            "observed_eta_min",
            "rating",
            "area",
            "node_order_count",
            "direct_depot_customer_km",
            "is_distance_outlier",
            "is_routing_eligible",
        ]
    ].copy()

    final.reset_index(drop=True, inplace=True)
    return final


def reconstruct_raw_zomato_dataset(
    df: pd.DataFrame, mapping: FieldMapping
) -> pd.DataFrame:
    """
    Reconstruct raw Zomato upload into the cleaned route-eligible schema
    aligned with the same node-aware routing structure.
    """
    out = _base_reconstruct_from_mapping(df, mapping)

    agent_col = None
    if mapping.agent_id and mapping.agent_id in df.columns:
        agent_col = mapping.agent_id
    elif "Delivery_person_ID" in df.columns:
        agent_col = "Delivery_person_ID"
    elif "delivery_person_id" in df.columns:
        agent_col = "delivery_person_id"

    if agent_col:
        out["agent_id"] = df.loc[out.index, agent_col].astype(str).fillna("UNKNOWN")
        out["agent_id"] = out["agent_id"].replace(
            {"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"}
        )
    else:
        out["agent_id"] = "UNKNOWN"

    # For Zomato, reconstruct node identity from destination coordinates within depot
    node_keys = (
        out["depot_id"].astype(str)
        + "_"
        + out["customer_lat"].round(5).astype(str)
        + "_"
        + out["customer_lon"].round(5).astype(str)
    )
    node_codes, _ = pd.factorize(node_keys)
    out["customer_node_id"] = pd.Series(node_codes, index=out.index).map(
        lambda x: f"NODE-{x + 1:04d}"
    )

    # Readable UI names
    node_name_map = {
        node_id: f"Customer {i + 1:04d}"
        for i, node_id in enumerate(
            pd.Series(out["customer_node_id"]).drop_duplicates().tolist()
        )
    }
    out["customer_name"] = out["customer_node_id"].map(node_name_map)

    # Demand / repeated orders per node
    out["node_order_count"] = out.groupby(["depot_id", "customer_node_id"])[
        "order_id"
    ].transform("count")

    # Direct distance
    out["direct_depot_customer_km"] = out.apply(
        lambda r: haversine_km(
            float(r["depot_lat"]),
            float(r["depot_lon"]),
            float(r["customer_lat"]),
            float(r["customer_lon"]),
        ),
        axis=1,
    )

    # Same initial outlier threshold for consistency
    out["is_distance_outlier"] = out["direct_depot_customer_km"] > 50.0
    out["is_routing_eligible"] = ~out["is_distance_outlier"]

    if out["rating"].notna().any():
        out["rating"] = out["rating"].fillna(out["rating"].median())
    else:
        out["rating"] = 4.0

    out["observed_eta_min"] = out["observed_eta_min"].fillna(
        (out["direct_depot_customer_km"] / 18.0) * 60.0 + 8.0
    )

    final = out[
        [
            "order_id",
            "order_date",
            "customer_id",
            "customer_node_id",
            "depot_id",
            "agent_id",
            "depot_lat",
            "depot_lon",
            "customer_lat",
            "customer_lon",
            "customer_name",
            "observed_eta_min",
            "rating",
            "area",
            "node_order_count",
            "direct_depot_customer_km",
            "is_distance_outlier",
            "is_routing_eligible",
        ]
    ].copy()

    final.reset_index(drop=True, inplace=True)
    return final


def reconstruct_generic_uploaded_dataset(
    df: pd.DataFrame, mapping: FieldMapping
) -> pd.DataFrame:
    """
    Generic fallback for other uploaded delivery datasets.
    Keeps behavior simple but still produces the cleaned route-eligible schema.
    """
    out = _base_reconstruct_from_mapping(df, mapping)

    node_keys = (
        out["depot_id"].astype(str)
        + "_"
        + out["customer_lat"].round(5).astype(str)
        + "_"
        + out["customer_lon"].round(5).astype(str)
    )
    node_codes, _ = pd.factorize(node_keys)
    out["customer_node_id"] = pd.Series(node_codes, index=out.index).map(
        lambda x: f"NODE-{x + 1:04d}"
    )

    node_name_map = {
        node_id: f"Customer {i + 1:04d}"
        for i, node_id in enumerate(
            pd.Series(out["customer_node_id"]).drop_duplicates().tolist()
        )
    }
    out["customer_name"] = out["customer_node_id"].map(node_name_map)

    out["node_order_count"] = out.groupby(["depot_id", "customer_node_id"])[
        "order_id"
    ].transform("count")

    out["direct_depot_customer_km"] = out.apply(
        lambda r: haversine_km(
            float(r["depot_lat"]),
            float(r["depot_lon"]),
            float(r["customer_lat"]),
            float(r["customer_lon"]),
        ),
        axis=1,
    )

    out["is_distance_outlier"] = out["direct_depot_customer_km"] > 50.0
    out["is_routing_eligible"] = ~out["is_distance_outlier"]

    if out["rating"].notna().any():
        out["rating"] = out["rating"].fillna(out["rating"].median())
    else:
        out["rating"] = 4.0

    out["observed_eta_min"] = out["observed_eta_min"].fillna(
        (out["direct_depot_customer_km"] / 18.0) * 60.0 + 8.0
    )

    final = out[
        [
            "order_id",
            "order_date",
            "customer_id",
            "customer_node_id",
            "depot_id",
            "depot_lat",
            "depot_lon",
            "customer_lat",
            "customer_lon",
            "customer_name",
            "observed_eta_min",
            "rating",
            "area",
            "node_order_count",
            "direct_depot_customer_km",
            "is_distance_outlier",
            "is_routing_eligible",
        ]
    ].copy()

    final.reset_index(drop=True, inplace=True)
    return final


def normalize_dataset(
    df: pd.DataFrame, mapping: FieldMapping, source_role: str
) -> pd.DataFrame:
    needed = [
        mapping.depot_lat,
        mapping.depot_lon,
        mapping.customer_id,
        mapping.customer_lat,
        mapping.customer_lon,
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400, detail=f"Missing mapped columns: {missing}"
        )

    cleaned_cols = {
        "order_id",
        "customer_id",
        "customer_node_id",
        "depot_id",
        "depot_lat",
        "depot_lon",
        "customer_lat",
        "customer_lon",
        "customer_name",
        "observed_eta_min",
        "rating",
        "area",
        "node_order_count",
        "direct_depot_customer_km",
        "is_distance_outlier",
        "is_routing_eligible",
    }

    if cleaned_cols.issubset(set(df.columns)):
        out = df.copy()

        out["depot_lat"] = pd.to_numeric(out["depot_lat"], errors="coerce")
        out["depot_lon"] = pd.to_numeric(out["depot_lon"], errors="coerce")
        out["customer_lat"] = pd.to_numeric(out["customer_lat"], errors="coerce")
        out["customer_lon"] = pd.to_numeric(out["customer_lon"], errors="coerce")
        out["observed_eta_min"] = pd.to_numeric(
            out["observed_eta_min"], errors="coerce"
        )
        out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
        out["node_order_count"] = pd.to_numeric(
            out["node_order_count"], errors="coerce"
        ).fillna(1)

        if "agent_age" in out.columns:
            out["agent_age"] = (
                pd.to_numeric(out["agent_age"], errors="coerce").fillna(-1).astype(int)
            )

        if "agent_id" not in out.columns:
            if "agent_age" in out.columns:
                out["agent_id"] = (
                    "AGENT-"
                    + out["depot_id"].astype(str)
                    + "-AGE-"
                    + out["agent_age"].astype(str)
                )
            else:
                out["agent_id"] = (
                    "AGENT-" + out["depot_id"].astype(str) + "-AGE-UNKNOWN"
                )

        if "order_date" in out.columns:
            out["order_date"] = parse_order_date_series(out["order_date"])

        out = out.dropna(
            subset=["depot_lat", "depot_lon", "customer_lat", "customer_lon"]
        ).copy()
        out = out[
            (out["customer_lat"] != 0)
            & (out["customer_lon"] != 0)
            & (out["depot_lat"] != 0)
            & (out["depot_lon"] != 0)
        ].copy()

        out.reset_index(drop=True, inplace=True)
        return out

    # dataset-specific raw reconstruction
    if source_role == "primary_reconstruction":
        return reconstruct_raw_amazon_dataset(df, mapping)

    if source_role == "comparative_template":
        return reconstruct_raw_zomato_dataset(df, mapping)

    return reconstruct_generic_uploaded_dataset(df, mapping)


def validation_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        raise HTTPException(
            status_code=400, detail="No valid rows remain after coordinate filtering."
        )
    dup_orders = int(df["order_id"].duplicated().sum())
    invalid = 0
    coords = df[["customer_lat", "customer_lon"]].round(4)
    near_dupes = int(coords.duplicated().sum())

    avg_rating = 4.0
    if df["rating"].notna().any():
        avg_rating = float(df["rating"].fillna(df["rating"].median()).mean())

    return {
        "isValid": True,
        "invalidCoordinates": invalid,
        "duplicateRows": dup_orders,
        "nearDuplicates": near_dupes,
        "summary": {
            "records": int(len(df)),
            "depots": int(df["depot_id"].nunique()),
            "customers": int(df["customer_node_id"].nunique())
            if "customer_node_id" in df.columns
            else int(df["customer_id"].nunique()),
            "orders": int(df["order_id"].nunique()),
            "avgRating": round(avg_rating, 2),
        },
    }


def build_eta_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    feat["direct_distance_km"] = [
        haversine_km(r.depot_lat, r.depot_lon, r.customer_lat, r.customer_lon)
        for r in feat.itertuples(index=False)
    ]
    feat["rating"] = feat["rating"].fillna(
        feat["rating"].median() if feat["rating"].notna().any() else 4.0
    )
    feat["observed_eta_min"] = feat["observed_eta_min"].fillna(
        (feat["direct_distance_km"] / 18.0) * 60.0 + 8.0
    )
    return feat


def train_eta_models(df: pd.DataFrame, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    feat = build_eta_features(df)
    target = feat["observed_eta_min"].values
    features = feat[["direct_distance_km", "rating", "area"]]

    numeric = ["direct_distance_km", "rating"]
    categorical = ["area"]

    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )

    ridge = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])
    rf = Pipeline(
        [
            ("pre", pre),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=60,
                    max_depth=10,
                    min_samples_leaf=3,
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )

    ridge.fit(features, target)
    rf.fit(features, target)

    pred_baseline = ridge.predict(features)
    pred_enhanced = rf.predict(features)

    metrics = {
        "baseline": {
            "mae": float(mean_absolute_error(target, pred_baseline)),
            "rmse": float(np.sqrt(mean_squared_error(target, pred_baseline))),
            "r2": float(r2_score(target, pred_baseline)),
        },
        "enhanced": {
            "mae": float(mean_absolute_error(target, pred_enhanced)),
            "rmse": float(np.sqrt(mean_squared_error(target, pred_enhanced))),
            "r2": float(r2_score(target, pred_enhanced)),
        },
    }
    return pred_enhanced, metrics


def route_one_rep(
    group: pd.DataFrame,
    speed_kmph: float,
    service_min: float,
    distance_matrix: Dict[str, Dict[str, float]],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    rows = ensure_preview_node_ids(group).to_dict("records")
    if not rows:
        return [], {
            "distance_km": 0.0,
            "travel_minutes": 0.0,
            "operational_minutes": 0.0,
        }

    current_node = "DEPOT"
    unvisited = rows[:]

    route: List[Dict[str, Any]] = []
    cumulative_distance = 0.0
    cumulative_eta = 0.0
    stop_no = 1

    while unvisited:
        best = min(
            unvisited,
            key=lambda r: matrix_cost(distance_matrix, current_node, str(r["node_id"])),
        )

        leg = matrix_cost(distance_matrix, current_node, str(best["node_id"]))
        cumulative_distance += leg

        travel_min = (leg / speed_kmph) * 60.0 if speed_kmph > 0 else 0.0
        service_component = float(service_min)
        cumulative_eta += travel_min + service_component

        route.append(
            {
                "stopNumber": stop_no,
                "nodeId": best.get("customer_node_id", best["customer_id"]),
                "nodeName": best["customer_name"],
                "orderCount": int(best.get("node_order_count", 1)),
                "lat": float(best["customer_lat"]),
                "lon": float(best["customer_lon"]),
                "legDistance": round(leg, 2),
                "cumulativeDistance": round(cumulative_distance, 2),
                "eta": round(cumulative_eta, 2),
                "orderId": best["order_id"],
                "predictedEtaMin": round(float(best.get("predicted_eta_min", 0.0)), 2),
            }
        )

        current_node = str(best["node_id"])
        unvisited.remove(best)
        stop_no += 1

    return_leg = matrix_cost(distance_matrix, current_node, "DEPOT")
    total_distance = cumulative_distance + return_leg
    travel_minutes = (total_distance / speed_kmph) * 60.0 if speed_kmph > 0 else 0.0
    operational_minutes = travel_minutes + (len(rows) * float(service_min))
    return route, {
        "distance_km": total_distance,
        "travel_minutes": travel_minutes,
        "operational_minutes": operational_minutes,
    }


def append_added_customers_to_assign_df(
    assign_df: pd.DataFrame,
    customers: List[AddedCustomerPayload],
) -> pd.DataFrame:
    work = ensure_preview_node_ids(assign_df.copy())

    if work.empty:
        raise HTTPException(status_code=400, detail="Baseline preview is empty.")

    depot_lat = float(work.iloc[0]["depot_lat"])
    depot_lon = float(work.iloc[0]["depot_lon"])
    depot_id = str(work.iloc[0]["depot_id"])

    rows: List[Dict[str, Any]] = []

    for idx, customer in enumerate(customers, start=1):
        customer_number = customer.customer_number or (100000 + idx)
        customer_name = f"Customer {customer_number}"
        customer_node_id = f"ADDED-NODE-{uuid.uuid4().hex[:10].upper()}"
        order_id = f"ADDED-ORDER-{uuid.uuid4().hex[:10].upper()}"

        direct_km = haversine_km(
            depot_lat,
            depot_lon,
            float(customer.lat),
            float(customer.lon),
        )

        rows.append(
            {
                "depot_id": depot_id,
                "depot_lat": depot_lat,
                "depot_lon": depot_lon,
                "customer_node_id": customer_node_id,
                "node_id": customer_node_id,
                "customer_id": f"ADDED-CUST-{uuid.uuid4().hex[:10].upper()}",
                "order_id": order_id,
                "order_date": pd.NaT,
                "agent_id": customer.assigned_rep or "UNASSIGNED",
                "customer_name": customer_name,
                "node_name": customer_name,
                "customer_lat": float(customer.lat),
                "customer_lon": float(customer.lon),
                "observed_eta_min": 8.0 + (direct_km / 18.0) * 60.0,
                "predicted_eta_min": 8.0 + (direct_km / 18.0) * 60.0,
                "rating": 4.0,
                "area": "ADDED_CUSTOMER",
                "node_order_count": 1,
                "direct_depot_customer_km": direct_km,
                "rep_id": customer.assigned_rep or "UNASSIGNED",
            }
        )

    added_df = pd.DataFrame(rows)
    combined = pd.concat([work, added_df], ignore_index=True)
    return combined.reset_index(drop=True)


def route_all(
    assign_df: pd.DataFrame,
    speed_kmph: float,
    service_min: float,
    name: str,
    distance_matrix: Dict[str, Dict[str, float]],
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, float]]:
    work = ensure_preview_node_ids(assign_df)
    routes = []
    rep_rows = []

    palette = [
        "#2563eb",
        "#16a34a",
        "#dc2626",
        "#ca8a04",
        "#9333ea",
        "#0891b2",
        "#db2777",
        "#4f46e5",
    ]

    for idx, (rep_id, grp) in enumerate(work.groupby("rep_id"), start=1):
        ordered_stops, stats = route_one_rep(
            grp, speed_kmph, service_min, distance_matrix
        )
        color = palette[(idx - 1) % len(palette)]

        routes.append(
            {
                "id": f"{name}-{rep_id}",
                "representativeId": rep_id,
                "representativeName": rep_id,
                "color": color,
                "stops": ordered_stops,
            }
        )

        rep_rows.append(
            {
                "rep_id": rep_id,
                "customers": int(len(grp)),
                "workload_min": float(stats["operational_minutes"]),
                "distance_km": float(stats["distance_km"]),
                "travel_minutes": float(stats["travel_minutes"]),
                "operational_minutes": float(stats["operational_minutes"]),
                "centroid_lat": float(grp["customer_lat"].mean()),
                "centroid_lon": float(grp["customer_lon"].mean()),
            }
        )

    rep_df = pd.DataFrame(rep_rows)
    total = {
        "distance_km": float(rep_df["distance_km"].sum()) if not rep_df.empty else 0.0,
        "travel_minutes": float(rep_df["travel_minutes"].sum())
        if not rep_df.empty
        else 0.0,
        "operational_minutes": float(rep_df["operational_minutes"].sum())
        if not rep_df.empty
        else 0.0,
    }
    return routes, rep_df, total


def compute_thesis_priority_scores(
    assign_df: pd.DataFrame,
    rep_df: pd.DataFrame,
    alpha: float = 0.60,
    beta: float = 0.40,
) -> pd.DataFrame:
    """
    Thesis priority score:
        PS = alpha * (Delta T) + beta * (1 - Rating)

    Lower PS = higher queue priority.
    """
    if rep_df.empty:
        return rep_df.copy()

    work = assign_df.copy()
    reps = rep_df.copy()

    # Representative rating = mean assigned-customer rating
    rep_rating = (
        work.groupby("rep_id")["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "avg_rating"})
    )

    reps = reps.merge(rep_rating, on="rep_id", how="left")
    reps["avg_rating"] = pd.to_numeric(reps["avg_rating"], errors="coerce").fillna(1.0)

    # Delta T from operational minutes, normalized to [0,1]
    op = pd.to_numeric(reps["operational_minutes"], errors="coerce").fillna(0.0)
    op_min = float(op.min())
    op_max = float(op.max())

    if op_max > op_min:
        reps["delta_t"] = (op - op_min) / (op_max - op_min)
    else:
        reps["delta_t"] = 0.0

    # Ratings assumed already on 0..1 scale in your thesis examples
    # If your actual ratings are 1..5, normalize first:
    if reps["avg_rating"].max() > 1.0:
        reps["avg_rating_norm"] = reps["avg_rating"] / 5.0
    else:
        reps["avg_rating_norm"] = reps["avg_rating"]

    reps["priority_score"] = alpha * reps["delta_t"] + beta * (
        1.0 - reps["avg_rating_norm"]
    )

    reps = reps.sort_values("priority_score", ascending=True).reset_index(drop=True)
    reps["queue_position"] = np.arange(1, len(reps) + 1)

    return reps


def jains_fairness(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    if len(arr) == 0 or np.allclose(arr.sum(), 0):
        return 1.0
    return float((arr.sum() ** 2) / (len(arr) * np.square(arr).sum()))


def workload_balance_index(values: List[float]) -> float:
    """
        WBI = sigma(W) / mu(W)

    Lower is better.
    Example display can be percentage: WBI * 100
    """
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0.0
    mean = arr.mean()
    if mean == 0:
        return 0.0
    return float(arr.std(ddof=0) / mean)


def rep_cards(
    rep_df: pd.DataFrame, assign_df: Optional[pd.DataFrame] = None
) -> List[Dict[str, Any]]:
    if rep_df.empty:
        return []

    scored = rep_df.copy()

    if assign_df is not None and not assign_df.empty:
        scored = compute_thesis_priority_scores(
            assign_df, scored, alpha=0.60, beta=0.40
        )
    else:
        scored["priority_score"] = 0.0
        scored["queue_position"] = np.arange(1, len(scored) + 1)
        scored["avg_rating_norm"] = 1.0

    max_workload = max(float(scored["workload_min"].max()), 1.0)

    out = []
    for _, row in scored.iterrows():
        out.append(
            {
                "id": row["rep_id"],
                "name": row["rep_id"],
                "workload": round(float(row["workload_min"]), 2),
                "opportunityScore": round(
                    max(
                        0.0, 100.0 - (float(row["workload_min"]) / max_workload) * 100.0
                    ),
                    1,
                ),
                "priorityScore": round(float(row["priority_score"]), 3),
                "queuePosition": int(row["queue_position"]),
                "assignedCustomers": int(row["customers"]),
                "totalDistance": round(float(row["distance_km"]), 2),
                "totalTime": round(float(row["operational_minutes"]), 2),
            }
        )
    return out


def kpis_from_totals(
    total: Dict[str, float], rep_df: pd.DataFrame, dataset_size: int
) -> Dict[str, Any]:
    fairness = jains_fairness(rep_df["workload_min"].tolist())
    wbi = workload_balance_index(rep_df["workload_min"].tolist())

    n_routes = max(1, len(rep_df))

    total_distance_km = float(total["distance_km"])
    total_travel_hr = float(total["travel_minutes"]) / 60.0
    total_operational_hr = float(total["operational_minutes"]) / 60.0

    avg_total_distance = total_distance_km / n_routes
    avg_travel_time = total_travel_hr / n_routes

    assigned_customers = int(rep_df["customers"].sum()) if not rep_df.empty else 0
    coverage_ratio = (assigned_customers / max(1, dataset_size)) * 100.0

    return {
        "totalDistance": round(total_distance_km, 2),
        "travelTime": round(total_travel_hr, 2),
        "operationalTime": round(total_operational_hr, 2),
        "computeTime": round(max(0.5, dataset_size / 80.0), 2),
        "fairness": round(fairness, 6),
        "workloadBalance": round(wbi * 100.0, 2),
        "coverage": round(coverage_ratio, 2),
        "scalability": round(dataset_size / n_routes, 2),
        # new compare-specific fields
        "avgTotalDistance": round(avg_total_distance, 2),
        "avgTravelTime": round(avg_travel_time, 2),
        "coverageRatio": round(coverage_ratio, 2),
        # compatibility fields
        "totalTime": round(total_operational_hr, 2),
        "numberOfStops": assigned_customers,
        "delayScore": 0.0,
        "ratingPenalty": 0.0,
        "workloadBalanceIndex": round(wbi * 100.0, 2),
        "jainsFairnessIndex": round(fairness, 6),
    }


def make_algorithm_run(
    name: str,
    routes: List[Dict[str, Any]],
    rep_df: pd.DataFrame,
    total: Dict[str, float],
    dataset_size: int,
    metrics: Dict[str, float],
    notes: Optional[List[str]] = None,
    assign_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "algorithm": name,
        "routes": routes,
        "representatives": rep_cards(rep_df, assign_df),
        "kpis": kpis_from_totals(total, rep_df, dataset_size),
        "trainingMetrics": metrics,
        "notes": notes or [],
    }


def border_candidates(
    assign_df: pd.DataFrame, heavy_rep: str, light_rep: str, fraction: float
) -> List[int]:
    heavy = assign_df[assign_df["rep_id"] == heavy_rep].copy()
    light = assign_df[assign_df["rep_id"] == light_rep].copy()

    if heavy.empty or light.empty:
        return []

    target_lat = light["customer_lat"].mean()
    target_lon = light["customer_lon"].mean()

    heavy["to_target"] = heavy.apply(
        lambda r: haversine_km(
            r["customer_lat"], r["customer_lon"], target_lat, target_lon
        ),
        axis=1,
    )
    heavy = heavy.sort_values(["to_target", "predicted_eta_min"]).reset_index()
    take = max(1, int(math.ceil(len(heavy) * fraction)))
    return heavy.head(take)["index"].astype(int).tolist()


def swap_candidates(
    assign_df: pd.DataFrame,
    heavy_rep: str,
    light_rep: str,
    fraction: float,
) -> Tuple[List[int], List[int]]:
    """
    Candidate rows for one-for-one swap search.
    We take a border subset from both reps, guided by proximity toward the other rep.
    """
    heavy = assign_df[assign_df["rep_id"] == heavy_rep].copy()
    light = assign_df[assign_df["rep_id"] == light_rep].copy()

    if heavy.empty or light.empty:
        return [], []

    heavy_target_lat = light["customer_lat"].mean()
    heavy_target_lon = light["customer_lon"].mean()

    light_target_lat = heavy["customer_lat"].mean()
    light_target_lon = heavy["customer_lon"].mean()

    heavy["to_target"] = heavy.apply(
        lambda r: haversine_km(
            r["customer_lat"], r["customer_lon"], heavy_target_lat, heavy_target_lon
        ),
        axis=1,
    )
    light["to_target"] = light.apply(
        lambda r: haversine_km(
            r["customer_lat"], r["customer_lon"], light_target_lat, light_target_lon
        ),
        axis=1,
    )

    heavy = heavy.sort_values(["to_target", "predicted_eta_min"]).reset_index()
    light = light.sort_values(["to_target", "predicted_eta_min"]).reset_index()

    heavy_take = max(1, int(math.ceil(len(heavy) * fraction)))
    light_take = max(1, int(math.ceil(len(light) * fraction)))

    heavy_idx = heavy.head(heavy_take)["index"].astype(int).tolist()
    light_idx = light.head(light_take)["index"].astype(int).tolist()

    return heavy_idx, light_idx


def evaluate_assignment(
    assign_df: pd.DataFrame,
    speed_kmph: float,
    service_min: float,
    distance_matrix: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    routes, rep_df, total = route_all(
        assign_df, speed_kmph, service_min, "eval", distance_matrix
    )
    fairness = (
        jains_fairness(rep_df["workload_min"].tolist()) if not rep_df.empty else 1.0
    )
    wbi = (
        workload_balance_index(rep_df["workload_min"].tolist())
        if not rep_df.empty
        else 0.0
    )

    assigned_customers = int(rep_df["customers"].sum()) if not rep_df.empty else 0
    coverage_ratio = assigned_customers / max(1, len(assign_df))
    return {
        "routes": routes,
        "rep_df": rep_df,
        "total": total,
        "fairness": fairness,
        "wbi": wbi,
        "coverage_ratio": coverage_ratio,
    }


def objective_value(
    wbi: float,
    total_distance_km: float,
    operational_minutes: float,
    fairness_weight: float,
    distance_weight: float,
    time_weight: float,
) -> float:
    # Lower is better
    return (
        fairness_weight * (wbi * 100.0)
        + distance_weight * total_distance_km
        + time_weight * (operational_minutes / 10.0)
    )


def enhance_assignment(
    assign_df: pd.DataFrame,
    speed_kmph: float,
    service_min: float,
    alpha_weight: float,
    beta_weight: float,
    fairness_weight: float,
    distance_weight: float,
    time_weight: float,
    max_iterations: int,
    border_fraction: float,
    distance_matrix: Dict[str, Dict[str, float]],
    is_zomato_mode: bool = False,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    current = ensure_preview_node_ids(assign_df.copy())
    logs: List[Dict[str, Any]] = []
    current_eval = evaluate_assignment(
        current, speed_kmph, service_min, distance_matrix
    )

    for iteration in range(1, max_iterations + 1):
        rep_perf = compute_thesis_priority_scores(
            current,
            current_eval["rep_df"],
            alpha=alpha_weight,
            beta=beta_weight,
        ).reset_index(drop=True)

        if len(rep_perf) < 2:
            break

        overall_gap = float(
            rep_perf.iloc[-1]["operational_minutes"]
            - rep_perf.iloc[0]["operational_minutes"]
        )
        if overall_gap < 5.0:
            break

        current_score = objective_value(
            current_eval["wbi"],
            current_eval["total"]["distance_km"],
            current_eval["total"]["operational_minutes"],
            fairness_weight,
            distance_weight,
            time_weight,
        )

        n_reps = len(rep_perf)
        heavy_count = min(3, n_reps - 1)
        light_count = min(3, n_reps - 1)

        light_ids = [
            str(rep_perf.iloc[i]["rep_id"]) for i in range(min(light_count, n_reps))
        ]
        heavy_ids = [
            str(rep_perf.iloc[n_reps - 1 - j]["rep_id"])
            for j in range(min(heavy_count, n_reps))
        ]

        # ---------------------------
        # 1) TRANSFER SEARCH
        # ---------------------------
        best_trial = None
        best_trial_eval = None
        best_log = None
        best_score_gain = 0.0

        tried_pairs = set()

        for heavy_rep in heavy_ids:
            for light_rep in light_ids:
                if heavy_rep == light_rep:
                    continue
                if (heavy_rep, light_rep) in tried_pairs:
                    continue
                tried_pairs.add((heavy_rep, light_rep))

                heavy_minutes = float(
                    rep_perf.loc[
                        rep_perf["rep_id"] == heavy_rep, "operational_minutes"
                    ].iloc[0]
                )
                light_minutes = float(
                    rep_perf.loc[
                        rep_perf["rep_id"] == light_rep, "operational_minutes"
                    ].iloc[0]
                )

                workload_gap = heavy_minutes - light_minutes
                if workload_gap < 5.0:
                    continue

                candidates = border_candidates(
                    current, heavy_rep, light_rep, border_fraction
                )

                for idx in candidates:
                    # do not allow a move that would empty the source rep
                    current_source_count = int((current["rep_id"] == heavy_rep).sum())
                    if current_source_count <= 1:
                        continue

                    trial = current.copy()
                    trial.loc[idx, "rep_id"] = light_rep

                    source_rep_remaining = int((trial["rep_id"] == heavy_rep).sum())
                    if source_rep_remaining == 0:
                        continue

                    trial_eval = evaluate_assignment(
                        trial, speed_kmph, service_min, distance_matrix
                    )

                    trial_score = objective_value(
                        trial_eval["wbi"],
                        trial_eval["total"]["distance_km"],
                        trial_eval["total"]["operational_minutes"],
                        fairness_weight,
                        distance_weight,
                        time_weight,
                    )

                    fairness_gain = trial_eval["fairness"] - current_eval["fairness"]
                    distance_gain = (
                        current_eval["total"]["distance_km"]
                        - trial_eval["total"]["distance_km"]
                    )
                    time_gain = (
                        current_eval["total"]["operational_minutes"]
                        - trial_eval["total"]["operational_minutes"]
                    )
                    score_gain = current_score - trial_score

                    EPS = 1e-6

                    if fairness_gain <= 0:
                        continue

                    if is_zomato_mode:
                        # Slightly looser for Zomato:
                        # allow small distance/time worsening if fairness gain is meaningful
                        if distance_gain < -12.0:
                            continue
                        if distance_gain < -4.0:
                            continue
                        if fairness_gain < 0.01 and (
                            distance_gain < -EPS or time_gain < -EPS
                        ):
                            continue
                    else:
                        # Keep Amazon strict
                        if distance_gain < -EPS:
                            continue
                        if time_gain < -EPS:
                            continue

                    if score_gain > best_score_gain:
                        best_score_gain = score_gain
                        best_trial = trial
                        best_trial_eval = trial_eval
                        best_log = {
                            "iteration": iteration,
                            "move_type": "transfer",
                            "moved_order": str(current.loc[idx, "order_id"]),
                            "from_rep": heavy_rep,
                            "to_rep": light_rep,
                            "fairness_before": round(current_eval["fairness"], 6),
                            "fairness_after": round(trial_eval["fairness"], 6),
                            "distance_before": round(
                                current_eval["total"]["distance_km"], 2
                            ),
                            "distance_after": round(
                                trial_eval["total"]["distance_km"], 2
                            ),
                            "operational_before": round(
                                current_eval["total"]["operational_minutes"], 2
                            ),
                            "operational_after": round(
                                trial_eval["total"]["operational_minutes"], 2
                            ),
                            "score_before": round(current_score, 4),
                            "score_after": round(trial_score, 4),
                            "score_gain": round(score_gain, 4),
                            "fairness_gain": round(fairness_gain, 6),
                            "distance_gain": round(distance_gain, 4),
                            "time_gain": round(time_gain, 4),
                            "accepted": score_gain > 0,
                        }

        if best_trial is not None and best_score_gain > 0:
            current = best_trial
            current_eval = best_trial_eval
            logs.append(best_log)
            continue

        # ---------------------------
        # 2) SWAP SEARCH
        # ---------------------------
        swap_best_trial = None
        swap_best_trial_eval = None
        swap_best_log = None
        swap_best_score_gain = 0.0

        tried_pairs = set()

        for heavy_rep in heavy_ids:
            for light_rep in light_ids:
                if heavy_rep == light_rep:
                    continue
                if (heavy_rep, light_rep) in tried_pairs:
                    continue
                tried_pairs.add((heavy_rep, light_rep))

                heavy_minutes = float(
                    rep_perf.loc[
                        rep_perf["rep_id"] == heavy_rep, "operational_minutes"
                    ].iloc[0]
                )
                light_minutes = float(
                    rep_perf.loc[
                        rep_perf["rep_id"] == light_rep, "operational_minutes"
                    ].iloc[0]
                )

                workload_gap = heavy_minutes - light_minutes
                if workload_gap < 5.0:
                    continue

                heavy_idx_list, light_idx_list = swap_candidates(
                    current,
                    heavy_rep,
                    light_rep,
                    border_fraction,
                )

                for idx_h in heavy_idx_list:
                    for idx_l in light_idx_list:
                        if idx_h == idx_l:
                            continue

                        trial = current.copy()

                        rep_h = str(trial.loc[idx_h, "rep_id"])
                        rep_l = str(trial.loc[idx_l, "rep_id"])

                        if rep_h == rep_l:
                            continue

                        trial.loc[idx_h, "rep_id"] = rep_l
                        trial.loc[idx_l, "rep_id"] = rep_h

                        trial_eval = evaluate_assignment(
                            trial,
                            speed_kmph,
                            service_min,
                            distance_matrix,
                        )

                        trial_score = objective_value(
                            trial_eval["wbi"],
                            trial_eval["total"]["distance_km"],
                            trial_eval["total"]["operational_minutes"],
                            alpha_weight,
                            beta_weight,
                            time_weight,
                        )

                        fairness_gain = (
                            trial_eval["fairness"] - current_eval["fairness"]
                        )
                        distance_gain = (
                            current_eval["total"]["distance_km"]
                            - trial_eval["total"]["distance_km"]
                        )
                        time_gain = (
                            current_eval["total"]["operational_minutes"]
                            - trial_eval["total"]["operational_minutes"]
                        )
                        score_gain = current_score - trial_score

                    EPS = 1e-6

                    if fairness_gain <= 0:
                        continue

                    if is_zomato_mode:
                        # Slightly looser for Zomato:
                        # allow small distance/time worsening if fairness gain is meaningful
                        if distance_gain < -12.0:
                            continue
                        if distance_gain < -4.0:
                            continue
                        if fairness_gain < 0.01 and (
                            distance_gain < -EPS or time_gain < -EPS
                        ):
                            continue
                    else:
                        # Keep Amazon strict
                        if distance_gain < -EPS:
                            continue
                        if time_gain < -EPS:
                            continue

                        if score_gain > swap_best_score_gain:
                            swap_best_score_gain = score_gain
                            swap_best_trial = trial
                            swap_best_trial_eval = trial_eval
                            swap_best_log = {
                                "iteration": iteration,
                                "move_type": "swap",
                                "moved_order": str(current.loc[idx_h, "order_id"]),
                                "swapped_with_order": str(
                                    current.loc[idx_l, "order_id"]
                                ),
                                "from_rep": heavy_rep,
                                "to_rep": light_rep,
                                "fairness_before": round(current_eval["fairness"], 6),
                                "fairness_after": round(trial_eval["fairness"], 6),
                                "distance_before": round(
                                    current_eval["total"]["distance_km"], 2
                                ),
                                "distance_after": round(
                                    trial_eval["total"]["distance_km"], 2
                                ),
                                "operational_before": round(
                                    current_eval["total"]["operational_minutes"], 2
                                ),
                                "operational_after": round(
                                    trial_eval["total"]["operational_minutes"], 2
                                ),
                                "score_before": round(current_score, 4),
                                "score_after": round(trial_score, 4),
                                "score_gain": round(score_gain, 4),
                                "fairness_gain": round(fairness_gain, 6),
                                "distance_gain": round(distance_gain, 4),
                                "time_gain": round(time_gain, 4),
                                "accepted": score_gain > 0,
                            }

        if swap_best_trial is not None and swap_best_score_gain > 0:
            current = swap_best_trial
            current_eval = swap_best_trial_eval
            logs.append(swap_best_log)
            continue

        # ---------------------------
        # 3) STOP
        # ---------------------------
        logs.append(
            {
                "iteration": iteration,
                "from_rep": heavy_ids[0] if heavy_ids else "",
                "to_rep": light_ids[0] if light_ids else "",
                "accepted": False,
                "reason": "no improving transfer or swap found",
            }
        )
        break

    print(
        "rep priority/workload snapshot:",
        rep_perf[
            ["rep_id", "priority_score", "operational_minutes", "customers"]
        ].to_dict("records"),
    )
    print("overall_gap minutes:", round(overall_gap, 2))

    print("accepted moves:", sum(1 for x in logs if x.get("accepted")))
    print("enhancement logs:", logs)

    return current, logs


def amazon_distance_polish_assignment(
    assign_df: pd.DataFrame,
    speed_kmph: float,
    service_min: float,
    distance_matrix: Dict[str, Dict[str, float]],
    max_iterations: int = 12,
    min_distance_gain_km: float = 0.25,
    min_fairness_floor: float = 0.995,
    max_wbi_increase: float = 0.0,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Amazon-only final improvement pass.

    The normal DEQ pass prioritizes fairness first. For Amazon, fairness often reaches
    ~1.00 quickly, so strict fairness-gain rules can reject distance-improving moves.
    This pass only runs for the Amazon role and accepts a move when it reduces
    distance/time while keeping fairness very high and workload balance within a
    small tolerance.
    """
    current = ensure_preview_node_ids(assign_df.copy()).reset_index(drop=True)
    logs: List[Dict[str, Any]] = []
    current_eval = evaluate_assignment(
        current, speed_kmph, service_min, distance_matrix
    )

    for iteration in range(1, max_iterations + 1):
        rep_ids = [str(x) for x in current["rep_id"].dropna().unique().tolist()]
        if len(rep_ids) < 2:
            break

        best_trial = None
        best_eval = None
        best_log = None
        best_score = 0.0

        # 1) Try distance-improving transfers.
        for idx, row in current.iterrows():
            source_rep = str(row["rep_id"])
            if int((current["rep_id"] == source_rep).sum()) <= 2:
                continue

            for target_rep in rep_ids:
                if target_rep == source_rep:
                    continue

                trial = current.copy()
                trial.loc[idx, "rep_id"] = target_rep
                trial_eval = evaluate_assignment(
                    trial, speed_kmph, service_min, distance_matrix
                )

                distance_gain = (
                    current_eval["total"]["distance_km"]
                    - trial_eval["total"]["distance_km"]
                )
                time_gain = (
                    current_eval["total"]["operational_minutes"]
                    - trial_eval["total"]["operational_minutes"]
                )
                wbi_increase = trial_eval["wbi"] - current_eval["wbi"]

                if distance_gain < min_distance_gain_km:
                    continue
                if time_gain < -1e-6:
                    continue
                if trial_eval["fairness"] < min_fairness_floor:
                    continue
                # WBI is sigma/mu, so lower is better. Do not accept an Amazon polish
                # move that makes WBI worse, even if distance improves.
                if wbi_increase > max_wbi_increase + 1e-9:
                    continue

                wbi_gain = current_eval["wbi"] - trial_eval["wbi"]
                score = distance_gain + (time_gain / 60.0) + max(0.0, wbi_gain * 20.0)
                if score > best_score:
                    best_score = score
                    best_trial = trial
                    best_eval = trial_eval
                    best_log = {
                        "iteration": iteration,
                        "move_type": "amazon_distance_transfer",
                        "moved_order": str(current.loc[idx, "order_id"]),
                        "from_rep": source_rep,
                        "to_rep": target_rep,
                        "fairness_before": round(current_eval["fairness"], 6),
                        "fairness_after": round(trial_eval["fairness"], 6),
                        "distance_before": round(
                            current_eval["total"]["distance_km"], 2
                        ),
                        "distance_after": round(trial_eval["total"]["distance_km"], 2),
                        "operational_before": round(
                            current_eval["total"]["operational_minutes"], 2
                        ),
                        "operational_after": round(
                            trial_eval["total"]["operational_minutes"], 2
                        ),
                        "distance_gain": round(distance_gain, 4),
                        "time_gain": round(time_gain, 4),
                        "wbi_before_pct": round(current_eval["wbi"] * 100.0, 2),
                        "wbi_after_pct": round(trial_eval["wbi"] * 100.0, 2),
                        "wbi_gain_pct": round(
                            (current_eval["wbi"] - trial_eval["wbi"]) * 100.0, 2
                        ),
                        "accepted": True,
                    }

        # 2) Try swaps too, because swaps usually preserve workload balance better.
        for idx_a in range(len(current)):
            rep_a = str(current.loc[idx_a, "rep_id"])
            for idx_b in range(idx_a + 1, len(current)):
                rep_b = str(current.loc[idx_b, "rep_id"])
                if rep_a == rep_b:
                    continue

                trial = current.copy()
                trial.loc[idx_a, "rep_id"] = rep_b
                trial.loc[idx_b, "rep_id"] = rep_a
                trial_eval = evaluate_assignment(
                    trial, speed_kmph, service_min, distance_matrix
                )

                distance_gain = (
                    current_eval["total"]["distance_km"]
                    - trial_eval["total"]["distance_km"]
                )
                time_gain = (
                    current_eval["total"]["operational_minutes"]
                    - trial_eval["total"]["operational_minutes"]
                )
                wbi_increase = trial_eval["wbi"] - current_eval["wbi"]

                if distance_gain < min_distance_gain_km:
                    continue
                if time_gain < -1e-6:
                    continue
                if trial_eval["fairness"] < min_fairness_floor:
                    continue
                # WBI is sigma/mu, so lower is better. Do not accept an Amazon polish
                # move that makes WBI worse, even if distance improves.
                if wbi_increase > max_wbi_increase + 1e-9:
                    continue

                wbi_gain = current_eval["wbi"] - trial_eval["wbi"]
                score = distance_gain + (time_gain / 60.0) + max(0.0, wbi_gain * 20.0)
                if score > best_score:
                    best_score = score
                    best_trial = trial
                    best_eval = trial_eval
                    best_log = {
                        "iteration": iteration,
                        "move_type": "amazon_distance_swap",
                        "moved_order": str(current.loc[idx_a, "order_id"]),
                        "swapped_with_order": str(current.loc[idx_b, "order_id"]),
                        "from_rep": rep_a,
                        "to_rep": rep_b,
                        "fairness_before": round(current_eval["fairness"], 6),
                        "fairness_after": round(trial_eval["fairness"], 6),
                        "distance_before": round(
                            current_eval["total"]["distance_km"], 2
                        ),
                        "distance_after": round(trial_eval["total"]["distance_km"], 2),
                        "operational_before": round(
                            current_eval["total"]["operational_minutes"], 2
                        ),
                        "operational_after": round(
                            trial_eval["total"]["operational_minutes"], 2
                        ),
                        "distance_gain": round(distance_gain, 4),
                        "time_gain": round(time_gain, 4),
                        "wbi_before_pct": round(current_eval["wbi"] * 100.0, 2),
                        "wbi_after_pct": round(trial_eval["wbi"] * 100.0, 2),
                        "wbi_gain_pct": round(
                            (current_eval["wbi"] - trial_eval["wbi"]) * 100.0, 2
                        ),
                        "accepted": True,
                    }

        if best_trial is None or best_eval is None:
            break

        current = best_trial.reset_index(drop=True)
        current_eval = best_eval
        logs.append(best_log)

    if logs:
        print("amazon distance polish accepted moves:", len(logs))
        print("amazon distance polish logs:", logs)

    return current, logs


def select_spatially_spread_rows(
    df: pd.DataFrame,
    target_n: int,
    depot_lat: float,
    depot_lon: float,
) -> pd.DataFrame:
    if df.empty or target_n <= 0:
        return df.head(0).copy()

    pool = df.copy()

    # Start with the nearest point to depot
    pool["to_depot_km"] = pool.apply(
        lambda r: haversine_km(
            r["customer_lat"], r["customer_lon"], depot_lat, depot_lon
        ),
        axis=1,
    )
    pool = pool.sort_values("to_depot_km").reset_index(drop=True)

    selected_idx = [0]
    remaining = set(range(1, len(pool)))

    while len(selected_idx) < min(target_n, len(pool)) and remaining:
        best_i = None
        best_score = -1.0

        for i in remaining:
            row = pool.iloc[i]
            min_dist_to_selected = min(
                haversine_km(
                    row["customer_lat"],
                    row["customer_lon"],
                    pool.iloc[j]["customer_lat"],
                    pool.iloc[j]["customer_lon"],
                )
                for j in selected_idx
            )

            # prefer points that are still reasonably near depot,
            # but also far from already selected points
            depot_dist = row["to_depot_km"]
            score = min_dist_to_selected - (0.15 * depot_dist)

            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None:
            break

        selected_idx.append(best_i)
        remaining.remove(best_i)

    out = pool.iloc[selected_idx].copy()
    out = out.drop(columns=["to_depot_km"], errors="ignore")
    return out


def assign_preview_rep_ids_from_agent(
    preview_df: pd.DataFrame,
    num_representatives: int,
    max_total_stops: Optional[int] = None,
    strict_existing_agents: bool = False,
    cap_total_stops: bool = True,
) -> pd.DataFrame:
    if preview_df.empty:
        return preview_df.copy()

    work = preview_df.copy().reset_index(drop=True)

    if "agent_id" not in work.columns:
        return assign_preview_rep_ids_uneven(work, num_representatives)

    work["agent_id"] = work["agent_id"].astype(str).fillna("UNKNOWN")
    work["agent_id"] = work["agent_id"].replace(
        {"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"}
    )

    valid = work[work["agent_id"] != "UNKNOWN"].copy()
    if valid.empty:
        return assign_preview_rep_ids_uneven(work, num_representatives)

    agent_counts = valid.groupby("agent_id").size().sort_values(ascending=False)

    unique_agents = agent_counts.index.tolist()
    if strict_existing_agents:
        top_agents = unique_agents
    else:
        if len(unique_agents) < num_representatives:
            print(
                f"existing agent-based preview has only {len(unique_agents)} unique agents "
                f"for {num_representatives} requested reps."
            )
            return assign_preview_rep_ids_uneven(work, num_representatives)
        top_agents = unique_agents[:num_representatives]

    filtered = valid[valid["agent_id"].isin(top_agents)].copy()

    if len(filtered) < num_representatives:
        print(
            f"existing-agent filtered rows too small ({len(filtered)} rows) for "
            f"{num_representatives} requested reps."
        )
        if strict_existing_agents:
            filtered["rep_id"] = filtered["agent_id"].astype(str)
            return filtered.drop(columns=["to_depot_km"], errors="ignore").reset_index(
                drop=True
            )
        return assign_preview_rep_ids_uneven(work, num_representatives)

    filtered["to_depot_km"] = filtered.apply(
        lambda r: haversine_km(
            float(r["depot_lat"]),
            float(r["depot_lon"]),
            float(r["customer_lat"]),
            float(r["customer_lon"]),
        ),
        axis=1,
    )

    filtered = filtered.sort_values(["agent_id", "to_depot_km"]).copy()

    # Zomato/default preview can still cap rows. Amazon calls this with
    # cap_total_stops=False so all selected local order rows survive.
    if (
        cap_total_stops
        and max_total_stops is not None
        and len(filtered) > max_total_stops
    ):
        counts = filtered["agent_id"].value_counts()
        total = counts.sum()

        keep_rows = []
        allocations = {}
        for agent_id, cnt in counts.items():
            share = max(1, int(round((cnt / total) * max_total_stops)))
            allocations[agent_id] = min(cnt, share)

        allocated_total = sum(allocations.values())

        while allocated_total > max_total_stops:
            for agent_id in sorted(allocations, key=allocations.get, reverse=True):
                if allocations[agent_id] > 1 and allocated_total > max_total_stops:
                    allocations[agent_id] -= 1
                    allocated_total -= 1

        while allocated_total < max_total_stops:
            for agent_id, cnt in counts.items():
                if allocations[agent_id] < cnt and allocated_total < max_total_stops:
                    allocations[agent_id] += 1
                    allocated_total += 1

        for agent_id in counts.index:
            grp = filtered[filtered["agent_id"] == agent_id].copy()
            keep_rows.append(grp.head(allocations[agent_id]))

        filtered = pd.concat(keep_rows, ignore_index=True)

    filtered["rep_id"] = filtered["agent_id"].astype(str)
    return filtered.drop(columns=["to_depot_km"], errors="ignore").reset_index(
        drop=True
    )


def assign_preview_rep_ids_uneven(
    preview_df: pd.DataFrame,
    num_representatives: int,
) -> pd.DataFrame:
    if preview_df.empty:
        return preview_df.copy()

    work = preview_df.copy().reset_index(drop=True)

    depot_lat = float(work["depot_lat"].iloc[0])
    depot_lon = float(work["depot_lon"].iloc[0])

    # Angle of each customer relative to depot
    work["angle"] = work.apply(
        lambda r: math.atan2(
            float(r["customer_lat"]) - depot_lat,
            float(r["customer_lon"]) - depot_lon,
        ),
        axis=1,
    )

    # Secondary sort by distance to keep each sector internally coherent
    work["to_depot_km"] = work.apply(
        lambda r: haversine_km(
            depot_lat,
            depot_lon,
            float(r["customer_lat"]),
            float(r["customer_lon"]),
        ),
        axis=1,
    )

    work = work.sort_values(["angle", "to_depot_km"]).reset_index(drop=True)

    rep_ids = [f"REP-{i}" for i in range(1, num_representatives + 1)]
    n = len(work)

    # Slightly uneven but still spatially grouped
    base = n // num_representatives
    rem = n % num_representatives

    sizes = [base] * num_representatives

    # Front-load only a little, not a hardcoded 4-3-3-2 row chunk pattern
    for i in range(rem):
        sizes[i] += 1

    # Optional slight imbalance for the first rep if possible
    # if num_representatives > 1 and n >= num_representatives * 2:
    #     for j in range(num_representatives - 1, 0, -1):
    #         if sizes[j] > 1:
    #             sizes[0] += 1
    #             sizes[j] -= 1
    #             break

    assigned = []
    for rep_id, size in zip(rep_ids, sizes):
        assigned.extend([rep_id] * size)

    work["rep_id"] = assigned[:n]

    return work.drop(columns=["angle", "to_depot_km"], errors="ignore")


def choose_best_local_depot_cluster(
    df: pd.DataFrame,
    candidate_pool_size: int = 12,
    prefer_agent_coverage: bool = False,
    min_agents: int = 1,
) -> Tuple[float, float, pd.DataFrame]:
    """
    Choose the depot whose nearby customer cluster is most suitable for preview.

    Default behavior:
    - prefers compact clusters

    When prefer_agent_coverage=True:
    - prefers more distinct agents
    - then higher total order demand
    - then more nearby nodes
    - then compactness
    """
    work = df.copy()

    depot_groups = (
        work.groupby(["depot_lat", "depot_lon"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
    )

    if depot_groups.empty:
        raise HTTPException(
            status_code=400, detail="No depot coordinates available for preview."
        )

    best_score = None
    best_depot_lat = None
    best_depot_lon = None
    best_cluster = None

    for depot in depot_groups.itertuples(index=False):
        depot_lat = float(depot.depot_lat)
        depot_lon = float(depot.depot_lon)

        cluster = work[
            (work["depot_lat"] == depot_lat) & (work["depot_lon"] == depot_lon)
        ].copy()

        if cluster.empty:
            continue

        cluster["to_depot_km"] = cluster.apply(
            lambda r: haversine_km(
                depot_lat,
                depot_lon,
                float(r["customer_lat"]),
                float(r["customer_lon"]),
            ),
            axis=1,
        )

        if "customer_node_id" in cluster.columns:
            cluster = (
                cluster.sort_values("to_depot_km")
                .drop_duplicates(subset=["customer_node_id"])
                .copy()
            )
        else:
            cluster["lat_round"] = cluster["customer_lat"].round(4)
            cluster["lon_round"] = cluster["customer_lon"].round(4)
            cluster = (
                cluster.sort_values("to_depot_km")
                .drop_duplicates(subset=["lat_round", "lon_round"])
                .copy()
            )
            cluster = cluster.drop(columns=["lat_round", "lon_round"], errors="ignore")

        if cluster.empty:
            continue

        nearest = cluster.nsmallest(candidate_pool_size, "to_depot_km").copy()

        distinct_agents = 0
        if "agent_id" in nearest.columns:
            agent_series = (
                nearest["agent_id"]
                .astype(str)
                .replace({"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"})
            )
            distinct_agents = int(agent_series[agent_series != "UNKNOWN"].nunique())

        total_orders = 0.0
        if "node_order_count" in cluster.columns:
            total_orders = float(
                pd.to_numeric(cluster["node_order_count"], errors="coerce")
                .fillna(1)
                .sum()
            )
        else:
            total_orders = float(len(cluster))

        nearby_orders = 0.0
        if "node_order_count" in nearest.columns:
            nearby_orders = float(
                pd.to_numeric(nearest["node_order_count"], errors="coerce")
                .fillna(1)
                .sum()
            )
        else:
            nearby_orders = float(len(nearest))

        nearby_nodes = int(len(nearest))
        mean_dist = float(nearest["to_depot_km"].mean())
        max_dist = float(nearest["to_depot_km"].max())

        if prefer_agent_coverage:
            insufficient_agent_penalty = 1 if distinct_agents < min_agents else 0
            score = (
                insufficient_agent_penalty,
                -distinct_agents,
                -total_orders,
                -nearby_orders,
                -nearby_nodes,
                mean_dist,
                max_dist,
            )
        else:
            score = (
                -nearby_nodes,
                -distinct_agents,
                -total_orders,
                mean_dist,
                max_dist,
            )

        print(
            "candidate depot:",
            {
                "depot_lat": depot_lat,
                "depot_lon": depot_lon,
                "distinct_agents": distinct_agents,
                "total_orders": round(total_orders, 2),
                "nearby_orders": round(nearby_orders, 2),
                "nearby_nodes": nearby_nodes,
                "mean_dist": round(mean_dist, 2),
                "max_dist": round(max_dist, 2),
            },
        )

        if best_score is None or score < best_score:
            best_score = score
            best_depot_lat = depot_lat
            best_depot_lon = depot_lon
            best_cluster = cluster.copy()

    if best_cluster is None:
        raise HTTPException(
            status_code=400, detail="Could not build a local depot preview cluster."
        )

    print(
        "selected depot cluster:",
        {
            "depot_lat": best_depot_lat,
            "depot_lon": best_depot_lon,
            "score": best_score,
        },
    )

    return best_depot_lat, best_depot_lon, best_cluster


def build_local_preview_subset(
    df: pd.DataFrame,
    num_representatives: int,
    max_total_stops: int = 12,
    initial_radius_km: float = 7.0,
    max_radius_km: float = 16.0,
    local_cap_km: float = 14.0,
    use_existing_agents: bool = False,
    strict_existing_agents: bool = False,
    min_nodes_per_rep: int = 3,
) -> pd.DataFrame:
    work = df.copy()

    if "order_date" in work.columns:
        work["order_date"] = parse_order_date_series(work["order_date"])
        valid_dates = sorted(work["order_date"].dropna().unique())

        if len(valid_dates) > 0:
            selected_dates = [valid_dates[-1]]
            dated = work[work["order_date"].isin(selected_dates)].copy()

            # Expand backward in time until we have enough candidate rows
            idx = len(valid_dates) - 2
            target_min_rows = max(max_total_stops, num_representatives * 2)

            while len(dated) < target_min_rows and idx >= 0:
                selected_dates.append(valid_dates[idx])
                dated = work[work["order_date"].isin(selected_dates)].copy()
                idx -= 1

            work = dated.copy()
            selected_dates_sorted = sorted(pd.to_datetime(selected_dates))
            print(
                "order_date window used for preview:",
                [d.strftime("%Y-%m-%d") for d in selected_dates_sorted],
            )

    if len(work) < num_representatives:
        print("date-filtered preview too small, falling back to all dates")
        work = df.copy()

    depot_lat, depot_lon, depot_cluster = choose_best_local_depot_cluster(
        work,
        candidate_pool_size=max(max_total_stops, num_representatives * 3),
        prefer_agent_coverage=use_existing_agents,
        min_agents=num_representatives,
    )

    print(f"chosen preview depot: ({depot_lat}, {depot_lon})")

    depot_cluster["to_depot_km"] = depot_cluster.apply(
        lambda r: haversine_km(
            depot_lat,
            depot_lon,
            float(r["customer_lat"]),
            float(r["customer_lon"]),
        ),
        axis=1,
    )

    radius = initial_radius_km
    local = depot_cluster[depot_cluster["to_depot_km"] <= radius].copy()

    while len(local) < max_total_stops and radius < max_radius_km:
        radius *= 1.5
        local = depot_cluster[depot_cluster["to_depot_km"] <= radius].copy()

    local = local[local["to_depot_km"] <= local_cap_km].copy()

    if use_existing_agents:
        # Keep a larger pool first so real agent coverage survives.
        if len(local) < max(max_total_stops * 2, num_representatives * 3):
            refill = depot_cluster.sort_values("to_depot_km").copy()
            refill = refill.head(
                max(max_total_stops * 3, num_representatives * 4)
            ).copy()
            local = refill.copy()
    else:
        target_local_nodes = max(
            max_total_stops, num_representatives * min_nodes_per_rep, 24
        )

        if len(local) >= target_local_nodes:
            local = local.head(target_local_nodes).copy()
        else:
            refill = depot_cluster.sort_values("to_depot_km").copy()
            refill = refill.head(target_local_nodes).copy()
            local = refill.copy()

    # If still too small, refill from the same chosen depot cluster only
    # If still too small, first refill from the same chosen depot cluster with a much bigger pool
    if len(local) < num_representatives:
        refill_target = max(max_total_stops * 4, num_representatives * 6)
        refill = depot_cluster.nsmallest(refill_target, "to_depot_km").copy()
        if "customer_node_id" in refill.columns:
            refill = (
                refill.sort_values("to_depot_km")
                .drop_duplicates(subset=["customer_node_id"])
                .copy()
            )
        local = refill.copy()

    # If still too small, refill from the same chosen depot cluster with a much bigger pool
    if len(local) < num_representatives:
        refill_target = max(max_total_stops * 4, num_representatives * 6)
        refill = depot_cluster.nsmallest(refill_target, "to_depot_km").copy()
        if "customer_node_id" in refill.columns:
            refill = (
                refill.sort_values("to_depot_km")
                .drop_duplicates(subset=["customer_node_id"])
                .copy()
            )
        local = refill.copy()

    # Absolute fallback: use all dates, but still only for the same chosen depot
    if len(local) < num_representatives:
        same_depot_all_dates = df[
            (df["depot_lat"] == depot_lat) & (df["depot_lon"] == depot_lon)
        ].copy()

        same_depot_all_dates["to_depot_km"] = same_depot_all_dates.apply(
            lambda r: haversine_km(
                depot_lat,
                depot_lon,
                float(r["customer_lat"]),
                float(r["customer_lon"]),
            ),
            axis=1,
        )

        if "customer_node_id" in same_depot_all_dates.columns:
            same_depot_all_dates = (
                same_depot_all_dates.sort_values("to_depot_km")
                .drop_duplicates(subset=["customer_node_id"])
                .copy()
            )

        local = same_depot_all_dates.head(
            max(max_total_stops * 4, num_representatives * 5)
        ).copy()

    print(f"chosen preview depot: ({depot_lat}, {depot_lon})")
    print(f"chosen local preview stop count before rep assignment: {len(local)}")
    print(
        f"chosen local max distance from depot: {float(local['to_depot_km'].max()) if not local.empty else 0.0:.2f} km"
    )
    print(
        f"final preview local max distance before drop: {float(local['to_depot_km'].max()) if not local.empty else 0.0:.2f} km"
    )

    print(f"local rows after all fallback stages: {len(local)}")
    if "agent_id" in local.columns:
        print(
            f"distinct agent_id in local: {local['agent_id'].astype(str).replace({'': 'UNKNOWN', 'nan': 'UNKNOWN', 'None': 'UNKNOWN'}).nunique()}"
        )
    if "customer_node_id" in local.columns:
        print(
            f"distinct customer_node_id in local: {local['customer_node_id'].nunique()}"
        )

    local = local.drop(columns=["to_depot_km"], errors="ignore").copy()

    if use_existing_agents and strict_existing_agents and "agent_id" in local.columns:
        strict_local = local.copy()
        strict_local["agent_id"] = (
            strict_local["agent_id"].astype(str).fillna("UNKNOWN")
        )
        strict_local["agent_id"] = strict_local["agent_id"].replace(
            {"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"}
        )

        strict_local = strict_local[strict_local["agent_id"] != "UNKNOWN"].copy()

        if not strict_local.empty:
            strict_local["rep_id"] = strict_local["agent_id"].astype(str)
            print(
                "strict existing-agent preservation active; "
                f"keeping all available real agents: {strict_local['rep_id'].nunique()}"
            )
            return strict_local.reset_index(drop=True)

    if use_existing_agents:
        preview_assigned = assign_preview_rep_ids_from_agent(
            local,
            num_representatives,
            max_total_stops=max_total_stops,
            strict_existing_agents=strict_existing_agents,
        )
        if not preview_assigned.empty and preview_assigned["rep_id"].nunique() > 0:
            return preview_assigned

    preview_assigned = assign_preview_rep_ids_uneven(local, num_representatives)
    return preview_assigned


def build_local_preview_subset_amazon(
    df: pd.DataFrame,
    num_representatives: int,
    max_total_stops: Optional[int] = None,
    initial_radius_km: float = 25.0,
    max_radius_km: float = 120.0,
    local_cap_km: float = 100.0,
    use_existing_agents: bool = True,
    strict_existing_agents: bool = True,
    min_nodes_per_rep: int = 1,
    max_customers_per_rep: Optional[int] = AMAZON_MAX_CUSTOMERS_PER_REP,
) -> pd.DataFrame:
    """
    Amazon-specific preview builder with stronger agent-based clustering.

    This keeps the Zomato-style idea of preserving agent_id as rep_id, but adds
    an Amazon-only customer selection cap so one agent does not receive too many
    preview customers/orders.

    Amazon behavior:
    - uses only Amazon logic; Zomato flow is untouched
    - preserves order-level Amazon stops instead of collapsing to 12 physical nodes
    - preserves agent_id as rep_id
    - selects the strongest 6 agent clusters from one depot
    - keeps at most AMAZON_MAX_CUSTOMERS_PER_REP customers/orders per selected rep
    - within each selected agent, keeps the best compact customers first
    """
    work = df.copy()

    effective_reps = max(AMAZON_DEFAULT_REPRESENTATIVES, int(num_representatives))
    per_rep_cap = int(max_customers_per_rep or AMAZON_MAX_CUSTOMERS_PER_REP)
    per_rep_cap = max(1, per_rep_cap)

    # Total Amazon preview size is controlled by reps × cap.
    # This prevents one selected Amazon agent from carrying too many stops.
    target_preview_rows = effective_reps * per_rep_cap

    depot_lat, depot_lon, depot_cluster = choose_best_local_depot_cluster(
        work,
        candidate_pool_size=max(target_preview_rows * 4, effective_reps * 8),
        prefer_agent_coverage=True,
        min_agents=effective_reps,
    )

    print(f"chosen preview depot: ({depot_lat}, {depot_lon})")

    depot_cluster["to_depot_km"] = depot_cluster.apply(
        lambda r: haversine_km(
            depot_lat,
            depot_lon,
            float(r["customer_lat"]),
            float(r["customer_lon"]),
        ),
        axis=1,
    )

    radius = initial_radius_km
    local = depot_cluster[depot_cluster["to_depot_km"] <= radius].copy()

    while len(local) < target_preview_rows and radius < max_radius_km:
        radius *= 1.5
        local = depot_cluster[depot_cluster["to_depot_km"] <= radius].copy()

    # Keep the cluster local. If the cap is too restrictive, refill from the same
    # depot only, still sorted by compactness.
    local = local[local["to_depot_km"] <= local_cap_km].copy()
    if len(local) < target_preview_rows:
        local = (
            depot_cluster.sort_values("to_depot_km")
            .head(target_preview_rows * 5)
            .copy()
        )

    local = local.sort_values("to_depot_km").copy()

    if "agent_id" in local.columns:
        local["agent_id"] = local["agent_id"].astype(str).fillna("UNKNOWN")
        local["agent_id"] = local["agent_id"].replace(
            {"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"}
        )

        valid_local = local[local["agent_id"] != "UNKNOWN"].copy()

        if not valid_local.empty:
            # Score each Amazon agent cluster. This avoids simply taking the agents
            # with the most orders, which can create too many customers per rep.
            agent_summary_rows: List[Dict[str, Any]] = []
            for agent_id, grp in valid_local.groupby("agent_id"):
                grp = grp.copy()
                rows = int(len(grp))
                physical_nodes = (
                    int(grp["physical_customer_node_id"].nunique())
                    if "physical_customer_node_id" in grp.columns
                    else int(grp["customer_node_id"].nunique())
                )
                mean_dist = float(grp["to_depot_km"].mean())
                max_dist = float(grp["to_depot_km"].max())
                mean_eta = float(
                    pd.to_numeric(grp["predicted_eta_min"], errors="coerce")
                    .fillna(0)
                    .mean()
                )
                mean_rating = float(
                    pd.to_numeric(grp["rating"], errors="coerce").fillna(4.0).mean()
                )

                # Prefer agents with enough rows for the cap, compact stops, and
                # good quality. Do not over-prefer huge groups.
                shortage = max(0, per_rep_cap - rows)
                agent_summary_rows.append(
                    {
                        "agent_id": str(agent_id),
                        "rows": rows,
                        "physical_nodes": physical_nodes,
                        "mean_dist": mean_dist,
                        "max_dist": max_dist,
                        "mean_eta": mean_eta,
                        "mean_rating": mean_rating,
                        "score": (
                            shortage,
                            -min(rows, per_rep_cap),
                            mean_dist,
                            max_dist,
                            mean_eta,
                            -mean_rating,
                        ),
                    }
                )

            agent_summary = pd.DataFrame(agent_summary_rows)
            agent_summary = agent_summary.sort_values("score").reset_index(drop=True)
            top_agents = (
                agent_summary.head(effective_reps)["agent_id"].astype(str).tolist()
            )

            selected_parts: List[pd.DataFrame] = []
            for agent_id in top_agents:
                grp = valid_local[
                    valid_local["agent_id"].astype(str) == str(agent_id)
                ].copy()

                # Keep best customers/orders per selected agent:
                # nearby first, then lower predicted ETA, then higher rating.
                grp["_pred_eta_sort"] = pd.to_numeric(
                    grp["predicted_eta_min"], errors="coerce"
                ).fillna(grp["to_depot_km"] * 3.0 + 8.0)
                grp["_rating_sort"] = pd.to_numeric(
                    grp["rating"], errors="coerce"
                ).fillna(4.0)

                grp = grp.sort_values(
                    ["to_depot_km", "_pred_eta_sort", "_rating_sort"],
                    ascending=[True, True, False],
                )

                # Avoid showing duplicate physical nodes for the same agent unless
                # there are not enough distinct physical customers.
                if "physical_customer_node_id" in grp.columns:
                    distinct_first = grp.drop_duplicates(
                        subset=["physical_customer_node_id"]
                    ).copy()
                    if len(distinct_first) >= per_rep_cap:
                        grp = distinct_first

                selected_parts.append(grp.head(per_rep_cap))

            if selected_parts:
                selected = pd.concat(selected_parts, ignore_index=True)
                selected["rep_id"] = selected["agent_id"].astype(str)

                # In rare cases, an agent may have fewer than cap rows. Refill from
                # unselected valid rows without exceeding the per-rep cap.
                if len(selected) < target_preview_rows:
                    used_order_ids = (
                        set(selected["order_id"].astype(str))
                        if "order_id" in selected.columns
                        else set()
                    )
                    counts = selected["rep_id"].value_counts().to_dict()
                    for agent_id in top_agents:
                        need = per_rep_cap - int(counts.get(agent_id, 0))
                        if need <= 0:
                            continue
                        extra = valid_local[
                            (valid_local["agent_id"].astype(str) == str(agent_id))
                            & (
                                ~valid_local["order_id"]
                                .astype(str)
                                .isin(used_order_ids)
                            )
                        ].copy()
                        if extra.empty:
                            continue
                        extra = extra.sort_values("to_depot_km").head(need).copy()
                        extra["rep_id"] = extra["agent_id"].astype(str)
                        selected = pd.concat([selected, extra], ignore_index=True)
                        used_order_ids.update(extra["order_id"].astype(str).tolist())

                # Final guard: never allow more than max customers/orders per rep.
                selected = (
                    selected.sort_values(["rep_id", "to_depot_km"])
                    .groupby("rep_id", group_keys=False)
                    .head(per_rep_cap)
                    .reset_index(drop=True)
                )
                print(
                    "Amazon stronger clustering active; "
                    f"selected agents: {selected['rep_id'].nunique()}, "
                    f"max customers/orders per rep: {per_rep_cap}"
                )
                print(f"chosen preview depot: ({depot_lat}, {depot_lon})")
                print(
                    f"chosen local preview stop count before rep assignment: {len(selected)}"
                )
                print(
                    f"chosen local max distance from depot: "
                    f"{float(selected['to_depot_km'].max()) if not selected.empty else 0.0:.2f} km"
                )
                print(f"local rows after all fallback stages: {len(selected)}")
                print(
                    "customers/orders per selected agent:",
                    selected["rep_id"].value_counts().to_dict(),
                )
                if "customer_node_id" in selected.columns:
                    print(
                        f"distinct route customer_node_id in local: {selected['customer_node_id'].nunique()}"
                    )
                if "physical_customer_node_id" in selected.columns:
                    print(
                        f"distinct physical_customer_node_id in local: {selected['physical_customer_node_id'].nunique()}"
                    )

                return selected.drop(
                    columns=["to_depot_km", "_pred_eta_sort", "_rating_sort"],
                    errors="ignore",
                ).reset_index(drop=True)

    # Fallback only if Amazon has no usable agent_id. This is not expected for
    # the reconstructed Amazon dataset, but keeps the backend safe.
    print("Amazon agent_id unavailable; falling back to uneven spatial assignment.")
    local = (
        local.head(target_preview_rows)
        .drop(columns=["to_depot_km"], errors="ignore")
        .copy()
    )
    preview_assigned = assign_preview_rep_ids_uneven(local, effective_reps)
    return preview_assigned


def preview_summary_from_assign_df(assign_df: pd.DataFrame) -> Dict[str, Any]:
    if assign_df.empty:
        return {
            "selectionStrategy": "single-depot high-node local preview",
            "maxRoutes": 0,
            "maxTotalStops": 0,
            "maxDistanceFromDepotKm": 0.0,
            "depotLat": None,
            "depotLon": None,
        }

    depot_row = assign_df.iloc[0]
    depot_lat = float(depot_row["depot_lat"])
    depot_lon = float(depot_row["depot_lon"])

    distances = assign_df.apply(
        lambda r: haversine_km(
            r["customer_lat"], r["customer_lon"], depot_lat, depot_lon
        ),
        axis=1,
    )

    return {
        "selectionStrategy": "single-depot nearest-customer compact preview",
        "maxRoutes": int(assign_df["rep_id"].nunique()),
        "maxTotalStops": int(len(assign_df)),
        "maxDistanceFromDepotKm": round(float(distances.max()), 2),
        "depotLat": depot_lat,
        "depotLon": depot_lon,
    }


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/datasets/validate")
async def validate_dataset(
    file: UploadFile = File(...),
    mapping_json: str = Form(...),
    dataset_role: Optional[str] = Form(None),
) -> Dict[str, Any]:
    try:
        mapping = FieldMapping(**json.loads(mapping_json))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid mapping JSON: {exc}"
        ) from exc

    resolved_role = dataset_role or infer_dataset_role(file.filename or "")
    df = read_csv_upload(file)

    mapping = autofill_mapping_from_known_columns(df, mapping, resolved_role)

    normalized = normalize_dataset(df, mapping, resolved_role)

    print("validate dataset role:", resolved_role)
    print("normalized rows:", len(normalized))
    print("effective mapping:", mapping.model_dump())
    if "order_date" in normalized.columns:
        print(
            "normalized unique order_date sample:",
            sorted(normalized["order_date"].dropna().astype(str).unique())[:10],
        )
    if "agent_id" in normalized.columns:
        print(
            "normalized distinct agent_id:",
            int(normalized["agent_id"].astype(str).nunique()),
        )
    if "depot_id" in normalized.columns:
        print(
            "normalized distinct depot_id:",
            int(normalized["depot_id"].astype(str).nunique()),
        )

    summary = validation_summary(normalized)

    dataset_id = str(uuid.uuid4())
    reconstructed_name = (
        f"reconstructed_{(file.filename or 'dataset').replace('.csv', '')}.csv"
    )

    DATASETS[dataset_id] = {
        "data": normalized,
        "mapping": mapping.model_dump(),
        "filename": file.filename,
        "datasetRole": resolved_role,
        "sourceLabel": role_label(resolved_role),
        "reconstructedBaselineName": reconstructed_name,
    }

    return {
        "datasetId": dataset_id,
        "datasetRole": resolved_role,
        "sourceLabel": role_label(resolved_role),
        "reconstructedBaselineReady": True,
        "reconstructedBaselineName": reconstructed_name,
        **summary,
    }


@app.get("/api/datasets/{dataset_id}/meta")
def dataset_meta(dataset_id: str) -> Dict[str, Any]:
    payload = DATASETS.get(dataset_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    df = payload["data"]

    depot_row = df.iloc[0]
    depot = {
        "id": str(depot_row["depot_id"]),
        "lat": float(depot_row["depot_lat"]),
        "lon": float(depot_row["depot_lon"]),
        "name": str(depot_row["depot_id"]),
    }

    return {
        "datasetId": dataset_id,
        "filename": payload["filename"],
        "datasetRole": payload["datasetRole"],
        "sourceLabel": payload["sourceLabel"],
        "reconstructedBaselineName": payload["reconstructedBaselineName"],
        "records": int(len(df)),
        "depots": int(df["depot_id"].nunique()),
        "customers": int(df["customer_id"].nunique()),
        "customerNodes": int(df["customer_node_id"].nunique())
        if "customer_node_id" in df.columns
        else int(df["customer_id"].nunique()),
        "orders": int(df["order_id"].nunique()),
        "depot": depot,
    }


@app.get("/api/datasets/{dataset_id}/reconstructed")
def download_reconstructed_dataset(dataset_id: str):
    payload = DATASETS.get(dataset_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    df = payload["data"].copy()
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={payload['reconstructedBaselineName']}"
        },
    )


@app.post("/api/runs/baseline")
def run_baseline(req: BaselineRequest) -> Dict[str, Any]:
    print("run_baseline started")

    payload = DATASETS.get(req.dataset_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    if req.num_representatives < 4 or req.num_representatives > 15:
        raise HTTPException(
            status_code=400,
            detail="Number of representatives must be between 4 and 15.",
        )

    profile = get_run_profile(req.run_profile)
    print("baseline profile:", profile["profile_name"])

    print("dataset found")
    df = payload["data"].copy()
    print(f"data copied: {len(df)} rows")

    predicted_eta, metrics = train_eta_models(df, req.seed)
    print("train_eta_models done")
    df["predicted_eta_min"] = predicted_eta

    role = payload["datasetRole"]
    effective_num_representatives = (
        max(AMAZON_DEFAULT_REPRESENTATIVES, req.num_representatives)
        if role == "primary_reconstruction"
        else req.num_representatives
    )

    if role == "primary_reconstruction":
        routing_df = build_amazon_order_routing_rows(df)
        print(f"amazon order-level routing_df built: {len(routing_df)} order rows")
    else:
        routing_df = build_routing_nodes(df)
        print(f"routing_df built: {len(routing_df)} node rows")

    # Keep Zomato using the original fixed-depot strength check.
    # Amazon keeps the no-minimum fixed-depot behavior without changing its routing logic.
    depot_min_nodes = (
        AMAZON_FIXED_DEMO_NODES
        if role == "primary_reconstruction"
        else MIN_FIXED_DEMO_NODES
    )
    depot_min_agents = (
        AMAZON_FIXED_DEMO_AGENTS
        if role == "primary_reconstruction"
        else MIN_FIXED_DEMO_AGENTS
    )

    routing_df = filter_df_to_demo_depot(
        routing_df,
        payload["datasetRole"],
        min_nodes=depot_min_nodes,
        min_agents=depot_min_agents,
    )
    print(f"routing_df after demo depot filter: {len(routing_df)} rows")
    role_note = (
        "Primary Amazon-based reconstructed baseline workflow"
        if role == "primary_reconstruction"
        else "Comparative/template workflow using Zomato-aligned structure"
        if role == "comparative_template"
        else "Generic uploaded dataset workflow"
    )

    preview_max_total_stops = (
        max(profile["preview_max_total_stops"], 40)
        if role == "comparative_template"
        else max(profile["preview_max_total_stops"], AMAZON_MIN_PREVIEW_STOPS)
        if role == "primary_reconstruction"
        else profile["preview_max_total_stops"]
    )

    if role == "primary_reconstruction":
        preview_df = build_local_preview_subset_amazon(
            routing_df,
            num_representatives=effective_num_representatives,
            max_total_stops=preview_max_total_stops,
            initial_radius_km=profile["preview_initial_radius_km"],
            max_radius_km=profile["preview_max_radius_km"],
            local_cap_km=profile["preview_local_cap_km"],
            use_existing_agents=True,
            strict_existing_agents=True,
            min_nodes_per_rep=1,
            max_customers_per_rep=AMAZON_MAX_CUSTOMERS_PER_REP,
        )
    elif role == "comparative_template":
        preview_df = build_local_preview_subset(
            routing_df,
            num_representatives=req.num_representatives,
            max_total_stops=max(profile["preview_max_total_stops"], 40),
            initial_radius_km=profile["preview_initial_radius_km"],
            max_radius_km=profile["preview_max_radius_km"],
            local_cap_km=profile["preview_local_cap_km"],
            use_existing_agents=True,
            strict_existing_agents=True,
            min_nodes_per_rep=3,
        )
    else:
        preview_df = build_local_preview_subset(
            routing_df,
            num_representatives=effective_num_representatives,
            max_total_stops=preview_max_total_stops,
            initial_radius_km=profile["preview_initial_radius_km"],
            max_radius_km=profile["preview_max_radius_km"],
            local_cap_km=profile["preview_local_cap_km"],
            use_existing_agents=False,
            strict_existing_agents=False,
            min_nodes_per_rep=3,
        )
    print(f"preview_df built: {len(preview_df)} rows")

    preview_df = ensure_preview_node_ids(preview_df)
    depot_lat = float(preview_df.iloc[0]["depot_lat"])
    depot_lon = float(preview_df.iloc[0]["depot_lon"])

    preview_df["debug_to_depot_km"] = preview_df.apply(
        lambda r: haversine_km(
            depot_lat,
            depot_lon,
            float(r["customer_lat"]),
            float(r["customer_lon"]),
        ),
        axis=1,
    )

    print("preview stop distances from depot (km):")
    print(
        preview_df[["customer_node_id", "customer_name", "debug_to_depot_km"]]
        .sort_values("debug_to_depot_km", ascending=False)
        .head(12)
    )
    print(
        "max preview distance from depot:", float(preview_df["debug_to_depot_km"].max())
    )
    preview_matrix = build_preview_distance_matrix(
        preview_df,
        osm_threshold_km=profile["preview_osm_threshold_km"],
    )
    print("preview_matrix built")
    matrix_stats = preview_matrix_stats(preview_df)

    preview_routes, preview_rep_df, preview_total = route_all(
        preview_df,
        req.avg_speed_kmph,
        req.service_minutes_per_stop,
        "baseline",
        preview_matrix,
    )
    print("preview route_all done")

    preview_routes = attach_route_display_geometry(preview_routes, preview_df)
    print("baseline display geometry attached")

    preview_run = make_algorithm_run(
        "Baseline G-NN + Dijkstra",
        preview_routes,
        preview_rep_df,
        preview_total,
        len(preview_df),
        metrics["baseline"],
        notes=[
            role_note,
            "Preview mode for UI rendering",
            "Preview restricted to nearest customers within the fixed demo depot",
            "Uses existing agent_id where available; Zomato strongly preserves real agents before any fallback",
            f"Preview target: {preview_max_total_stops} stops; Amazon order-level preview is not capped to 12 customer nodes",
            f"Fixed demo depot override: {DEMO_PREVIEW_DEPOTS.get(role) or 'automatic'}",
        ],
        assign_df=preview_df,
    )

    preview_run["datasetId"] = req.dataset_id
    preview_run["runType"] = "baseline"
    preview_run["datasetRole"] = role
    preview_run["sourceLabel"] = payload["sourceLabel"]
    preview_run["trainingComparison"] = metrics
    preview_run["previewMode"] = True
    preview_run["previewSummary"] = preview_summary_from_assign_df(preview_df)
    preview_run["matrixMode"] = "osm_or_proxy_preview_matrix"
    preview_run["matrixStats"] = matrix_stats
    preview_run["runProfile"] = profile["profile_name"]
    preview_run["profileConfig"] = profile

    RUNS[preview_run["id"]] = {
        "assign_df": preview_df,
        "distance_matrix": preview_matrix,
        "request": req.model_dump(),
        "run": preview_run,
        "profile": profile,
    }

    print("run_baseline finished")
    return preview_run


@app.post("/api/runs/baseline/add-customers")
def add_customers_to_baseline(req: BaselineAddCustomersRequest) -> Dict[str, Any]:
    baseline_payload = RUNS.get(req.baseline_run_id)
    if not baseline_payload:
        raise HTTPException(status_code=404, detail="Baseline run not found.")

    if not req.customers:
        raise HTTPException(status_code=400, detail="No customers supplied.")

    assign_df = baseline_payload["assign_df"].copy()
    baseline_req = BaselineRequest(**baseline_payload["request"])
    base_run = baseline_payload["run"]
    profile = baseline_payload.get("profile", get_run_profile(None))

    resolved_customers: List[AddedCustomerPayload] = []
    updated_assign_df = assign_df.copy()

    # Assign added customers sequentially using the nearest existing route.
    # After each customer is assigned, append it immediately so the next added
    # customer sees the updated route state.
    for customer in req.customers:
        assigned_rep = assign_new_customer_to_nearest_rep(
            updated_assign_df,
            float(customer.lat),
            float(customer.lon),
        )

        resolved_customer = AddedCustomerPayload(
            label=customer.label,
            lat=customer.lat,
            lon=customer.lon,
            address=customer.address,
            assigned_rep=assigned_rep,
            customer_number=customer.customer_number,
        )
        resolved_customers.append(resolved_customer)
        updated_assign_df = append_added_customers_to_assign_df(
            updated_assign_df, [resolved_customer]
        )

    updated_assign_df = ensure_preview_node_ids(updated_assign_df)

    updated_matrix = build_preview_distance_matrix(
        updated_assign_df,
        osm_threshold_km=profile["preview_osm_threshold_km"],
    )
    updated_matrix_stats = preview_matrix_stats(updated_assign_df)

    routes, rep_df, total = route_all(
        updated_assign_df,
        baseline_req.avg_speed_kmph,
        baseline_req.service_minutes_per_stop,
        "baseline",
        updated_matrix,
    )

    routes = attach_route_display_geometry(routes, updated_assign_df)

    updated_run = make_algorithm_run(
        "Baseline G-NN + Dijkstra",
        routes,
        rep_df,
        total,
        len(updated_assign_df),
        base_run.get("trainingMetrics", {}),
        notes=(base_run.get("notes", []) + [f"Added customers: {len(req.customers)}"]),
        assign_df=updated_assign_df,
    )

    updated_run["datasetId"] = base_run["datasetId"]
    updated_run["runType"] = "baseline"
    updated_run["datasetRole"] = base_run.get("datasetRole")
    updated_run["sourceLabel"] = base_run.get("sourceLabel")
    updated_run["trainingComparison"] = base_run.get("trainingComparison")
    updated_run["previewMode"] = True
    updated_run["previewSummary"] = preview_summary_from_assign_df(updated_assign_df)
    updated_run["matrixMode"] = "osm_or_proxy_preview_matrix"
    updated_run["matrixStats"] = updated_matrix_stats
    updated_run["runProfile"] = base_run.get("runProfile")
    updated_run["profileConfig"] = base_run.get("profileConfig")
    updated_run["addedCustomers"] = [
        {
            "label": c.label,
            "lat": c.lat,
            "lon": c.lon,
            "address": c.address,
            "assignedRep": c.assigned_rep,
            "customerNumber": c.customer_number,
        }
        for c in resolved_customers
    ]

    RUNS[updated_run["id"]] = {
        "assign_df": updated_assign_df,
        "distance_matrix": updated_matrix,
        "request": baseline_req.model_dump(),
        "run": updated_run,
        "profile": profile,
    }

    return updated_run


def assign_new_customer_to_nearest_rep(
    assign_df: pd.DataFrame,
    customer_lat: float,
    customer_lon: float,
) -> str:
    """
    Assign a newly added customer to the representative whose existing route
    has the nearest customer stop to the new customer.

    This is used for Add Customer so multiple new customers are not all pushed
    to the same preselected rep unless that rep is truly nearest after each
    sequential update.
    """
    work = ensure_preview_node_ids(assign_df.copy())
    if work.empty or "rep_id" not in work.columns:
        return "UNASSIGNED"

    best_rep = None
    best_distance = None
    best_workload_count = None

    for rep_id, grp in work.groupby("rep_id"):
        rep_id_str = str(rep_id)
        if grp.empty:
            continue

        nearest_km = min(
            haversine_km(
                float(customer_lat),
                float(customer_lon),
                float(r["customer_lat"]),
                float(r["customer_lon"]),
            )
            for _, r in grp.iterrows()
        )
        workload_count = int(len(grp))

        # Main rule: nearest representative route wins.
        # Tie-breaker: fewer currently assigned stops.
        candidate = (nearest_km, workload_count, rep_id_str)
        if best_distance is None or candidate < (
            best_distance,
            best_workload_count or 0,
            best_rep or "",
        ):
            best_distance = nearest_km
            best_workload_count = workload_count
            best_rep = rep_id_str

    return best_rep or "UNASSIGNED"


@app.post("/api/runs/enhanced")
def run_enhanced(req: EnhancedRequest) -> Dict[str, Any]:
    print("enhanced started")
    dataset_payload = DATASETS.get(req.dataset_id)
    baseline_payload = RUNS.get(req.baseline_run_id)

    if not dataset_payload:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if not baseline_payload:
        raise HTTPException(status_code=404, detail="Baseline run not found.")

    role = dataset_payload["datasetRole"]
    baseline_profile = baseline_payload.get("profile", get_run_profile(None))
    profile = get_run_profile(req.run_profile or baseline_profile.get("profile_name"))

    print("enhanced profile:", profile["profile_name"])
    print("enhanced params:", req.model_dump())

    distance_matrix = baseline_payload.get("distance_matrix", {})

    # task 6: alpha/beta for priority scoring
    alpha = float(req.alpha_weight if req.alpha_weight is not None else 0.60)
    beta = float(req.beta_weight if req.beta_weight is not None else 0.40)

    if alpha < 0:
        alpha = 0.0
    if beta < 0:
        beta = 0.0

    weight_sum = alpha + beta
    if weight_sum <= 0:
        alpha, beta = 0.60, 0.40
    else:
        alpha = alpha / weight_sum
        beta = beta / weight_sum

    # keep current optimization weights for now
    effective_fairness_weight = profile["enhanced_fairness_weight"]
    effective_distance_weight = profile["enhanced_distance_weight"]
    effective_time_weight = profile["enhanced_time_weight"]

    effective_max_iterations = (
        req.max_iterations
        if req.max_iterations is not None
        else profile["enhanced_max_iterations"]
    )
    effective_border_fraction = (
        req.border_fraction
        if req.border_fraction is not None
        else profile["enhanced_border_fraction"]
    )

    print("normalized alpha/beta:", {"alpha": alpha, "beta": beta})

    print("enhanced dataset and baseline found")
    baseline_req = BaselineRequest(**baseline_payload["request"])
    assign_df = baseline_payload["assign_df"].copy()
    print(f"enhanced assign_df rows: {len(assign_df)}")

    is_zomato_mode = role == "comparative_template"

    improved_df, logs = enhance_assignment(
        assign_df,
        baseline_req.avg_speed_kmph,
        baseline_req.service_minutes_per_stop,
        alpha,
        beta,
        effective_fairness_weight,
        effective_distance_weight,
        effective_time_weight,
        effective_max_iterations,
        effective_border_fraction,
        distance_matrix,
        is_zomato_mode=is_zomato_mode,
    )

    if role == "primary_reconstruction":
        improved_df, amazon_polish_logs = amazon_distance_polish_assignment(
            improved_df,
            baseline_req.avg_speed_kmph,
            baseline_req.service_minutes_per_stop,
            distance_matrix,
            max_iterations=12,
        )
        logs.extend(amazon_polish_logs)

    print("enhance_assignment done")

    routes, rep_df, total = route_all(
        improved_df,
        baseline_req.avg_speed_kmph,
        baseline_req.service_minutes_per_stop,
        "enhanced",
        distance_matrix,
    )
    print("enhanced route_all done")

    routes = attach_route_display_geometry(routes, improved_df)
    print("enhanced display geometry attached")

    training_metrics = baseline_payload["run"].get("trainingComparison", {})
    role = dataset_payload["datasetRole"]
    role_note = (
        "Enhanced DEQ run over Amazon-derived reconstructed baseline"
        if role == "primary_reconstruction"
        else "Enhanced DEQ run for comparative/template Zomato evaluation"
        if role == "comparative_template"
        else "Enhanced DEQ run over uploaded dataset"
    )

    run = make_algorithm_run(
        "Enhanced G-NN + DEQ Rebalancing",
        routes,
        rep_df,
        total,
        len(improved_df),
        training_metrics.get("enhanced", baseline_payload["run"]["trainingMetrics"]),
        notes=[
            role_note,
            "Baseline-seeded DEQ rebalancing",
            "Priority scoring uses alpha/beta for time difference and rating",
            "Joint acceptance on workload balance, distance, and operational time",
            f"Accepted rebalances: {sum(1 for x in logs if x.get('accepted'))}",
        ],
        assign_df=improved_df,
    )

    run["datasetId"] = req.dataset_id
    run["baselineRunId"] = req.baseline_run_id
    run["runType"] = "enhanced"
    run["datasetRole"] = role
    run["sourceLabel"] = dataset_payload["sourceLabel"]
    run["runLog"] = logs
    run["runProfile"] = profile["profile_name"]
    run["profileConfig"] = profile
    run["previewSummary"] = preview_summary_from_assign_df(improved_df)
    run["previewMode"] = True

    RUNS[run["id"]] = {
        "assign_df": improved_df,
        "distance_matrix": distance_matrix,
        "request": baseline_req.model_dump(),
        "run": run,
        "profile": profile,
    }
    return run
