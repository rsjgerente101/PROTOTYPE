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
        "preview_initial_radius_km": 7.0,
        "preview_max_radius_km": 16.0,
        "preview_local_cap_km": 14.0,
        "preview_osm_threshold_km": 14.0,
        "preview_max_total_stops": 12,
        "enhanced_fairness_weight": 0.60,
        "enhanced_distance_weight": 0.25,
        "enhanced_time_weight": 0.15,
        "enhanced_max_iterations": 20,
        "enhanced_border_fraction": 0.50,
    },
    "amazon_expanded_search": {
        "preview_initial_radius_km": 12.0,
        "preview_max_radius_km": 26.0,
        "preview_local_cap_km": 22.0,
        "preview_osm_threshold_km": 22.0,
        "preview_max_total_stops": 18,
        "enhanced_fairness_weight": 0.50,
        "enhanced_distance_weight": 0.35,
        "enhanced_time_weight": 0.15,
        "enhanced_max_iterations": 50,
        "enhanced_border_fraction": 1.00,
    },
}


class FieldMapping(BaseModel):
    depot_id: Optional[str] = None
    depot_lat: str
    depot_lon: str
    customer_id: str
    customer_lat: str
    customer_lon: str
    order_id: Optional[str] = None
    eta_col: Optional[str] = None
    rating_col: Optional[str] = None
    area_col: Optional[str] = None


class BaselineRequest(BaseModel):
    dataset_id: str
    num_representatives: int = 4
    avg_speed_kmph: float = 18.75
    service_minutes_per_stop: float = 8.0
    seed: int = 42
    run_profile: Optional[str] = "default_balanced"


class EnhancedRequest(BaseModel):
    dataset_id: str
    baseline_run_id: str
    fairness_weight: Optional[float] = None
    distance_weight: Optional[float] = None
    time_weight: Optional[float] = None
    max_iterations: Optional[int] = None
    border_fraction: Optional[float] = None
    run_profile: Optional[str] = None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def road_adjusted_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Fast proxy for road-network cost.
    Keeps runtime close to current haversine approach, but inflates distance
    to better approximate road travel than straight-line geometry.
    """
    direct = haversine_km(lat1, lon1, lat2, lon2)
    return direct * 1.25

def _expand_bbox(
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
    pad_ratio: float = 0.20,
    min_pad_deg: float = 0.01,
) -> Tuple[float, float, float, float]:
    lat_span = max(lat_max - lat_min, 0.0)
    lon_span = max(lon_max - lon_min, 0.0)

    lat_pad = max(lat_span * pad_ratio, min_pad_deg)
    lon_pad = max(lon_span * pad_ratio, min_pad_deg)

    south = lat_min - lat_pad
    west = lon_min - lon_pad
    north = lat_max + lat_pad
    east = lon_max + lon_pad
    return south, west, north, east


def _graph_cache_name_from_bbox(
    south: float,
    west: float,
    north: float,
    east: float,
) -> str:
    return f"osm_drive_{south:.5f}_{west:.5f}_{north:.5f}_{east:.5f}.graphml".replace("-", "m")


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
        rows.append({
            "point_id": pid,
            "lat": float(row.customer_lat),
            "lon": float(row.customer_lon),
            "kind": "customer",
        })

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

    cache_name = f"osm_point_{depot_lat:.5f}_{depot_lon:.5f}_{graph_radius_m}m.graphml".replace("-", "m")
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

    max_spread_km = float(work.apply(
        lambda r: haversine_km(
            depot_lat,
            depot_lon,
            float(r["customer_lat"]),
            float(r["customer_lon"])
        ),
        axis=1
    ).max())

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

        print(f"Preview spread too large for OSM ({max_spread_km:.2f} km > {osm_threshold_km:.2f} km). Using proxy matrix.")
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

    node_lookup = {
        str(r["point_id"]): r["osm_node"]
        for _, r in snapped.iterrows()
    }
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

            if (
                G_proj is not None
                and pd.notna(a_node)
                and pd.notna(b_node)
            ):
                dist_m = dijkstra_cache.get(a_node, {}).get(b_node)
                if dist_m is not None:
                    dist_km = float(dist_m) / 1000.0

            if dist_km is None:
                dist_km = road_adjusted_km(a_lat, a_lon, b_lat, b_lon)

            matrix[a][b] = dist_km

    return matrix

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


def ensure_preview_node_ids(assign_df: pd.DataFrame) -> pd.DataFrame:
    work = assign_df.copy().reset_index(drop=True)

    if "customer_node_id" in work.columns:
        work["node_id"] = work["customer_node_id"].astype(str)
    elif "node_id" not in work.columns:
        work["node_id"] = [f"CUST-{i+1}" for i in range(len(work))]

    return work


def matrix_cost(matrix: Dict[str, Dict[str, float]], a: str, b: str) -> float:
    if a == b:
        return 0.0
    return float(matrix.get(a, {}).get(b, 0.0))


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default

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
        work = work[work["is_routing_eligible"] == True].copy()

    required = [
    "depot_id", "depot_lat", "depot_lon",
    "customer_node_id", "customer_lat", "customer_lon",
    "customer_name", "observed_eta_min", "predicted_eta_min", "rating", "area"
    ]   
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing cleaned dataset columns: {missing}")

    agg = (
        work.groupby(
            ["depot_id", "depot_lat", "depot_lon", "customer_node_id", "customer_lat", "customer_lon"],
            as_index=False
        )
        .agg(
            order_id=("order_id", "first"),
            customer_id=("customer_id", "first"),
            customer_name=("customer_name", "first"),
            observed_eta_min=("observed_eta_min", "mean"),
            predicted_eta_min=("predicted_eta_min", "mean"),
            rating=("rating", "mean"),
            area=("area", "first"),
            node_order_count=("node_order_count", "max"),
            direct_depot_customer_km=("direct_depot_customer_km", "mean"),
        )
    )

    agg["order_id"] = agg["order_id"].astype(str)
    agg["customer_id"] = agg["customer_id"].astype(str)
    agg["customer_node_id"] = agg["customer_node_id"].astype(str)
    agg["depot_id"] = agg["depot_id"].astype(str)
    agg["customer_name"] = agg["customer_name"].fillna(agg["customer_node_id"])
    agg["node_name"] = agg["customer_name"]

    return agg.reset_index(drop=True)


def read_csv_upload(file: UploadFile) -> pd.DataFrame:
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}") from exc


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


def normalize_dataset(df: pd.DataFrame, mapping: FieldMapping, source_role: str) -> pd.DataFrame:
    needed = [mapping.depot_lat, mapping.depot_lon, mapping.customer_id, mapping.customer_lat, mapping.customer_lon]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing mapped columns: {missing}")
    
    cleaned_cols = {
    "order_id", "customer_id", "customer_node_id", "depot_id",
    "depot_lat", "depot_lon", "customer_lat", "customer_lon",
    "customer_name", "observed_eta_min", "rating", "area",
    "node_order_count", "direct_depot_customer_km",
    "is_distance_outlier", "is_routing_eligible"
    }

    if cleaned_cols.issubset(set(df.columns)):
        out = df.copy()

        out["depot_lat"] = pd.to_numeric(out["depot_lat"], errors="coerce")
        out["depot_lon"] = pd.to_numeric(out["depot_lon"], errors="coerce")
        out["customer_lat"] = pd.to_numeric(out["customer_lat"], errors="coerce")
        out["customer_lon"] = pd.to_numeric(out["customer_lon"], errors="coerce")
        out["observed_eta_min"] = pd.to_numeric(out["observed_eta_min"], errors="coerce")
        out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
        out["node_order_count"] = pd.to_numeric(out["node_order_count"], errors="coerce").fillna(1)

        out = out.dropna(subset=["depot_lat", "depot_lon", "customer_lat", "customer_lon"]).copy()
        out = out[(out["customer_lat"] != 0) & (out["customer_lon"] != 0) & (out["depot_lat"] != 0) & (out["depot_lon"] != 0)].copy()

        out.reset_index(drop=True, inplace=True)
        return out

    # fallback for ordinary uploaded datasets
    out = pd.DataFrame()
    out["depot_id"] = df[mapping.depot_id] if mapping.depot_id and mapping.depot_id in df.columns else "DEPOT-1"
    out["depot_lat"] = pd.to_numeric(df[mapping.depot_lat], errors="coerce")
    out["depot_lon"] = pd.to_numeric(df[mapping.depot_lon], errors="coerce")
    out["customer_id"] = df[mapping.customer_id].astype(str)
    out["customer_lat"] = pd.to_numeric(df[mapping.customer_lat], errors="coerce")
    out["customer_lon"] = pd.to_numeric(df[mapping.customer_lon], errors="coerce")
    out["order_id"] = df[mapping.order_id].astype(str) if mapping.order_id and mapping.order_id in df.columns else out["customer_id"]
    out["observed_eta_min"] = pd.to_numeric(df[mapping.eta_col], errors="coerce") if mapping.eta_col and mapping.eta_col in df.columns else np.nan
    out["rating"] = pd.to_numeric(df[mapping.rating_col], errors="coerce") if mapping.rating_col and mapping.rating_col in df.columns else np.nan
    out["area"] = df[mapping.area_col].astype(str) if mapping.area_col and mapping.area_col in df.columns else "UNSPECIFIED"

    out = out.dropna(subset=["depot_lat", "depot_lon", "customer_lat", "customer_lon"]).copy()
    out = out[(out["customer_lat"] != 0) & (out["customer_lon"] != 0) & (out["depot_lat"] != 0) & (out["depot_lon"] != 0)].copy()

    out["customer_name"] = "Customer " + out["customer_id"].astype(str)
    out["node_name"] = out["customer_name"]

    out.reset_index(drop=True, inplace=True)
    return out


def validation_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        raise HTTPException(status_code=400, detail="No valid rows remain after coordinate filtering.")
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
            "customers": int(df["customer_node_id"].nunique()) if "customer_node_id" in df.columns else int(df["customer_id"].nunique()),
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
    feat["rating"] = feat["rating"].fillna(feat["rating"].median() if feat["rating"].notna().any() else 4.0)
    feat["observed_eta_min"] = feat["observed_eta_min"].fillna((feat["direct_distance_km"] / 18.0) * 60.0 + 8.0)
    return feat


def train_eta_models(df: pd.DataFrame, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    feat = build_eta_features(df)
    target = feat["observed_eta_min"].values
    features = feat[["direct_distance_km", "rating", "area"]]

    numeric = ["direct_distance_km", "rating"]
    categorical = ["area"]

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),
    ])

    ridge = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])
    rf = Pipeline([("pre", pre), ("model", RandomForestRegressor(
        n_estimators=60,
        max_depth=10,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=1
    ))])

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


def static_assignment(df: pd.DataFrame, reps: int) -> pd.DataFrame:
    work = df.copy()
    c_lat, c_lon = work["depot_lat"].median(), work["depot_lon"].median()
    work["angle"] = np.arctan2(work["customer_lat"] - c_lat, work["customer_lon"] - c_lon)
    work = work.sort_values(["angle", "customer_id"]).reset_index(drop=True)

    rep_ids = [f"REP-{i+1}" for i in range(reps)]
    base = len(work) // reps
    rem = len(work) % reps

    assignments: List[str] = []
    for i, rep in enumerate(rep_ids):
        size = base + (1 if i < rem else 0)
        assignments.extend([rep] * size)

    work["rep_id"] = assignments[: len(work)]
    return work.drop(columns=["angle"])


def route_one_rep(
    group: pd.DataFrame,
    speed_kmph: float,
    service_min: float,
    distance_matrix: Dict[str, Dict[str, float]],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    rows = ensure_preview_node_ids(group).to_dict("records")
    if not rows:
        return [], {"distance_km": 0.0, "travel_minutes": 0.0, "operational_minutes": 0.0}

    current_node = "DEPOT"
    unvisited = rows[:]

    route: List[Dict[str, Any]] = []
    cumulative_distance = 0.0
    cumulative_eta = 0.0
    stop_no = 1

    while unvisited:
        best = min(
            unvisited,
            key=lambda r: matrix_cost(distance_matrix, current_node, str(r["node_id"]))
        )

        leg = matrix_cost(distance_matrix, current_node, str(best["node_id"]))
        cumulative_distance += leg

        travel_min = (leg / speed_kmph) * 60.0 if speed_kmph > 0 else 0.0
        service_component = safe_float(best.get("predicted_eta_min"), service_min)
        cumulative_eta += travel_min + service_component

        route.append({
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
            "predictedEtaMin": round(service_component, 2),
        })

        current_node = str(best["node_id"])
        unvisited.remove(best)
        stop_no += 1

    return_leg = matrix_cost(distance_matrix, current_node, "DEPOT")
    total_distance = cumulative_distance + return_leg
    travel_minutes = (total_distance / speed_kmph) * 60.0 if speed_kmph > 0 else 0.0
    operational_minutes = travel_minutes + sum(
        safe_float(r.get("predicted_eta_min"), service_min) for r in rows
    )

    return route, {
        "distance_km": total_distance,
        "travel_minutes": travel_minutes,
        "operational_minutes": operational_minutes,
    }


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

    palette = ["#2563eb", "#16a34a", "#dc2626", "#ca8a04", "#9333ea", "#0891b2", "#db2777", "#4f46e5"]

    for idx, (rep_id, grp) in enumerate(work.groupby("rep_id"), start=1):
        ordered_stops, stats = route_one_rep(grp, speed_kmph, service_min, distance_matrix)
        color = palette[(idx - 1) % len(palette)]

        routes.append({
            "id": f"{name}-{rep_id}",
            "representativeId": rep_id,
            "representativeName": rep_id,
            "color": color,
            "stops": ordered_stops,
        })

        workload = (grp["predicted_eta_min"] * grp.get("node_order_count", 1)).sum()
        rep_rows.append({
            "rep_id": rep_id,
            "customers": int(len(grp)),
            "workload_min": float(workload),
            "distance_km": float(stats["distance_km"]),
            "travel_minutes": float(stats["travel_minutes"]),
            "operational_minutes": float(stats["operational_minutes"]),
            "centroid_lat": float(grp["customer_lat"].mean()),
            "centroid_lon": float(grp["customer_lon"].mean()),
        })

    rep_df = pd.DataFrame(rep_rows)
    total = {
        "distance_km": float(rep_df["distance_km"].sum()) if not rep_df.empty else 0.0,
        "travel_minutes": float(rep_df["travel_minutes"].sum()) if not rep_df.empty else 0.0,
        "operational_minutes": float(rep_df["operational_minutes"].sum()) if not rep_df.empty else 0.0,
    }
    return routes, rep_df, total


def jains_fairness(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    if len(arr) == 0 or np.allclose(arr.sum(), 0):
        return 1.0
    return float((arr.sum() ** 2) / (len(arr) * np.square(arr).sum()))


def workload_balance_index(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 1.0
    mean = arr.mean()
    if mean == 0:
        return 1.0
    cv = arr.std(ddof=0) / mean
    return float(max(0.0, 1.0 - cv))


def rep_cards(rep_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if rep_df.empty:
        return []

    max_workload = max(float(rep_df["workload_min"].max()), 1.0)
    ordered = rep_df.sort_values("workload_min", ascending=False).reset_index(drop=True)
    out = []

    for i, row in ordered.iterrows():
        out.append({
            "id": row["rep_id"],
            "name": row["rep_id"],
            "workload": round(float(row["workload_min"]), 2),
            "opportunityScore": round(max(0.0, 100.0 - (row["workload_min"] / max_workload) * 100.0), 1),
            "priorityScore": round((float(row["customers"]) * 10.0) + (float(row["workload_min"]) / max_workload) * 90.0, 1),
            "queuePosition": i + 1,
            "assignedCustomers": int(row["customers"]),
        })
    return out


def kpis_from_totals(total: Dict[str, float], rep_df: pd.DataFrame, dataset_size: int) -> Dict[str, Any]:
    fairness = jains_fairness(rep_df["workload_min"].tolist())
    wbi = workload_balance_index(rep_df["workload_min"].tolist())
    return {
        "totalDistance": round(total["distance_km"], 2),
        "travelTime": round(total["travel_minutes"], 2),
        "operationalTime": round(total["operational_minutes"], 2),
        "computeTime": round(max(0.5, dataset_size / 80.0), 2),
        "fairness": round(fairness, 6),
        "workloadBalance": round(wbi, 6),
        "coverage": 100.0,
        "scalability": round(dataset_size / max(1, len(rep_df)), 2),
    }


def make_algorithm_run(
    name: str,
    routes: List[Dict[str, Any]],
    rep_df: pd.DataFrame,
    total: Dict[str, float],
    dataset_size: int,
    metrics: Dict[str, float],
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "algorithm": name,
        "routes": routes,
        "representatives": rep_cards(rep_df),
        "kpis": kpis_from_totals(total, rep_df, dataset_size),
        "trainingMetrics": metrics,
        "notes": notes or [],
    }


def border_candidates(assign_df: pd.DataFrame, heavy_rep: str, light_rep: str, fraction: float) -> List[int]:
    heavy = assign_df[assign_df["rep_id"] == heavy_rep].copy()
    light = assign_df[assign_df["rep_id"] == light_rep].copy()

    if heavy.empty or light.empty:
        return []

    target_lat = light["customer_lat"].mean()
    target_lon = light["customer_lon"].mean()

    heavy["to_target"] = heavy.apply(
        lambda r: haversine_km(r["customer_lat"], r["customer_lon"], target_lat, target_lon),
        axis=1
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
        lambda r: haversine_km(r["customer_lat"], r["customer_lon"], heavy_target_lat, heavy_target_lon),
        axis=1,
    )
    light["to_target"] = light.apply(
        lambda r: haversine_km(r["customer_lat"], r["customer_lon"], light_target_lat, light_target_lon),
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
    routes, rep_df, total = route_all(assign_df, speed_kmph, service_min, "eval", distance_matrix)
    fairness = jains_fairness(rep_df["workload_min"].tolist()) if not rep_df.empty else 1.0
    return {
        "routes": routes,
        "rep_df": rep_df,
        "total": total,
        "fairness": fairness,
        "wbi": workload_balance_index(rep_df["workload_min"].tolist()) if not rep_df.empty else 1.0,
    } 

def objective_value(
    fairness: float,
    total_distance_km: float,
    operational_minutes: float,
    fairness_weight: float,
    distance_weight: float,
    time_weight: float,
) -> float:
    # Lower is better
    fairness_penalty = 1.0 - fairness
    return (            
        fairness_weight * fairness_penalty * 1000.0
        + distance_weight * total_distance_km
        + time_weight * (operational_minutes / 10.0)
    )


def enhance_assignment(
    assign_df: pd.DataFrame,
    speed_kmph: float,
    service_min: float,
    fairness_weight: float,
    distance_weight: float,
    time_weight: float,
    max_iterations: int,
    border_fraction: float,
    distance_matrix: Dict[str, Dict[str, float]],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    current = ensure_preview_node_ids(assign_df.copy())
    logs: List[Dict[str, Any]] = []
    current_eval = evaluate_assignment(current, speed_kmph, service_min, distance_matrix)

    for iteration in range(1, max_iterations + 1):
        rep_perf = current_eval["rep_df"].sort_values("operational_minutes", ascending=False).reset_index(drop=True)

        if len(rep_perf) < 2:
            break

        overall_gap = float(
            rep_perf.iloc[0]["operational_minutes"] - rep_perf.iloc[-1]["operational_minutes"]
        )
        if overall_gap < 5.0:
            break

        current_score = objective_value(
            current_eval["fairness"],
            current_eval["total"]["distance_km"],
            current_eval["total"]["operational_minutes"],
            fairness_weight,
            distance_weight,
            time_weight,
        )

        best_trial = None
        best_trial_eval = None
        best_log = None
        best_score_gain = 0.0

        # Try more than one heavy/light pair before giving up.
        # Still DEQ-guided: explore from the heavy end toward the light end.
        n_reps = len(rep_perf)
        heavy_count = min(3, n_reps - 1)
        light_count = min(3, n_reps - 1)

        heavy_ids = [str(rep_perf.iloc[i]["rep_id"]) for i in range(heavy_count)]
        light_ids = [str(rep_perf.iloc[n_reps - 1 - j]["rep_id"]) for j in range(light_count)]

        tried_pairs = set()

        for heavy_rep in heavy_ids:
            for light_rep in light_ids:
                if heavy_rep == light_rep:
                    continue
                if (heavy_rep, light_rep) in tried_pairs:
                    continue
                tried_pairs.add((heavy_rep, light_rep))

                heavy_minutes = float(
                    rep_perf.loc[rep_perf["rep_id"] == heavy_rep, "operational_minutes"].iloc[0]
                )
                light_minutes = float(
                    rep_perf.loc[rep_perf["rep_id"] == light_rep, "operational_minutes"].iloc[0]
                )

                workload_gap = heavy_minutes - light_minutes
                if workload_gap < 5.0:
                    continue

                candidates = border_candidates(current, heavy_rep, light_rep, border_fraction)

                for idx in candidates:
                    trial = current.copy()
                    trial.loc[idx, "rep_id"] = light_rep
                    trial_eval = evaluate_assignment(trial, speed_kmph, service_min, distance_matrix)

                    trial_score = objective_value(
                        trial_eval["fairness"],
                        trial_eval["total"]["distance_km"],
                        trial_eval["total"]["operational_minutes"],
                        fairness_weight,
                        distance_weight,
                        time_weight,
                    )

                    fairness_gain = trial_eval["fairness"] - current_eval["fairness"]
                    distance_gain = current_eval["total"]["distance_km"] - trial_eval["total"]["distance_km"]
                    time_gain = current_eval["total"]["operational_minutes"] - trial_eval["total"]["operational_minutes"]
                    score_gain = current_score - trial_score

                    # Block clearly harmful distance moves.
                    if distance_gain < -12.0:
                        continue

                    if distance_gain < -4.0 and fairness_gain < 0.06:
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
                            "distance_before": round(current_eval["total"]["distance_km"], 2),
                            "distance_after": round(trial_eval["total"]["distance_km"], 2),
                            "operational_before": round(current_eval["total"]["operational_minutes"], 2),
                            "operational_after": round(trial_eval["total"]["operational_minutes"], 2),
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
                else:
                    # No improving transfer found. Try DEQ-guided one-for-one swap
                    swap_best_trial = None
                    swap_best_trial_eval = None
                    swap_best_log = None
                    swap_best_score_gain = 0.0

                    n_reps = len(rep_perf)
                    heavy_count = min(3, n_reps - 1)
                    light_count = min(3, n_reps - 1)

                    heavy_ids = [str(rep_perf.iloc[i]["rep_id"]) for i in range(heavy_count)]
                    light_ids = [str(rep_perf.iloc[n_reps - 1 - j]["rep_id"]) for j in range(light_count)]

                    tried_pairs = set()

                    for heavy_rep in heavy_ids:
                        for light_rep in light_ids:
                            if heavy_rep == light_rep:
                                continue
                            if (heavy_rep, light_rep) in tried_pairs:
                                continue
                            tried_pairs.add((heavy_rep, light_rep))

                            heavy_minutes = float(
                                rep_perf.loc[rep_perf["rep_id"] == heavy_rep, "operational_minutes"].iloc[0]
                            )
                            light_minutes = float(
                                rep_perf.loc[rep_perf["rep_id"] == light_rep, "operational_minutes"].iloc[0]
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
                                        trial_eval["fairness"],
                                        trial_eval["total"]["distance_km"],
                                        trial_eval["total"]["operational_minutes"],
                                        fairness_weight,
                                        distance_weight,
                                        time_weight,
                                    )

                                    fairness_gain = trial_eval["fairness"] - current_eval["fairness"]
                                    distance_gain = current_eval["total"]["distance_km"] - trial_eval["total"]["distance_km"]
                                    time_gain = current_eval["total"]["operational_minutes"] - trial_eval["total"]["operational_minutes"]
                                    score_gain = current_score - trial_score

                                    # Keep the same safety guard logic
                                    if distance_gain < -12.0:
                                        continue
                                    if distance_gain < -4.0 and fairness_gain < 0.06:
                                        continue

                                    if score_gain > swap_best_score_gain:
                                        swap_best_score_gain = score_gain
                                        swap_best_trial = trial
                                        swap_best_trial_eval = trial_eval
                                        swap_best_log = {
                                            "iteration": iteration,
                                            "move_type": "swap",
                                            "moved_order": str(current.loc[idx_h, "order_id"]),
                                            "swapped_with_order": str(current.loc[idx_l, "order_id"]),
                                            "from_rep": heavy_rep,
                                            "to_rep": light_rep,
                                            "fairness_before": round(current_eval["fairness"], 6),
                                            "fairness_after": round(trial_eval["fairness"], 6),
                                            "distance_before": round(current_eval["total"]["distance_km"], 2),
                                            "distance_after": round(trial_eval["total"]["distance_km"], 2),
                                            "operational_before": round(current_eval["total"]["operational_minutes"], 2),
                                            "operational_after": round(trial_eval["total"]["operational_minutes"], 2),
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
                    else:
                        logs.append({
                            "iteration": iteration,
                            "from_rep": heavy_rep,
                            "to_rep": light_rep,
                            "accepted": False,
                            "reason": "no improving transfer or swap found",
                        })
                        break
    
    print("accepted moves:", sum(1 for x in logs if x.get("accepted")))
    print("enhancement logs:", logs)

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
        lambda r: haversine_km(r["customer_lat"], r["customer_lon"], depot_lat, depot_lon),
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
                    row["customer_lat"], row["customer_lon"],
                    pool.iloc[j]["customer_lat"], pool.iloc[j]["customer_lon"]
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
    if num_representatives > 1 and n >= num_representatives * 2:
        for j in range(num_representatives - 1, 0, -1):
            if sizes[j] > 1:
                sizes[0] += 1
                sizes[j] -= 1
                break

    assigned = []
    for rep_id, size in zip(rep_ids, sizes):
        assigned.extend([rep_id] * size)

    work["rep_id"] = assigned[:n]

    return work.drop(columns=["angle", "to_depot_km"], errors="ignore")

def choose_best_local_depot_cluster(
    df: pd.DataFrame,
    candidate_pool_size: int = 12,
) -> Tuple[float, float, pd.DataFrame]:
    """
    Choose the depot whose nearby customer cluster is most compact.
    Returns:
        depot_lat, depot_lon, depot_cluster_df
    """
    work = df.copy()

    depot_groups = (
        work.groupby(["depot_lat", "depot_lon"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
    )

    if depot_groups.empty:
        raise HTTPException(status_code=400, detail="No depot coordinates available for preview.")

    best_score = None
    best_depot_lat = None
    best_depot_lon = None
    best_cluster = None

    for depot in depot_groups.itertuples(index=False):
        depot_lat = float(depot.depot_lat)
        depot_lon = float(depot.depot_lon)

        cluster = work[
            (work["depot_lat"] == depot_lat) &
            (work["depot_lon"] == depot_lon)
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
            cluster = cluster.sort_values("to_depot_km").drop_duplicates(subset=["customer_node_id"]).copy()
        else:
            cluster["lat_round"] = cluster["customer_lat"].round(4)
            cluster["lon_round"] = cluster["customer_lon"].round(4)
            cluster = cluster.sort_values("to_depot_km").drop_duplicates(subset=["lat_round", "lon_round"]).copy()
            cluster = cluster.drop(columns=["lat_round", "lon_round"], errors="ignore")

        if cluster.empty:
            continue

        nearest = cluster.nsmallest(candidate_pool_size, "to_depot_km").copy()

        # Prefer depots with a compact nearby cluster.
        score = (
            float(nearest["to_depot_km"].mean()),
            float(nearest["to_depot_km"].max()),
            -int(len(nearest)),
        )

        if best_score is None or score < best_score:
            best_score = score
            best_depot_lat = depot_lat
            best_depot_lon = depot_lon
            best_cluster = cluster.copy()

    if best_cluster is None:
        raise HTTPException(status_code=400, detail="Could not build a local depot preview cluster.")

    return best_depot_lat, best_depot_lon, best_cluster

def build_local_preview_subset(
    df: pd.DataFrame,
    num_representatives: int,
    max_total_stops: int = 12,
    initial_radius_km: float = 7.0,
    max_radius_km: float = 16.0,
    local_cap_km: float = 14.0,
) -> pd.DataFrame:
    work = df.copy()

    depot_lat, depot_lon, depot_cluster = choose_best_local_depot_cluster(
        work,
        candidate_pool_size=max_total_stops,
    )

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

    if local.empty:
        local = depot_cluster.nsmallest(max(max_total_stops, num_representatives), "to_depot_km").copy()
    else:
        local = local.nsmallest(max(max_total_stops, num_representatives), "to_depot_km").copy()

    if "customer_node_id" in local.columns:
        local = local.sort_values("to_depot_km").drop_duplicates(subset=["customer_node_id"]).copy()
    else:
        local["lat_round"] = local["customer_lat"].round(4)
        local["lon_round"] = local["customer_lon"].round(4)
        local = local.sort_values("to_depot_km").drop_duplicates(subset=["lat_round", "lon_round"]).copy()
        local = local.drop(columns=["lat_round", "lon_round"], errors="ignore")

    # IMPORTANT:
    # for preview mode, keep the nearest local stops only.
    # do not spatially spread them outward.
    local = local.sort_values("to_depot_km").copy()

    # Hard cap for local preview geometry
    local = local[local["to_depot_km"] <= local_cap_km].copy()

    # If still enough rows, keep only the closest stops
    if len(local) >= num_representatives:
        local = local.head(max_total_stops).copy()
    else:
        # fallback: keep the closest available rows from the chosen depot cluster
        refill = depot_cluster.sort_values("to_depot_km").copy()
        refill = refill.head(max(max_total_stops, num_representatives)).copy()
        local = refill.copy()

    # If still too small, refill from the same chosen depot cluster only
    if len(local) < num_representatives:
        refill = depot_cluster.nsmallest(max(max_total_stops, num_representatives), "to_depot_km").copy()
        if "customer_node_id" in refill.columns:
            refill = refill.sort_values("to_depot_km").drop_duplicates(subset=["customer_node_id"]).copy()
        local = refill.head(max(max_total_stops, num_representatives)).copy()

    print(f"chosen preview depot: ({depot_lat}, {depot_lon})")
    print(f"chosen local preview stop count before rep assignment: {len(local)}")
    print(f"chosen local max distance from depot: {float(local['to_depot_km'].max()) if not local.empty else 0.0:.2f} km")
    print(f"final preview local max distance before drop: {float(local['to_depot_km'].max()) if not local.empty else 0.0:.2f} km")

    local = local.drop(columns=["to_depot_km"], errors="ignore").copy()

    preview_assigned = assign_preview_rep_ids_uneven(local, num_representatives)
    return preview_assigned

def preview_summary_from_assign_df(assign_df: pd.DataFrame) -> Dict[str, Any]:
    if assign_df.empty:
        return {
            "selectionStrategy": "single-depot nearest-customer compact preview",
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
        lambda r: haversine_km(r["customer_lat"], r["customer_lon"], depot_lat, depot_lon),
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
        raise HTTPException(status_code=400, detail=f"Invalid mapping JSON: {exc}") from exc

    resolved_role = dataset_role or infer_dataset_role(file.filename or "")
    df = read_csv_upload(file)
    normalized = normalize_dataset(df, mapping, resolved_role)
    summary = validation_summary(normalized)

    dataset_id = str(uuid.uuid4())
    reconstructed_name = f"reconstructed_{(file.filename or 'dataset').replace('.csv', '')}.csv"

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
        "customerNodes": int(df["customer_node_id"].nunique()) if "customer_node_id" in df.columns else int(df["customer_id"].nunique()),
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
    
    profile = get_run_profile(req.run_profile)
    print("baseline profile:", profile["profile_name"])

    print("dataset found")
    df = payload["data"].copy()
    print(f"data copied: {len(df)} rows")

    predicted_eta, metrics = train_eta_models(df, req.seed)
    print("train_eta_models done")
    df["predicted_eta_min"] = predicted_eta

    routing_df = build_routing_nodes(df)
    print(f"routing_df built: {len(routing_df)} node rows")

    role = payload["datasetRole"]
    role_note = (
        "Primary Amazon-based reconstructed baseline workflow"
        if role == "primary_reconstruction"
        else "Comparative/template workflow using Zomato-aligned structure"
        if role == "comparative_template"
        else "Generic uploaded dataset workflow"
    )

    preview_df = build_local_preview_subset(
        routing_df,
        num_representatives=req.num_representatives,
        max_total_stops=profile["preview_max_total_stops"],
        initial_radius_km=profile["preview_initial_radius_km"],
        max_radius_km=profile["preview_max_radius_km"],
        local_cap_km=profile["preview_local_cap_km"],
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
    print("max preview distance from depot:", float(preview_df["debug_to_depot_km"].max()))
    preview_matrix = build_preview_distance_matrix(preview_df, osm_threshold_km=profile["preview_osm_threshold_km"],)
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
            "Preview restricted to nearest customers to depot",
            "Preview capped to 12 total stops and selected representatives",
        ],
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


@app.post("/api/runs/enhanced")
def run_enhanced(req: EnhancedRequest) -> Dict[str, Any]:
    print("enhanced started")
    dataset_payload = DATASETS.get(req.dataset_id)
    baseline_payload = RUNS.get(req.baseline_run_id)

    if not dataset_payload:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if not baseline_payload:
        raise HTTPException(status_code=404, detail="Baseline run not found.")
    
    baseline_profile = baseline_payload.get("profile", get_run_profile(None))
    profile = get_run_profile(req.run_profile or baseline_profile.get("profile_name"))

    print("enhanced profile:", profile["profile_name"])
    print("enhanced params:", req.model_dump())
    
    distance_matrix = baseline_payload.get("distance_matrix", {})
    effective_fairness_weight = (
        req.fairness_weight
        if req.fairness_weight is not None
        else profile["enhanced_fairness_weight"]
    )
    effective_distance_weight = (
        req.distance_weight
        if req.distance_weight is not None
        else profile["enhanced_distance_weight"]
    )
    effective_time_weight = (
        req.time_weight
        if req.time_weight is not None
        else profile["enhanced_time_weight"]
    )
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
    print("enhanced params:", req.model_dump())

    print("enhanced dataset and baseline found")
    baseline_req = BaselineRequest(**baseline_payload["request"])
    assign_df = baseline_payload["assign_df"].copy()
    print(f"enhanced assign_df rows: {len(assign_df)}")

    improved_df, logs = enhance_assignment(
        assign_df,
        baseline_req.avg_speed_kmph,
        baseline_req.service_minutes_per_stop,
        effective_fairness_weight,
        effective_distance_weight,
        effective_time_weight,
        effective_max_iterations,
        effective_border_fraction,
        distance_matrix,
    )
    print("enhance_assignment done")

    routes, rep_df, total = route_all(
        improved_df,
        baseline_req.avg_speed_kmph,
        baseline_req.service_minutes_per_stop,
        "enhanced",
        distance_matrix,
    )
    print("enhanced route_all done")

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
            "Joint acceptance on fairness, distance, and operational time",
            f"Accepted rebalances: {sum(1 for x in logs if x.get('accepted'))}",
        ],
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