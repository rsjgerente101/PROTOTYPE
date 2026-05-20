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
from fastapi.responses import StreamingResponse, PlainTextResponse
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

from schemas import (
    BaselineRequest,
    EnhancedRequest,
    AddedCustomerPayload,
    BaselineAddCustomersRequest,
)

from config import (
    EARTH_RADIUS_KM,
    OSM_CACHE_DIR,
    RUN_PROFILES,
    DEMO_PREVIEW_DEPOTS,
    MIN_FIXED_DEMO_NODES,
    MIN_FIXED_DEMO_AGENTS,
    AMAZON_FIXED_DEMO_NODES,
    AMAZON_FIXED_DEMO_AGENTS,
    AMAZON_DEFAULT_REPRESENTATIVES,
    AMAZON_MAX_CUSTOMERS_PER_REP,
    AMAZON_MIN_PREVIEW_STOPS,
)

# Bring in shared helpers and dataset-specific handlers
from helpers import (
    FieldMapping,
    parse_order_date_series as helpers_parse_order_date_series,
    infer_dataset_role as helpers_infer_dataset_role,
    role_label as helpers_role_label,
    autofill_mapping_from_known_columns as helpers_autofill_mapping_from_known_columns,
    validation_summary as helpers_validation_summary,
    _base_reconstruct_from_mapping as helpers_base_reconstruct,
    finalize_reconstructed_dataset as helpers_finalize_reconstructed,
    haversine_km,
    road_adjusted_km,
    ensure_preview_node_ids as helpers_ensure_preview_node_ids,
    choose_best_local_depot_cluster as helpers_choose_best_local_depot_cluster,
    assign_preview_rep_ids_uneven as helpers_assign_preview_rep_ids_uneven,
)

import amazon
import zomato
from services.metrics_service import (
    compute_thesis_priority_scores,
    jains_fairness,
    workload_balance_index,
    rep_cards,
    kpis_from_totals,
)
from services.routing_service import (
    route_one_rep,
    route_all,
    append_added_customers_to_assign_df,
)
from services.osm_service import (
    build_preview_points,
    load_or_build_osm_preview_graph,
    snap_preview_points_to_osm,
    build_preview_distance_matrix,
    load_or_build_osm_preview_graphs,
    build_snapped_point_lookup,
    path_coords_from_osm,
    build_display_leg_path,
    attach_route_display_geometry,
)
from services.enhancement_service import (
    enhance_assignment,
    evaluate_assignment,
)
from services.add_customer_service import (
    process_added_customers,
    assign_new_customer_to_nearest_rep as _assign_new_customer_to_nearest_rep,
)

# Pydantic models and configuration constants have been moved to
# backend/schemas.py and backend/config.py respectively.


# OSM and distance/matrix utilities moved to backend/osm_service.py


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
    return helpers_ensure_preview_node_ids(assign_df)


def matrix_cost(matrix: Dict[str, Dict[str, float]], a: str, b: str) -> float:
    if a == b:
        return 0.0
    return float(matrix.get(a, {}).get(b, 0.0))


def parse_order_date_series(series: pd.Series) -> pd.Series:
    return helpers_parse_order_date_series(series)


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
    return amazon.build_amazon_order_routing_rows(df)


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
    return helpers_infer_dataset_role(filename)


def role_label(role: str) -> str:
    return helpers_role_label(role)


def autofill_mapping_from_known_columns(
    df: pd.DataFrame,
    mapping: FieldMapping,
    source_role: str,
) -> FieldMapping:
    """
    Backend safety net so raw uploads normalize consistently even if the frontend
    did not send some optional mapped fields.
    """
    return helpers_autofill_mapping_from_known_columns(df, mapping, source_role)


def _base_reconstruct_from_mapping(
    df: pd.DataFrame, mapping: FieldMapping
) -> pd.DataFrame:
    return helpers_base_reconstruct(df, mapping)


def reconstruct_raw_amazon_dataset(
    df: pd.DataFrame, mapping: FieldMapping
) -> pd.DataFrame:
    """
    Reconstruct raw Amazon upload into the cleaned route-eligible schema
    aligned with the known-good reconstructed Amazon dataset design.
    """
    return amazon.reconstruct_raw_amazon_dataset(df, mapping)


def reconstruct_raw_zomato_dataset(
    df: pd.DataFrame, mapping: FieldMapping
) -> pd.DataFrame:
    """
    Reconstruct raw Zomato upload into the cleaned route-eligible schema
    aligned with the same node-aware routing structure.
    """
    return zomato.reconstruct_raw_zomato_dataset(df, mapping)


def reconstruct_generic_uploaded_dataset(
    df: pd.DataFrame, mapping: FieldMapping
) -> pd.DataFrame:
    """
    Generic fallback for other uploaded delivery datasets.
    Keeps behavior simple but still produces the cleaned route-eligible schema.
    """
    out = helpers_base_reconstruct(df, mapping)

    out = helpers_finalize_reconstructed(out)

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
    return helpers_validation_summary(df)


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


# Routing helpers moved to backend/routing_service.py


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
        "travel_minutes": (
            float(rep_df["travel_minutes"].sum()) if not rep_df.empty else 0.0
        ),
        "operational_minutes": (
            float(rep_df["operational_minutes"].sum()) if not rep_df.empty else 0.0
        ),
    }
    return routes, rep_df, total


# KPI and fairness implementations moved to backend/metrics_service.py


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


# Enhancement logic (DEQ) moved to backend/enhancement_service.py


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
    return amazon.amazon_distance_polish_assignment(
        assign_df,
        speed_kmph,
        service_min,
        distance_matrix,
        evaluate_fn=evaluate_assignment,
        max_iterations=max_iterations,
        min_distance_gain_km=min_distance_gain_km,
        min_fairness_floor=min_fairness_floor,
        max_wbi_increase=max_wbi_increase,
    )


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
    return amazon.assign_preview_rep_ids_from_agent(
        preview_df,
        num_representatives,
        max_total_stops=max_total_stops,
        strict_existing_agents=strict_existing_agents,
        cap_total_stops=cap_total_stops,
    )


def assign_preview_rep_ids_uneven(
    preview_df: pd.DataFrame,
    num_representatives: int,
) -> pd.DataFrame:
    return helpers_assign_preview_rep_ids_uneven(preview_df, num_representatives)
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
    return helpers_choose_best_local_depot_cluster(
        df,
        candidate_pool_size=candidate_pool_size,
        prefer_agent_coverage=prefer_agent_coverage,
        min_agents=min_agents,
    )


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
    return amazon.build_local_preview_subset_amazon(
        df,
        num_representatives,
        max_total_stops=max_total_stops,
        initial_radius_km=initial_radius_km,
        max_radius_km=max_radius_km,
        local_cap_km=local_cap_km,
        use_existing_agents=use_existing_agents,
        strict_existing_agents=strict_existing_agents,
        min_nodes_per_rep=min_nodes_per_rep,
        max_customers_per_rep=max_customers_per_rep,
    )


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
        return PlainTextResponse("Dataset not found", status_code=404)

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
        "customerNodes": (
            int(df["customer_node_id"].nunique())
            if "customer_node_id" in df.columns
            else int(df["customer_id"].nunique())
        ),
        "orders": int(df["order_id"].nunique()),
        "depot": depot,
    }


@app.get("/api/datasets/{dataset_id}/reconstructed")
def download_reconstructed_dataset(dataset_id: str):
    payload = DATASETS.get(dataset_id)
    if not payload:
        return PlainTextResponse("Dataset not found", status_code=404)

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
        return PlainTextResponse("Dataset not found", status_code=404)

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
        else (
            "Comparative/template workflow using Zomato-aligned structure"
            if role == "comparative_template"
            else "Generic uploaded dataset workflow"
        )
    )

    preview_max_total_stops = (
        max(profile["preview_max_total_stops"], 40)
        if role == "comparative_template"
        else (
            max(profile["preview_max_total_stops"], AMAZON_MIN_PREVIEW_STOPS)
            if role == "primary_reconstruction"
            else profile["preview_max_total_stops"]
        )
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

    updated_assign_df, resolved_customers = process_added_customers(
        assign_df, req.customers
    )

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
    return _assign_new_customer_to_nearest_rep(assign_df, customer_lat, customer_lon)


@app.post("/api/runs/enhanced")
def run_enhanced(req: EnhancedRequest) -> Dict[str, Any]:
    print("enhanced started")
    dataset_payload = DATASETS.get(req.dataset_id)
    baseline_payload = RUNS.get(req.baseline_run_id)

    if not dataset_payload:
        return PlainTextResponse("Dataset not found", status_code=404)
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
        else (
            "Enhanced DEQ run for comparative/template Zomato evaluation"
            if role == "comparative_template"
            else "Enhanced DEQ run over uploaded dataset"
        )
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
