from __future__ import annotations
from typing import Any, Dict, List, Optional

import math
import uuid

import numpy as np
import pandas as pd
from pydantic import BaseModel
from fastapi import HTTPException

EARTH_RADIUS_KM = 6371.0


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
    direct = haversine_km(lat1, lon1, lat2, lon2)
    return direct * 1.25


def parse_order_date_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()

    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)

    fallback_mask = parsed.isna()
    if fallback_mask.any():
        parsed.loc[fallback_mask] = pd.to_datetime(
            text.loc[fallback_mask], errors="coerce"
        )

    if parsed.notna().any():
        return parsed.dt.normalize()
    return parsed


def _base_reconstruct_from_mapping(
    df: pd.DataFrame, mapping: FieldMapping
) -> pd.DataFrame:
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


def finalize_reconstructed_dataset(out: pd.DataFrame) -> pd.DataFrame:
    """Apply common post-processing to a reconstructed dataset.

    Adds `customer_node_id`, `customer_name`, `node_order_count`,
    `direct_depot_customer_km`, `is_distance_outlier`, `is_routing_eligible`,
    and fills `rating` and `observed_eta_min` defaults when missing.

    This function centralises steps shared by dataset-specific reconstructors.
    """
    # Create stable customer node ids based on depot + rounded coords
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

    return out


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
    df: pd.DataFrame, mapping: FieldMapping, source_role: str
) -> FieldMapping:
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
            "customers": (
                int(df["customer_node_id"].nunique())
                if "customer_node_id" in df.columns
                else int(df["customer_id"].nunique())
            ),
            "orders": int(df["order_id"].nunique()),
            "avgRating": round(avg_rating, 2),
        },
    }


def ensure_preview_node_ids(assign_df: pd.DataFrame) -> pd.DataFrame:
    work = assign_df.copy().reset_index(drop=True)

    if "customer_node_id" in work.columns:
        work["node_id"] = work["customer_node_id"].astype(str)
    elif "node_id" not in work.columns:
        work["node_id"] = [f"CUST-{i + 1}" for i in range(len(work))]

    return work


def choose_best_local_depot_cluster(
    df: pd.DataFrame,
    candidate_pool_size: int = 12,
    prefer_agent_coverage: bool = False,
    min_agents: int = 1,
) -> tuple[float, float, pd.DataFrame]:
    from fastapi import HTTPException

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

        if best_score is None or score < best_score:
            best_score = score
            best_depot_lat = depot_lat
            best_depot_lon = depot_lon
            best_cluster = cluster.copy()

    if best_cluster is None:
        raise HTTPException(
            status_code=400, detail="Could not build a local depot preview cluster."
        )

    return best_depot_lat, best_depot_lon, best_cluster


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

    for i in range(rem):
        sizes[i] += 1

    idx = 0
    assignments = []
    for i, s in enumerate(sizes):
        if s <= 0:
            continue
        group = work.iloc[idx : idx + s].copy()
        group["rep_id"] = rep_ids[i]
        assignments.append(group)
        idx += s

    if assignments:
        out = pd.concat(assignments, ignore_index=True)
    else:
        out = work.copy()
        out["rep_id"] = rep_ids[0] if rep_ids else "REP-1"

    out = out.drop(columns=["angle", "to_depot_km"], errors="ignore")
    return out.reset_index(drop=True)
