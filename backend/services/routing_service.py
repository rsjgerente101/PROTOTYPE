from typing import Any, Dict, List, Tuple
import pandas as pd
import uuid
from helpers import ensure_preview_node_ids, haversine_km


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
            key=lambda r: float(
                distance_matrix.get(current_node, {}).get(str(r["node_id"]), 0.0)
            ),
        )

        leg = float(
            distance_matrix.get(current_node, {}).get(str(best["node_id"]), 0.0)
        )
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

    return_leg = float(distance_matrix.get(current_node, {}).get("DEPOT", 0.0))
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
    customers: List[Any],
) -> pd.DataFrame:
    work = ensure_preview_node_ids(assign_df.copy())

    if work.empty:
        raise ValueError("Baseline preview is empty.")

    depot_lat = float(work.iloc[0]["depot_lat"])
    depot_lon = float(work.iloc[0]["depot_lon"])
    depot_id = str(work.iloc[0]["depot_id"])

    rows: List[Dict[str, Any]] = []

    for idx, customer in enumerate(customers, start=1):
        customer_number = getattr(customer, "customer_number", None) or (100000 + idx)
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
                "agent_id": getattr(customer, "assigned_rep", None) or "UNASSIGNED",
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
                "rep_id": getattr(customer, "assigned_rep", None) or "UNASSIGNED",
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
        "travel_minutes": (
            float(rep_df["travel_minutes"].sum()) if not rep_df.empty else 0.0
        ),
        "operational_minutes": (
            float(rep_df["operational_minutes"].sum()) if not rep_df.empty else 0.0
        ),
    }
    return routes, rep_df, total
