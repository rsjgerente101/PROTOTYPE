from __future__ import annotations
from typing import List, Tuple

import pandas as pd

from schemas import AddedCustomerPayload
from helpers import (
    haversine_km,
    ensure_preview_node_ids as helpers_ensure_preview_node_ids,
)
from .routing_service import append_added_customers_to_assign_df


def ensure_preview_node_ids(assign_df: pd.DataFrame) -> pd.DataFrame:
    return helpers_ensure_preview_node_ids(assign_df)


def assign_new_customer_to_nearest_rep(
    assign_df: pd.DataFrame,
    customer_lat: float,
    customer_lon: float,
) -> str:
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


def process_added_customers(
    assign_df: pd.DataFrame, customers: List[AddedCustomerPayload]
) -> Tuple[pd.DataFrame, List[AddedCustomerPayload]]:
    """
    Assign added customers sequentially to nearest reps and append them
    to the assignment DataFrame. Returns (updated_assign_df, resolved_customers).
    """
    resolved_customers: List[AddedCustomerPayload] = []
    updated_assign_df = assign_df.copy()

    for customer in customers:
        assigned_rep = assign_new_customer_to_nearest_rep(
            updated_assign_df, float(customer.lat), float(customer.lon)
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
    return updated_assign_df, resolved_customers
