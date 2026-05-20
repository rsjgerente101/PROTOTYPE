from __future__ import annotations
from typing import Any

import pandas as pd

from helpers import (
    _base_reconstruct_from_mapping,
    haversine_km,
    finalize_reconstructed_dataset,
)


def reconstruct_raw_zomato_dataset(df: pd.DataFrame, mapping: Any) -> pd.DataFrame:
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

    out = finalize_reconstructed_dataset(out)

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
