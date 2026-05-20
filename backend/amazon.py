from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from helpers import (
    haversine_km,
    _base_reconstruct_from_mapping,
    finalize_reconstructed_dataset,
    choose_best_local_depot_cluster,
    assign_preview_rep_ids_uneven,
)
from typing import Callable, Tuple, List


def amazon_distance_polish_assignment(
    assign_df: pd.DataFrame,
    speed_kmph: float,
    service_min: float,
    distance_matrix: Dict[str, Dict[str, float]],
    evaluate_fn: Callable,
    max_iterations: int = 12,
    min_distance_gain_km: float = 0.25,
    min_fairness_floor: float = 0.995,
    max_wbi_increase: float = 0.0,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    current = assign_df.copy().reset_index(drop=True)
    logs: List[Dict[str, Any]] = []
    current_eval = evaluate_fn(current, speed_kmph, service_min, distance_matrix)

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
                trial_eval = evaluate_fn(
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

        # 2) Try swaps too
        for idx_a in range(len(current)):
            rep_a = str(current.loc[idx_a, "rep_id"])
            for idx_b in range(idx_a + 1, len(current)):
                rep_b = str(current.loc[idx_b, "rep_id"])
                if rep_a == rep_b:
                    continue

                trial = current.copy()
                trial.loc[idx_a, "rep_id"] = rep_b
                trial.loc[idx_b, "rep_id"] = rep_a
                trial_eval = evaluate_fn(
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


def build_amazon_order_routing_rows(df: pd.DataFrame) -> pd.DataFrame:
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
        from fastapi import HTTPException

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

    out["customer_node_id"] = (
        out["physical_customer_node_id"].astype(str)
        + "-ORDER-"
        + out["order_id"].astype(str)
    )
    out["node_name"] = out["customer_name"].astype(str)

    return out.reset_index(drop=True)


def reconstruct_raw_amazon_dataset(df: pd.DataFrame, mapping: Any) -> pd.DataFrame:
    out = _base_reconstruct_from_mapping(df, mapping)

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
    out = finalize_reconstructed_dataset(out)

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


def assign_preview_rep_ids_from_agent(
    preview_df: pd.DataFrame,
    num_representatives: int,
    max_total_stops: Optional[int] = None,
    strict_existing_agents: bool = False,
    cap_total_stops: bool = True,
) -> pd.DataFrame:
    # Move Amazon-specific agent-based preview assignment here so app.py can delegate.
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
            return assign_preview_rep_ids_uneven(work, num_representatives)
        top_agents = unique_agents[:num_representatives]

    filtered = valid[valid["agent_id"].isin(top_agents)].copy()

    if len(filtered) < num_representatives:
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
    max_customers_per_rep: Optional[int] = None,
) -> pd.DataFrame:
    # Amazon-specific preview builder moved from app.py
    work = df.copy()

    effective_reps = max(6, int(num_representatives))
    per_rep_cap = int(max_customers_per_rep or 3)
    per_rep_cap = max(1, per_rep_cap)

    target_preview_rows = effective_reps * per_rep_cap

    depot_lat, depot_lon, depot_cluster = choose_best_local_depot_cluster(
        work,
        candidate_pool_size=max(target_preview_rows * 4, effective_reps * 8),
        prefer_agent_coverage=True,
        min_agents=effective_reps,
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

    while len(local) < target_preview_rows and radius < max_radius_km:
        radius *= 1.5
        local = depot_cluster[depot_cluster["to_depot_km"] <= radius].copy()

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

                selected = (
                    selected.sort_values(["rep_id", "to_depot_km"])
                    .groupby("rep_id", group_keys=False)
                    .head(per_rep_cap)
                    .reset_index(drop=True)
                )
                return selected.drop(
                    columns=["to_depot_km", "_pred_eta_sort", "_rating_sort"],
                    errors="ignore",
                ).reset_index(drop=True)

    local = (
        local.head(target_preview_rows)
        .drop(columns=["to_depot_km"], errors="ignore")
        .copy()
    )
    preview_assigned = assign_preview_rep_ids_uneven(local, effective_reps)
    return preview_assigned
