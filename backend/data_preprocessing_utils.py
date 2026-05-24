import os
from pathlib import Path
from typing import List, Dict, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize_selected_features(
    df: pd.DataFrame, columns_to_normalize: List[str]
) -> pd.DataFrame:
    out = df.copy()
    valid_cols = [c for c in columns_to_normalize if c in out.columns]

    if not valid_cols:
        return out

    scaler = MinMaxScaler()

    for col in valid_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    mask = out[valid_cols].notna().all(axis=1)
    if mask.any():
        out.loc[mask, valid_cols] = scaler.fit_transform(out.loc[mask, valid_cols])

    return out


def save_feature_distribution_plots(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    columns: List[str],
    output_dir: str = "artifacts/feature_distributions",
    prefix: str = "dataset",
) -> Dict[str, Any]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    generated: Dict[str, Any] = {
        "before_histograms": {},
        "after_histograms": {},
        "boxplots": {},
    }

    for col in columns:
        if col not in df_before.columns or col not in df_after.columns:
            continue

        before = pd.to_numeric(df_before[col], errors="coerce").dropna()
        after = pd.to_numeric(df_after[col], errors="coerce").dropna()

        if before.empty or after.empty:
            continue

        plt.figure(figsize=(7, 4))
        plt.hist(before, bins=30, edgecolor="black")
        plt.title(f"Unnormalized Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        before_path = os.path.join(output_dir, f"{prefix}_{col}_before.png")
        plt.tight_layout()
        plt.savefig(before_path, dpi=150)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.hist(after, bins=30, edgecolor="black")
        plt.title(f"Normalized Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        after_path = os.path.join(output_dir, f"{prefix}_{col}_after.png")
        plt.tight_layout()
        plt.savefig(after_path, dpi=150)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.boxplot([before, after], tick_labels=["Before", "After"])
        plt.title(f"Before vs After Normalization: {col}")
        plt.ylabel(col)
        box_path = os.path.join(output_dir, f"{prefix}_{col}_boxplot.png")
        plt.tight_layout()
        plt.savefig(box_path, dpi=150)
        plt.close()

        generated["before_histograms"][col] = before_path
        generated["after_histograms"][col] = after_path
        generated["boxplots"][col] = box_path

    return generated
