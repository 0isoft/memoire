#!/usr/bin/env python3
"""
plot_garmin_progression.py

Plots Garmin-derived running progression over normalized program length.

Input:
- all-sessions-cleaned.csv or all-sessions.csv
  Must contain:
    id/PID, group, progress, progress_bin, include_in_garmin_analysis,
    max_continuous_run_min, estimated_running_time_min

Output:
- statistics/plots/garmin_median_max_run_by_progress.png
- statistics/plots/garmin_median_estimated_running_time_by_progress.png
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PLOTS_DIR = "statistics/plots"

GROUP_ORDER = ["A", "B"]
GROUP_COLORS = {
    "A": "blue",
    "B": "red",
}

PROGRESS_BIN_ORDER = ["0-25%", "25-50%", "50-75%", "75-100%"]
PROGRESS_BIN_X = {
    "0-25%": 0.125,
    "25-50%": 0.375,
    "50-75%": 0.625,
    "75-100%": 0.875,
}


def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_garmin(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    if "PID" not in df.columns and "id" in df.columns:
        df["PID"] = df["id"]

    df["PID"] = pd.to_numeric(df["PID"], errors="coerce")
    df = df.dropna(subset=["PID"])
    df["PID"] = df["PID"].astype(int)

    df["group"] = df["group"].astype(str).str.strip().str.upper()

    for col in [
        "progress",
        "program_week",
        "max_continuous_run_min",
        "estimated_running_time_min",
        "total_duration_min",
        "distance_km",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "include_in_garmin_analysis" in df.columns:
        df["include_in_garmin_analysis"] = (
            df["include_in_garmin_analysis"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
        df = df[df["include_in_garmin_analysis"] == True].copy()

    df = df[df["progress"].between(0, 1, inclusive="both")].copy()

    if "progress_bin" not in df.columns:
        df["progress_bin"] = pd.cut(
            df["progress"],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=PROGRESS_BIN_ORDER,
            include_lowest=True,
        )

    return df


def build_pid_bin_table(df, value_col):
    """
    One value per participant per progress bin.

    This avoids giving more weight to participants who uploaded many Garmin sessions.
    Within each PID/bin, we keep the maximum observed value for that bin,
    because the question is progression capacity, not average session noise.
    """

    pid_bin = (
        df.dropna(subset=[value_col, "progress_bin"])
        .groupby(["PID", "group", "progress_bin"], observed=True)[value_col]
        .max()
        .reset_index()
    )

    return pid_bin


def plot_group_median_progression(df, value_col, ylabel, filename, title):
    pid_bin = build_pid_bin_table(df, value_col)

    plt.figure(figsize=(10, 6))

    for group in GROUP_ORDER:
        sub = pid_bin[pid_bin["group"] == group].copy()

        grouped = (
            sub.groupby("progress_bin", observed=True)[value_col]
            .agg(
                median="median",
                q1=lambda x: x.quantile(0.25),
                q3=lambda x: x.quantile(0.75),
                n="count",
            )
            .reindex(PROGRESS_BIN_ORDER)
        )

        x = np.array([PROGRESS_BIN_X[b] for b in PROGRESS_BIN_ORDER])
        y = grouped["median"].to_numpy(dtype=float)
        q1 = grouped["q1"].to_numpy(dtype=float)
        q3 = grouped["q3"].to_numpy(dtype=float)

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2.5,
            label=f"Groupe {group} median",
            color=GROUP_COLORS.get(group),
        )

        plt.fill_between(
            x,
            q1,
            q3,
            alpha=0.15,
            color=GROUP_COLORS.get(group),
        )

        # Show individual participant-bin values faintly.
        for pid, pid_df in sub.groupby("PID"):
            pid_df = pid_df.copy()
            pid_df["x"] = pid_df["progress_bin"].astype(str).map(PROGRESS_BIN_X)

            plt.plot(
                pid_df["x"],
                pid_df[value_col],
                color=GROUP_COLORS.get(group),
                alpha=0.18,
                linewidth=1,
            )

        # Optional n labels.
        for bin_label, row in grouped.iterrows():
            if pd.notna(row["median"]):
                plt.text(
                    PROGRESS_BIN_X[bin_label],
                    row["median"] + 0.6,
                    f"n={int(row['n'])}",
                    ha="center",
                    fontsize=8,
                    color=GROUP_COLORS.get(group),
                )

    plt.xlabel("Progression normalisée dans le programme")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(
        [PROGRESS_BIN_X[b] for b in PROGRESS_BIN_ORDER],
        PROGRESS_BIN_ORDER,
    )
    plt.xlim(0, 1)
    plt.grid()
    plt.legend()
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


def print_progression_table(df, value_col):
    pid_bin = build_pid_bin_table(df, value_col)

    table = (
        pid_bin
        .groupby(["group", "progress_bin"], observed=True)[value_col]
        .agg(
            n="count",
            median="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
            mean="mean",
        )
        .reset_index()
    )

    print(f"\n--- Progression table: {value_col} ---")
    print(table.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Plot Garmin progression over normalized program length."
    )

    parser.add_argument(
        "--garmin",
        default="all-sessions-cleaned.csv",
        help="Path to cleaned Garmin sessions CSV.",
    )

    args = parser.parse_args()

    ensure_dirs()

    df = load_garmin(args.garmin)

    print(f"Loaded Garmin sessions: {len(df)}")
    print("Participants with Garmin data:")
    print(df.groupby("group")["PID"].nunique())

    plot_group_median_progression(
        df=df,
        value_col="max_continuous_run_min",
        ylabel="Durée maximale de course continue dans le bin (min)",
        filename="garmin_median_max_run_by_progress.png",
        title="Progression Garmin — durée maximale de course continue par progression normalisée",
    )

    print_progression_table(df, "max_continuous_run_min")

    if "estimated_running_time_min" in df.columns:
        plot_group_median_progression(
            df=df,
            value_col="estimated_running_time_min",
            ylabel="Temps de course estimé dans le bin (min)",
            filename="garmin_median_estimated_running_time_by_progress.png",
            title="Progression Garmin — temps de course estimé par progression normalisée",
        )

        print_progression_table(df, "estimated_running_time_min")


if __name__ == "__main__":
    main()