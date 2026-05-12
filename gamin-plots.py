import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, fisher_exact


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = "all-sessions-cleaned.csv"

OUT_WEEKLY = "garmin_weekly_metrics.csv"
OUT_PARTICIPANT = "garmin_participant_metrics.csv"
OUT_REPORT = "garmin_group_report.txt"

OUTLIER_IDS = []  # keep empty for now because all-sessions-cleaned.csv already excludes suspect sessions


# ============================================================
# LOAD
# ============================================================

def load_clean_sessions(path):
    df = pd.read_csv(path)

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["program_week"] = pd.to_numeric(df["program_week"], errors="coerce")
    df["progress"] = pd.to_numeric(df["progress"], errors="coerce")
    df["program_length"] = pd.to_numeric(df["program_length"], errors="coerce")
    df["max_continuous_run_min"] = pd.to_numeric(df["max_continuous_run_min"], errors="coerce")
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
    df["valid_time_ratio"] = pd.to_numeric(df["valid_time_ratio"], errors="coerce")
    df["running_time_ratio"] = pd.to_numeric(df["running_time_ratio"], errors="coerce")

    df["group"] = df["group"].astype(str).str.strip()

    df = df.dropna(subset=[
        "id",
        "group",
        "program_week",
        "progress",
        "max_continuous_run_min"
    ])

    # Keep only in-program normalized timeline
    df = df[(df["progress"] >= 0) & (df["progress"] <= 1.0)]

    # Optional manual exclusion layer
    if OUTLIER_IDS:
        df = df[~df["id"].isin(OUTLIER_IDS)]

    return df


# ============================================================
# WEEKLY + PARTICIPANT METRICS
# ============================================================

def build_weekly_metrics(df):
    weekly = (
        df.groupby(["id", "group", "program_week", "progress_bin"], observed=False)
        .agg(
            progress=("progress", "first"),
            weekly_best_run=("max_continuous_run_min", "max"),
            weekly_median_run=("max_continuous_run_min", "median"),
            weekly_mean_run=("max_continuous_run_min", "mean"),
            weekly_total_distance_km=("distance_km", "sum"),
            weekly_n_sessions=("max_continuous_run_min", "count"),
            weekly_mean_valid_time_ratio=("valid_time_ratio", "mean"),
            weekly_mean_running_time_ratio=("running_time_ratio", "mean"),
        )
        .reset_index()
    )

    return weekly


def build_participant_metrics(weekly):
    participant = (
        weekly.sort_values(["id", "program_week"])
        .groupby(["id", "group"])
        .agg(
            n_weeks_with_garmin=("program_week", "nunique"),
            first_week=("program_week", "min"),
            last_week=("program_week", "max"),
            garmin_max_weekly_best=("weekly_best_run", "max"),
            garmin_median_weekly_best=("weekly_best_run", "median"),
            garmin_mean_weekly_best=("weekly_best_run", "mean"),
            garmin_last_weekly_best=("weekly_best_run", "last"),
            garmin_total_distance_km=("weekly_total_distance_km", "sum"),
            garmin_mean_weekly_distance_km=("weekly_total_distance_km", "mean"),
            garmin_total_sessions=("weekly_n_sessions", "sum"),
            mean_valid_time_ratio=("weekly_mean_valid_time_ratio", "mean"),
            mean_running_time_ratio=("weekly_mean_running_time_ratio", "mean"),
        )
        .reset_index()
    )

    participant["reached_30min_ever"] = participant["garmin_max_weekly_best"] >= 30
    participant["reached_30min_last"] = participant["garmin_last_weekly_best"] >= 30

    return participant


# ============================================================
# PLOTS
# ============================================================

def plot_normalized_group_median_iqr(weekly):
    plt.figure(figsize=(10, 6))

    colors = {"A": "blue", "B": "red"}

    # Bin-level summary across participant-weeks
    for g in ["A", "B"]:
        subset = weekly[weekly["group"] == g].copy()

        summary = (
            subset.groupby("progress_bin", observed=False)["weekly_best_run"]
            .agg(
                median="median",
                q1=lambda x: x.quantile(0.25),
                q3=lambda x: x.quantile(0.75),
            )
            .reset_index()
        )

        x_labels = summary["progress_bin"].astype(str)
        x = range(len(summary))

        plt.plot(
            x,
            summary["median"],
            marker="o",
            linewidth=2.5,
            label=f"Groupe {g} médiane",
            color=colors[g]
        )

        plt.fill_between(
            x,
            summary["q1"],
            summary["q3"],
            alpha=0.15,
            color=colors[g]
        )

    plt.xticks(range(4), ["0-25%", "25-50%", "50-75%", "75-100%"])
    plt.title("Progression Garmin normalisée par groupe")
    plt.xlabel("Progression dans le programme")
    plt.ylabel("Meilleure durée de course continue hebdomadaire (min)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot_normalized_group_median_iqr.png", dpi=300)
    plt.close()


def plot_individual_normalized_progression(weekly):
    plt.figure(figsize=(10, 6))

    colors = {"A": "blue", "B": "red"}

    for pid, gdf in weekly.groupby("id"):
        group = gdf["group"].iloc[0]
        gdf = gdf.sort_values("progress")

        plt.plot(
            gdf["progress"],
            gdf["weekly_best_run"],
            color=colors.get(group, "gray"),
            alpha=0.25,
            linewidth=1
        )

    # Overlay group medians by bin
    x_map = {
        "0-25%": 0.125,
        "25-50%": 0.375,
        "50-75%": 0.625,
        "75-100%": 0.875
    }

    for g in ["A", "B"]:
        subset = weekly[weekly["group"] == g]

        grouped = (
            subset.groupby("progress_bin", observed=False)["weekly_best_run"]
            .median()
            .reset_index()
        )

        x = grouped["progress_bin"].astype(str).map(x_map)

        plt.plot(
            x,
            grouped["weekly_best_run"],
            marker="o",
            linewidth=3,
            color=colors[g],
            label=f"Groupe {g} médiane"
        )

    plt.title("Progression individuelle normalisée")
    plt.xlabel("Progression dans le programme (0 à 1)")
    plt.ylabel("Meilleure durée de course continue hebdomadaire (min)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot_individual_normalized_progression.png", dpi=300)
    plt.close()


def plot_group_boxplots_by_progress_bin(weekly):
    bins = ["0-25%", "25-50%", "50-75%", "75-100%"]

    data = []
    labels = []

    for b in bins:
        for g in ["A", "B"]:
            values = weekly[
                (weekly["progress_bin"].astype(str) == b) &
                (weekly["group"] == g)
            ]["weekly_best_run"].dropna()

            data.append(values)
            labels.append(f"{b}\n{g}")

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=labels, showfliers=True)

    plt.title("Distribution des performances Garmin par phase du programme")
    plt.xlabel("Progression / groupe")
    plt.ylabel("Meilleure durée de course continue hebdomadaire (min)")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot_group_boxplots_by_progress_bin.png", dpi=300)
    plt.close()


def plot_reached_30min_by_group(participant):
    rates = (
        participant.groupby("group")["reached_30min_ever"]
        .mean()
        .reindex(["A", "B"])
    )

    counts = (
        participant.groupby("group")["reached_30min_ever"]
        .count()
        .reindex(["A", "B"])
    )

    plt.figure(figsize=(6, 5))
    plt.bar(rates.index, rates.values)

    for i, g in enumerate(rates.index):
        plt.text(
            i,
            rates.loc[g] + 0.02,
            f"n={counts.loc[g]}",
            ha="center"
        )

    plt.ylim(0, 1)
    plt.title("Proportion de participantes ayant atteint 30 min")
    plt.xlabel("Groupe")
    plt.ylabel("Proportion")

    plt.tight_layout()
    plt.savefig("plot_reached_30min_by_group.png", dpi=300)
    plt.close()


# ============================================================
# REPORT
# ============================================================

def describe_by_group(participant):
    summary = (
        participant.groupby("group")
        .agg(
            n_participants=("id", "nunique"),
            median_max_weekly_best=("garmin_max_weekly_best", "median"),
            q1_max_weekly_best=("garmin_max_weekly_best", lambda x: x.quantile(0.25)),
            q3_max_weekly_best=("garmin_max_weekly_best", lambda x: x.quantile(0.75)),
            mean_max_weekly_best=("garmin_max_weekly_best", "mean"),
            median_typical_weekly_best=("garmin_median_weekly_best", "median"),
            median_last_weekly_best=("garmin_last_weekly_best", "median"),
            reached_30min_rate=("reached_30min_ever", "mean"),
            median_weeks_with_garmin=("n_weeks_with_garmin", "median"),
            median_total_sessions=("garmin_total_sessions", "median"),
            median_total_distance_km=("garmin_total_distance_km", "median"),
        )
        .round(3)
    )

    return summary


def run_group_tests(participant):
    results = []

    outcomes = [
        "garmin_max_weekly_best",
        "garmin_median_weekly_best",
        "garmin_last_weekly_best",
        "garmin_total_distance_km",
        "n_weeks_with_garmin",
        "garmin_total_sessions",
    ]

    A = participant[participant["group"] == "A"]
    B = participant[participant["group"] == "B"]

    for outcome in outcomes:
        a = A[outcome].dropna()
        b = B[outcome].dropna()

        if len(a) >= 2 and len(b) >= 2:
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            results.append({
                "outcome": outcome,
                "A_median": a.median(),
                "B_median": b.median(),
                "A_n": len(a),
                "B_n": len(b),
                "p_mannwhitney": p
            })

    test_df = pd.DataFrame(results)

    # Fisher exact test for reached_30min_ever
    fisher_text = "Fisher exact test not computed."

    table = pd.crosstab(
        participant["group"],
        participant["reached_30min_ever"]
    )

    try:
        table = table.reindex(index=["A", "B"], columns=[False, True], fill_value=0)
        oddsratio, p = fisher_exact(table)
        fisher_text = (
            "Reached 30 min Fisher exact test:\n"
            f"{table.to_string()}\n"
            f"odds ratio = {oddsratio:.3f}, p = {p:.4f}"
        )
    except Exception as e:
        fisher_text = f"Fisher exact test failed: {e}"

    return test_df, fisher_text


def write_report(df_sessions, weekly, participant):
    group_summary = describe_by_group(participant)
    test_df, fisher_text = run_group_tests(participant)

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("Garmin A/B Group Performance Report\n")
        f.write("===================================\n\n")

        f.write("Dataset and assumptions\n")
        f.write("-----------------------\n")
        f.write(f"Input file: {INPUT_PATH}\n")
        f.write("The analysis uses all-sessions-cleaned.csv, i.e. Garmin sessions already marked as suitable for analysis.\n")
        f.write("Sessions are normalized to program progression using the available week alignment in the dataset.\n")
        f.write("For this analysis, first Garmin date is accepted as a valid proxy for program start based on study oversight.\n")
        f.write("Performance is summarized at participant-week level before group comparison, avoiding raw session-level over-weighting.\n\n")

        f.write("Sample size\n")
        f.write("-----------\n")
        f.write(f"Clean session rows: {len(df_sessions)}\n")
        f.write(f"Participant-week rows: {len(weekly)}\n")
        f.write(f"Participants with usable Garmin data: {participant['id'].nunique()}\n\n")

        f.write("Participants by group\n")
        f.write("---------------------\n")
        f.write(participant.groupby("group")["id"].nunique().to_string())
        f.write("\n\n")

        f.write("Group summary\n")
        f.write("-------------\n")
        f.write(group_summary.to_string())
        f.write("\n\n")

        f.write("Mann-Whitney group comparisons\n")
        f.write("------------------------------\n")
        if not test_df.empty:
            f.write(test_df.round(4).to_string(index=False))
        else:
            f.write("No tests available.")
        f.write("\n\n")

        f.write("30-minute threshold comparison\n")
        f.write("------------------------------\n")
        f.write(fisher_text)
        f.write("\n\n")

        f.write("Interpretation notes\n")
        f.write("--------------------\n")
        f.write(
            "The main Garmin outcome should be interpreted as objective performance among participants "
            "with usable Garmin data, not necessarily as a perfect estimate of all randomized participants.\n"
        )
        f.write(
            "Median participant-level and participant-week metrics are preferred over raw means because Garmin data "
            "are sparse, repeated, and sensitive to outliers.\n"
        )
        f.write(
            "If group B appears stronger than expected, this should be checked against Garmin coverage, number of sessions, "
            "and participant-level distributions rather than interpreted directly from raw averages.\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    df = load_clean_sessions(INPUT_PATH)

    weekly = build_weekly_metrics(df)
    participant = build_participant_metrics(weekly)

    weekly.to_csv(OUT_WEEKLY, index=False)
    participant.to_csv(OUT_PARTICIPANT, index=False)

    plot_normalized_group_median_iqr(weekly)
    plot_individual_normalized_progression(weekly)
    plot_group_boxplots_by_progress_bin(weekly)
    plot_reached_30min_by_group(participant)

    write_report(df, weekly, participant)

    print(f"Saved: {OUT_WEEKLY}")
    print(f"Saved: {OUT_PARTICIPANT}")
    print(f"Saved: {OUT_REPORT}")
    print("Saved plots:")
    print("- plot_normalized_group_median_iqr.png")
    print("- plot_individual_normalized_progression.png")
    print("- plot_group_boxplots_by_progress_bin.png")
    print("- plot_reached_30min_by_group.png")

    print("\nParticipant-level summary:")
    print(describe_by_group(participant))


if __name__ == "__main__":
    main()