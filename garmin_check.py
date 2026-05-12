import pandas as pd

def load_data():
    df_sessions = pd.read_csv("sessions_cleaned.csv")
    df_prog = pd.read_csv("program_length.csv")

    df_sessions["id"] = pd.to_numeric(df_sessions["id"], errors="coerce")
    df_prog["PID"] = pd.to_numeric(df_prog["PID"], errors="coerce")

    return df_sessions, df_prog


def build_garmin_summary(df_sessions, df_prog):
    # Per-participant Garmin stats
    agg = df_sessions.groupby("id").agg(
        n_sessions        = ("session_global", "count"),
        first_date        = ("date", "min"),
        last_date         = ("date", "max"),
        first_week        = ("program_week", "min"),
        last_week_recorded= ("program_week", "max"),
        n_weeks_with_data = ("program_week", "nunique"),
        garmin_max_run    = ("max_continuous_run_min", "max"),
        garmin_mean_run   = ("max_continuous_run_min", "mean"),
    ).reset_index().rename(columns={"id": "PID"})

    agg["garmin_mean_run"] = agg["garmin_mean_run"].round(2)
    agg["garmin_max_run"]  = agg["garmin_max_run"].round(2)
    agg["reached_30min"]   = agg["garmin_max_run"] >= 30

    # Merge with program info
    df = df_prog.merge(agg, on="PID", how="left")

    # Coverage: how many weeks of garmin data vs expected program length
    df["garmin_coverage"] = (
        df["n_weeks_with_data"] / df["program_length"]
    ).round(2)

    # Flag: did garmin data even start at week 1?
    df["starts_at_week_1"] = df["first_week"] == 1

    # Flag: does last recorded week match program length?
    df["garmin_reached_end"] = df["last_week_recorded"] >= df["program_length"]

    df["has_garmin"] = df["n_sessions"].notna()

    # Reorder columns nicely
    cols = [
        "PID", "group", "program_length",
        "has_garmin",
        "n_sessions", "n_weeks_with_data", "garmin_coverage",
        "first_date", "last_date",
        "first_week", "last_week_recorded",
        "starts_at_week_1", "garmin_reached_end",
        "garmin_max_run", "garmin_mean_run", "reached_30min"
    ]

    return df[cols].sort_values(["group", "PID"])


def main():
    df_sessions, df_prog = load_data()
    df_summary = build_garmin_summary(df_sessions, df_prog)

    print(df_summary.to_string(index=False))
    print("\n--- Coverage by group ---")
    print(df_summary.groupby("group")["has_garmin"].value_counts())
    print("\n--- Mean garmin coverage (participants with data) ---")
    print(df_summary[df_summary["has_garmin"]].groupby("group")["garmin_coverage"].mean())

    df_summary.to_csv("garmin_debug_summary.csv", index=False)
    print("\nSaved: garmin_debug_summary.csv")


if __name__ == "__main__":
    main()