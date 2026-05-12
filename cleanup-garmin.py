import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. LOAD
# =========================
def load_sessions(path):
    df = pd.read_csv(path)

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])

    df = df.dropna(subset=["id", "date"])

    return df


# =========================
# 2. PROGRAM ALIGNMENT
# =========================
def assign_program_weeks(df):
    df = df.sort_values(["id", "date"]).copy()

    df["first_date"] = df.groupby("id")["date"].transform("min")
    df["days_since_start"] = (df["date"] - df["first_date"]).dt.days

    df["program_week"] = (df["days_since_start"] // 7) + 1

    return df


def assign_session_number(df):
    df = df.copy()

    df["session_in_week"] = df.groupby(["id", "program_week"]).cumcount() + 1
    df["session_global"] = df.groupby("id").cumcount() + 1

    return df


# =========================
# 3. CLEAN RUNNING DATA
# =========================
def filter_running(df):
    df = df.copy()

    df = df[df["session"].str.contains("course|running", case=False, na=False)]

    # 🔥 remove noise sessions
    df = df[df["max_continuous_run_min"] > 1]

    # 🔥 remove insane outliers
    df = df[df["max_continuous_run_min"] < 120]

    return df


# =========================
# 4. PLOTTING
# =========================
def plot_running_trajectories(df):
    df = filter_running(df)

    # 🔥 BEST performance per week
    weekly_perf = (
        df.groupby(["id", "program_week"])["max_continuous_run_min"]
        .max()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))

    for pid, group in weekly_perf.groupby("id"):
        plt.plot(
            group["program_week"],
            group["max_continuous_run_min"],
            alpha=0.4
        )

    plt.title("Progression de la course continue par participante")
    plt.xlabel("Semaine du programme")
    plt.ylabel("Durée maximale de course continue (minutes)")
    plt.grid()

    plt.savefig("running_trajectories.png", dpi=300)
    plt.close()


def plot_group_average(df, df_groups):
    df = filter_running(df)

    # 🔥 merge group info
    df = df.merge(df_groups, on="id", how="left")

    weekly_perf = (
        df.groupby(["id", "program_week", "group"])["max_continuous_run_min"]
        .max()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))

    for g in ["A", "B"]:
        subset = weekly_perf[weekly_perf["group"] == g]
        avg = subset.groupby("program_week")["max_continuous_run_min"].mean()

        plt.plot(avg.index, avg.values, marker='o', label=f"Groupe {g}")

    plt.title("Progression moyenne par groupe")
    plt.xlabel("Semaine du programme")
    plt.ylabel("Durée maximale de course continue (minutes)")
    plt.legend()
    plt.grid()

    plt.savefig("group_running_progression.png", dpi=300)
    plt.close()


# =========================
# 5. OUTLIER DEBUG
# =========================
def top_runners_per_week(df, top_n=3):
    df = filter_running(df)

    weekly_perf = (
        df.groupby(["id", "program_week"])["max_continuous_run_min"]
        .max()
        .reset_index()
    )

    weekly_perf = weekly_perf.sort_values(
        ["program_week", "max_continuous_run_min"],
        ascending=[True, False]
    )

    top = weekly_perf.groupby("program_week").head(top_n)

    return top


# =========================
# 6. MAIN
# =========================
def main():
    df = load_sessions("all_sessions.csv")

    df = assign_program_weeks(df)
    df = assign_session_number(df)

    # 🔥 OPTIONAL: load group info
    # df_groups must contain: id, group
    # Example:
    # df_groups = pd.read_csv("groups.csv")

    # plot_running_trajectories(df)
    # plot_group_average(df, df_groups)

    plot_running_trajectories(df)

    top = top_runners_per_week(df, top_n=3)
    print(top)

    # Save cleaned dataset
    df_clean = df[[
        "id",
        "program_week",
        "session_in_week",
        "session_global",
        "date",
        "total_duration_min",
        "max_continuous_run_min",
        "distance_km"
    ]]

    df_clean.to_csv("sessions_cleaned.csv", index=False)


if __name__ == "__main__":
    main()