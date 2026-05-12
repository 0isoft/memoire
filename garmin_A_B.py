import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. LOAD
# =========================
def load_data():
    df_sessions = pd.read_csv("sessions_cleaned.csv")
    df_prog = pd.read_csv("program_length.csv")

    df_sessions["id"] = pd.to_numeric(df_sessions["id"], errors="coerce")
    df_prog["PID"] = pd.to_numeric(df_prog["PID"], errors="coerce")

    df_prog = df_prog.rename(columns={"PID": "id"})

    return df_sessions, df_prog


# =========================
# 2. MERGE + NORMALIZE
# =========================
def prepare(df_sessions, df_prog):
    df = df_sessions.merge(df_prog, on="id", how="left")

    # remove missing program lengths
    df = df[df["program_length"].notna()]

    # 🔥 normalized progression
    df["progress"] = df["program_week"] / df["program_length"]

    # 🔥 bin progression
    df["progress_bin"] = pd.cut(
        df["progress"],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=["0-25%", "25-50%", "50-75%", "75-100%"]
    )

    return df


# =========================
# 3. AGGREGATION
# =========================
def aggregate(df):
    df_weekly = (
        df.groupby(["id", "program_week", "group"])
        .agg({
            "max_continuous_run_min": "max",
            "progress": "first",        # ✅ keep normalized x
            "progress_bin": "first"     # (optional but clean)
        })
        .reset_index()
    )

    return df_weekly


# =========================
# 4. PLOT
# =========================
def plot_group_progression(df_weekly):
    plt.figure(figsize=(10, 6))

    for g in ["A", "B"]:
        subset = df_weekly[df_weekly["group"] == g]

        grouped = (
            subset.groupby("progress_bin")["max_continuous_run_min"]
            .median()
        )

        plt.plot(
            grouped.index.astype(str),
            grouped.values,
            marker='o',
            label=f"Groupe {g}"
        )

    plt.title("Progression normalisée de la course continue (médiane)")
    plt.xlabel("Progression dans le programme (%)")
    plt.ylabel("Durée maximale de course continue (minutes)")
    plt.legend()
    plt.grid()

    plt.savefig("normalized_running_progression.png", dpi=300)
    plt.close()

def plot_individual_progression(df_weekly):
    plt.figure(figsize=(10, 6))

    # 🔥 CUT everything beyond program completion
    df_weekly = df_weekly[df_weekly["progress"] <= 1.0]
    df_weekly = df_weekly[df_weekly["id"] != 6302]

    colors = {"A": "blue", "B": "red"}

    for pid, group in df_weekly.groupby("id"):
        g = group["group"].iloc[0]

        group = group.sort_values("program_week")

        plt.plot(
            group["progress"],
            group["max_continuous_run_min"],
            color=colors.get(g, "gray"),
            alpha=0.25
        )

    # --- median overlay ---
    for g in ["A", "B"]:
        subset = df_weekly[df_weekly["group"] == g]

        grouped = (
            subset.groupby("progress_bin")["max_continuous_run_min"]
            .median()
        )

        x_map = {
            "0-25%": 0.125,
            "25-50%": 0.375,
            "50-75%": 0.625,
            "75-100%": 0.875
        }

        x = [x_map[str(k)] for k in grouped.index]

        plt.plot(
            x,
            grouped.values,
            marker='o',
            linewidth=3,
            label=f"Groupe {g} (médiane)",
            color=colors[g]
        )

    plt.title("Progression individuelle normalisée (toutes participantes)")
    plt.xlabel("Progression dans le programme (0 → 1)")
    plt.ylabel("Durée maximale de course continue (minutes)")
    plt.legend()
    plt.grid()

    plt.savefig("individual_normalized_progression.png", dpi=300)
    plt.close()


# =========================
# 5. MAIN
# =========================
def main():
    df_sessions, df_prog = load_data()

    df = prepare(df_sessions, df_prog)

    df_weekly = aggregate(df)

    top_B = (
    df_weekly[df_weekly["group"] == "B"]
    .sort_values(["program_week", "max_continuous_run_min"], ascending=[True, False])
    .groupby("program_week")
    .head(3)
)

    print(top_B)

    plot_group_progression(df_weekly)
    plot_individual_progression(df_weekly)

    print("Plot saved: normalized_running_progression.png")
    print("Plot saved: normalized_running_progression.png")



if __name__ == "__main__":
    main()