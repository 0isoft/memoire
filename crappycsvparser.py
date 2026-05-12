import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. LOAD + CLEAN
# =========================


def load_and_clean(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    cols = [
        "Votre PID",
        "A quelle semaine du programme êtes-vous arrivés ?",
        "Votre ressentis sur les 7 derniers jours : [J'ai pu réaliser les séances prévues]",
        "Votre ressentis sur les 7 derniers jours : [Les explications fournis étaient claires]",
        "Votre ressentis sur les 7 derniers jours : [J'ai respecté l'intensité/les consignes]",
        "Votre ressentis sur les 7 derniers jours : [Satisfaction globale de la séance]",
        "Votre ressentis sur les 7 derniers jours : [Je me sens en confiance pour poursuivre le programme]",
        "Votre ressentis sur les 7 derniers jours : [Je recommanderais ces séances à une amie]"
    ]

    df = df[cols].copy()

    df.columns = [
        "PID",
        "week",
        "sessions_done",
        "clarity",
        "intensity_respected",
        "satisfaction",
        "confidence",
        "recommendation"
    ]

    # Clean PID
    df["PID"] = df["PID"].astype(str).str.replace("PP-", "", regex=False)
    df["PID"] = pd.to_numeric(df["PID"], errors="coerce")

    # Numeric conversion
    df["week"] = pd.to_numeric(df["week"], errors="coerce")

    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["PID", "week"])

    return df

def load_garmin(path):
    df = pd.read_csv(path)

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["program_week"] = pd.to_numeric(df["program_week"], errors="coerce")
    df["max_continuous_run_min"] = pd.to_numeric(
        df["max_continuous_run_min"],
        errors="coerce"
    )

    df = df.dropna(subset=["id", "program_week", "max_continuous_run_min"])

    df = df.rename(columns={"id": "PID"})

    return df

def load_signup(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df_signup = df[[
        "Colonne 1",
        "Groupe",
        "Votre âge",
        "à combien de semaines de post-partum êtes-vous ?",
        "Combien d'enfants avec vous ?",
        "Quel a été le mode d'accouchement de votre dernier accouchement ?",
        "Avez-vous eu une déchirure lors de votre accouchement ? Et si oui, de quel grade ?"
    ]].copy()

    df_signup.columns = [
        "PID",
        "group",
        "age",
        "postpartum_weeks",
        "children_count",
        "delivery_mode",
        "tear_grade"
    ]

    df_signup["PID"] = df_signup["PID"].astype(str).str.replace("PP-", "", regex=False)
    df_signup["PID"] = pd.to_numeric(df_signup["PID"], errors="coerce")

    df_signup["group"] = df_signup["group"].astype(str).str.strip()
    df_signup["age"] = pd.to_numeric(df_signup["age"], errors="coerce")
    df_signup["postpartum_weeks"] = pd.to_numeric(df_signup["postpartum_weeks"], errors="coerce")

    df_signup["children_count"] = (
        df_signup["children_count"]
        .astype(str)
        .str.replace("4 ou plus", "4", regex=False)
    )
    df_signup["children_count"] = pd.to_numeric(df_signup["children_count"], errors="coerce")

    df_signup = df_signup.dropna(subset=["PID", "group"])

    return df_signup


def build_garmin_analysis(df_garmin):
    df_garmin = df_garmin.copy()

    df_pid = df_garmin.groupby("PID").agg({
        "max_continuous_run_min": ["max", "mean"],
        "program_week": "max"
    })

    df_pid.columns = [
        "garmin_max_run",
        "garmin_mean_run",
        "garmin_last_week"
    ]

    df_pid = df_pid.reset_index()

    # Dernière semaine Garmin observée par participante
    last_week = (
        df_garmin.sort_values(["PID", "program_week"])
        .groupby("PID")
        .tail(1)[["PID", "max_continuous_run_min"]]
        .rename(columns={"max_continuous_run_min": "garmin_last_run"})
    )

    df_pid = df_pid.merge(last_week, on="PID", how="left")

    df_pid["reached_30min_ever"] = df_pid["garmin_max_run"] >= 30
    df_pid["reached_30min_last"] = df_pid["garmin_last_run"] >= 30

    return df_pid

# =========================
# 2. FEATURE ENGINEERING
# =========================
def compute_adherence(df):
    df = df.copy()
    df["adherence"] = df["sessions_done"] / 5
    return df


# =========================
# 3. AGGREGATION (FIX DUPLICATES HERE)
# =========================
def aggregate_weekly(df):
    # 🔥 THIS FIXES YOUR ERROR
    df_agg = df.groupby(["PID", "week"])["adherence"].mean().reset_index()
    return df_agg


# =========================
# 4. PLOTTING
# =========================

def plot_group_questions_normalized(df):
    questions = [
        ("sessions_done", "J'ai pu réaliser les séances prévues"),
        ("clarity", "Les explications fournies étaient claires"),
        ("intensity_respected", "J'ai respecté l'intensité / les consignes"),
        ("satisfaction", "Satisfaction globale de la séance"),
        ("confidence", "Je me sens en confiance pour poursuivre le programme"),
        ("recommendation", "Je recommanderais ces séances à une amie")
    ]

    df_plot = df.groupby(["PID", "progress_bin", "group"]).mean(numeric_only=True).reset_index()

    for col, question_text in questions:
        plt.figure()

        for g in ["A", "B"]:
            subset = df_plot[df_plot["group"] == g]
            grouped = subset.groupby("progress_bin")[col].mean()

            plt.plot(
                grouped.index.astype(str),
                grouped.values,
                marker='o',
                label=f"Groupe {g}"
            )

        plt.title(question_text)
        plt.xlabel("Progression dans le programme (%)")
        plt.ylabel("Score (0–5)")
        plt.ylim(0, 5)
        plt.legend()
        plt.grid()

        plt.savefig(f"{col}_group_normalized.png", dpi=300)
        plt.close()

def plot_mean_questions_normalized(df):
    questions = [
        ("sessions_done", "J'ai pu réaliser les séances prévues"),
        ("clarity", "Les explications fournies étaient claires"),
        ("intensity_respected", "J'ai respecté l'intensité / les consignes"),
        ("satisfaction", "Satisfaction globale de la séance"),
        ("confidence", "Je me sens en confiance pour poursuivre le programme"),
        ("recommendation", "Je recommanderais ces séances à une amie")
    ]

    df_plot = df.groupby(["PID", "progress_bin"]).mean(numeric_only=True).reset_index()

    for col, question_text in questions:
        plt.figure()

        grouped = df_plot.groupby("progress_bin")[col].mean()

        plt.plot(grouped.index.astype(str), grouped.values, marker='o')

        plt.title(question_text)
        plt.xlabel("Progression dans le programme (%)")
        plt.ylabel("Score (0–5)")
        plt.ylim(0, 5)
        plt.grid()

        plt.savefig(f"{col}_normalized.png", dpi=300)
        plt.close()


def plot_participants(df):
    participants = df.groupby("week")["PID"].nunique()

    plt.figure()
    participants.plot(marker='o')
    plt.title("Active participants per week")
    plt.xlabel("Week")
    plt.ylabel("Number of participants")
    plt.grid()
    plt.savefig("participants.png", dpi=300)
    plt.close()
def plot_participants_by_group(df):
    # Count unique participants per week per group
    counts = df.groupby(["week", "group"])["PID"].nunique().reset_index()

    plt.figure()

    for g in ["A", "B"]:
        subset = counts[counts["group"] == g]

        plt.plot(
            subset["week"],
            subset["PID"],
            marker='o',
            label=f"Groupe {g}"
        )

    plt.title("Nombre de participantes actives par semaine (par groupe)")
    plt.xlabel("Semaine du programme")
    plt.ylabel("Nombre de participantes")
    plt.legend()
    plt.grid()

    plt.savefig("participants_by_group.png", dpi=300)
    plt.close()

def plot_trajectories(df):
    pivot = df.pivot(index="week", columns="PID", values="adherence")

    plt.figure()
    pivot.plot(legend=False, alpha=0.3)
    plt.title("Individual adherence trajectories")
    plt.xlabel("Week")
    plt.ylabel("Adherence")
    plt.savefig("trajectories.png", dpi=300)
    plt.close()


# =========================
# 5. PIPELINE
# =========================


def load_signup(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df_signup = df[[
        "Colonne 1",
        "Votre âge",
        "à combien de semaines de post-partum êtes-vous ?",
        "Combien d'enfants avec vous ?",
        "Quel a été le mode d'accouchement de votre dernier accouchement ?",
        "Avez-vous eu une déchirure lors de votre accouchement ? Et si oui, de quel grade ?"
    ]].copy()

    df_signup.columns = [
        "PID",
        "age",
        "postpartum_weeks",
        "children_count",
        "delivery_mode",
        "tear_grade"
    ]

    df_signup["PID"] = df_signup["PID"].astype(str).str.replace("PP-", "", regex=False)
    df_signup["PID"] = pd.to_numeric(df_signup["PID"], errors="coerce")

    df_signup["age"] = pd.to_numeric(df_signup["age"], errors="coerce")
    df_signup["postpartum_weeks"] = pd.to_numeric(df_signup["postpartum_weeks"], errors="coerce")

    df_signup["children_count"] = (
        df_signup["children_count"]
        .astype(str)
        .str.replace("4 ou plus", "4", regex=False)
    )
    df_signup["children_count"] = pd.to_numeric(df_signup["children_count"], errors="coerce")

    df_signup = df_signup.dropna(subset=["PID"])

    return df_signup


def build_participants(df):
    participants = {}

    for pid, group_df in df.groupby("PID"):
        group_df = group_df.sort_values("week")

        participant = {}

        participant["group"] = group_df["group"].iloc[0]
        participant["age"] = group_df["age"].iloc[0]
        participant["program_length"] = group_df["program_length"].iloc[0]

        participant["max_week"] = group_df["week"].max()

        if pd.notna(participant["program_length"]):
            participant["completion_rate"] = (
                participant["max_week"] / participant["program_length"]
            )
        else:
            participant["completion_rate"] = None

        # 🔥 THIS is key
        participant["data"] = group_df[[
            "week",
            "progress",   # ← normalized time
            "sessions_done",
            "clarity",
            "intensity_respected",
            "satisfaction",
            "confidence",
            "recommendation"
        ]]

        participants[pid] = participant

    return participants

def load_program_length(path):
    df = pd.read_csv(path)

    df["PID"] = pd.to_numeric(df["PID"], errors="coerce")
    df["program_length"] = pd.to_numeric(df["program_length"], errors="coerce")

    # 🔥 THIS is important
    df["group"] = df["group"].astype(str).str.strip()

    return df


def build_analysis_df(df_questionnaire, df_garmin_analysis=None):
    df_analysis = df_questionnaire.groupby("PID").agg({
        "group": "first",
        "age": "first",
        "postpartum_weeks": "first",
        "children_count": "first",
        "delivery_mode": "first",
        "tear_grade": "first",
        "program_length": "first",
        "week": "max",
        "sessions_done": "mean",
        "clarity": "mean",
        "intensity_respected": "mean",
        "satisfaction": "mean",
        "confidence": "mean",
        "recommendation": "mean"
    }).reset_index()

    df_analysis["completed"] = (
        df_analysis["week"] >= df_analysis["program_length"]
    )

    df_analysis["completion_rate"] = (
        df_analysis["week"] / df_analysis["program_length"]
    )

    if df_garmin_analysis is not None:
        df_analysis = df_analysis.merge(
            df_garmin_analysis,
            on="PID",
            how="left"
        )

    return df_analysis

def main():

    # 1. Load
    df = load_and_clean("data2.csv")
    df_signup = load_signup("start.csv")
    df_prog = load_program_length("program_length.csv")

    # 2. Merge all sources
    df = df.merge(df_prog, on="PID", how="left")     # group comes here
    df = df.merge(df_signup, on="PID", how="left")   # extra metadata
    
    # 🔍 sanity check
    print(df.columns)

    print("Group values:")
    print(df["group"].value_counts(dropna=False))

    # 3. Keep valid program lengths
    df = df[df["program_length"].notna()]

    # 4. Normalize
    df["progress"] = df["week"] / df["program_length"]

    df["progress_bin"] = pd.cut(
        df["progress"],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=["0-25%", "25-50%", "50-75%", "75-100%"]
    )

    # 5. Build participants
    participants = build_participants(df)

    pid = list(participants.keys())[0]
    print(participants[pid])

    # 6. Plots
    plot_participants_by_group(df)
    plot_mean_questions_normalized(df)
    plot_group_questions_normalized(df)
    plot_participants(df)

    # 7. Adherence (optional)
    df_adherence = compute_adherence(df)
    df_agg = aggregate_weekly(df_adherence)

    plot_trajectories(df_agg)

    df_analysis = build_analysis_df(df)

    A = df_analysis[df_analysis["group"] == "A"]
    B = df_analysis[df_analysis["group"] == "B"]
    from scipy.stats import chi2_contingency

    table = [
        [A["completed"].sum(), len(A) - A["completed"].sum()],
        [B["completed"].sum(), len(B) - B["completed"].sum()]
    ]

    chi2, p, _, _ = chi2_contingency(table)

    print("Completion p-value:", p)
    from scipy.stats import mannwhitneyu

    stat, p = mannwhitneyu(
        A["completion_rate"].dropna(),
        B["completion_rate"].dropna()
    )

    print("Completion rate p-value:", p)

    questions = [
    "sessions_done",
    "clarity",
    "intensity_respected",
    "satisfaction",
    "confidence",
    "recommendation"
]

    for q in questions:
        stat, p = mannwhitneyu(
            A[q].dropna(),
            B[q].dropna()
        )
        print(f"{q}: p = {p}")
    for q in questions:
        print(f"{q}: A = {A[q].mean():.2f}, B = {B[q].mean():.2f}")

    df_garmin = load_garmin("sessions_cleaned.csv")
    df_garmin_analysis = build_garmin_analysis(df_garmin)

    df_analysis = build_analysis_df(df, df_garmin_analysis)

    print(df_analysis.head())
    df_analysis.to_csv("final_analysis_table.csv", index=False)

    from scipy.stats import mannwhitneyu

    A = df_analysis[df_analysis["group"] == "A"]
    B = df_analysis[df_analysis["group"] == "B"]

    stat, p = mannwhitneyu(
        A["garmin_max_run"].dropna(),
        B["garmin_max_run"].dropna(),
        alternative="two-sided"
    )

    print("Garmin max run p-value:", p)
    print("A mean:", A["garmin_max_run"].mean())
    print("B mean:", B["garmin_max_run"].mean())

    from scipy.stats import chi2_contingency

    table = [
        [A["reached_30min_ever"].sum(), A["reached_30min_ever"].notna().sum() - A["reached_30min_ever"].sum()],
        [B["reached_30min_ever"].sum(), B["reached_30min_ever"].notna().sum() - B["reached_30min_ever"].sum()]
    ]

    chi2, p, _, _ = chi2_contingency(table)
    print("Reached 30 min p-value:", p)

    import statsmodels.formula.api as smf
    model = smf.ols(
        "garmin_max_run ~ C(group) + age + postpartum_weeks + program_length + children_count",
        data=df_analysis
    ).fit()

    print(model.summary())


    

if __name__ == "__main__":
    main()