import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, fisher_exact


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = "participant_table.csv"

OUTPUT_DIR = "statistics_output"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

GROUP_COL = "group"

CONTINUOUS_VARIABLES = [
    "age",
    "postpartum_weeks",
    "children_count",

    # JDB / weekly normalized variables
    "seances",
    "clarite",
    "respect_consignes",
    "satisfaction_seance",
    "confiance",
    "recommandation",
    "n_formulaires_suivi",
    "completion_suivi",

    # Algorithm feedback
    "satisfaction_generale_algorithme",
    "nombre_problemes_algorithme",

    # Nolio
    "nombre_sessions_nolio",
]

BINARY_VARIABLES = [
    "reached_30min_ever",
]

# Optional threshold variables created from existing columns
CREATE_THRESHOLDS = True


# ============================================================
# SETUP
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)
if "group" not in df.columns:
    program = pd.read_csv("program_length.csv")
    program.columns = program.columns.str.strip()

    program["PID"] = pd.to_numeric(program["PID"], errors="coerce")
    program["group"] = program["group"].astype(str).str.strip().str.upper()

    df["PID"] = pd.to_numeric(df["PID"], errors="coerce")

    df = df.merge(
        program[["PID", "group"]],
        on="PID",
        how="left"
    )

df[GROUP_COL] = df[GROUP_COL].astype(str).str.strip().str.upper()

# Keep only A/B participants
df = df[df[GROUP_COL].isin(["A", "B"])].copy()


# ============================================================
# SIMPLE CLEANING
# ============================================================

for col in CONTINUOUS_VARIABLES:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for col in BINARY_VARIABLES:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
                "oui": True,
                "non": False,
            })
        )


# ============================================================
# OPTIONAL BINARY THRESHOLDS
# ============================================================

if CREATE_THRESHOLDS:
    if "completion_suivi" in df.columns:
        df["completion_suivi_ge80"] = df["completion_suivi"] >= 0.80
        BINARY_VARIABLES.append("completion_suivi_ge80")

    if "satisfaction_seance" in df.columns:
        df["satisfaction_seance_ge4"] = df["satisfaction_seance"] >= 4
        BINARY_VARIABLES.append("satisfaction_seance_ge4")

    if "satisfaction_generale_algorithme" in df.columns:
        df["satisfaction_algorithme_ge4"] = df["satisfaction_generale_algorithme"] >= 4
        BINARY_VARIABLES.append("satisfaction_algorithme_ge4")

    if "nombre_problemes_algorithme" in df.columns:
        df["any_algorithm_problem"] = df["nombre_problemes_algorithme"] > 0
        BINARY_VARIABLES.append("any_algorithm_problem")


# ============================================================
# DESCRIPTIVE STATISTICS BY GROUP
# ============================================================

descriptive_rows = []

for var in CONTINUOUS_VARIABLES:
    if var not in df.columns:
        continue

    for group in ["A", "B"]:
        values = df.loc[df[GROUP_COL] == group, var].dropna()

        descriptive_rows.append({
            "variable": var,
            "group": group,
            "n": len(values),
            "mean": values.mean(),
            "std": values.std(ddof=1),
            "median": values.median(),
            "q1": values.quantile(0.25),
            "q3": values.quantile(0.75),
            "min": values.min(),
            "max": values.max(),
        })

descriptive = pd.DataFrame(descriptive_rows)
descriptive.to_csv(os.path.join(OUTPUT_DIR, "descriptive_by_group.csv"), index=False)


# ============================================================
# MANN-WHITNEY U TESTS
# ============================================================

mann_rows = []

for var in CONTINUOUS_VARIABLES:
    if var not in df.columns:
        continue

    a = df.loc[df[GROUP_COL] == "A", var].dropna()
    b = df.loc[df[GROUP_COL] == "B", var].dropna()

    if len(a) == 0 or len(b) == 0:
        continue

    stat, p_value = mannwhitneyu(a, b, alternative="two-sided")

    mann_rows.append({
        "variable": var,

        "n_A": len(a),
        "mean_A": a.mean(),
        "std_A": a.std(ddof=1),
        "median_A": a.median(),

        "n_B": len(b),
        "mean_B": b.mean(),
        "std_B": b.std(ddof=1),
        "median_B": b.median(),

        "mann_whitney_U": stat,
        "p_value": p_value,
    })

mann_results = pd.DataFrame(mann_rows)
mann_results.to_csv(os.path.join(OUTPUT_DIR, "mann_whitney_results.csv"), index=False)


# ============================================================
# FISHER EXACT TESTS
# ============================================================

fisher_rows = []

for var in BINARY_VARIABLES:
    if var not in df.columns:
        continue

    sub = df[[GROUP_COL, var]].dropna().copy()

    if len(sub) == 0:
        continue

    a = sub.loc[sub[GROUP_COL] == "A", var].astype(bool)
    b = sub.loc[sub[GROUP_COL] == "B", var].astype(bool)

    a_true = int(a.sum())
    a_false = int(len(a) - a_true)

    b_true = int(b.sum())
    b_false = int(len(b) - b_true)

    table = [
        [a_true, a_false],
        [b_true, b_false],
    ]

    odds_ratio, p_value = fisher_exact(table, alternative="two-sided")

    fisher_rows.append({
        "variable": var,

        "A_true": a_true,
        "A_false": a_false,
        "A_total": len(a),
        "A_percent_true": 100 * a_true / len(a) if len(a) else np.nan,

        "B_true": b_true,
        "B_false": b_false,
        "B_total": len(b),
        "B_percent_true": 100 * b_true / len(b) if len(b) else np.nan,

        "odds_ratio_A_vs_B": odds_ratio,
        "p_value": p_value,
    })

fisher_results = pd.DataFrame(fisher_rows)
fisher_results.to_csv(os.path.join(OUTPUT_DIR, "fisher_results.csv"), index=False)


# ============================================================
# BOXPLOTS FOR CONTINUOUS VARIABLES
# ============================================================

for var in CONTINUOUS_VARIABLES:
    if var not in df.columns:
        continue

    a = df.loc[df[GROUP_COL] == "A", var].dropna()
    b = df.loc[df[GROUP_COL] == "B", var].dropna()

    if len(a) == 0 or len(b) == 0:
        continue

    plt.figure(figsize=(6, 5))

    plt.boxplot(
        [a, b],
        labels=["A", "B"],
        showmeans=True
    )

    # Add individual points
    x_a = np.random.normal(1, 0.04, size=len(a))
    x_b = np.random.normal(2, 0.04, size=len(b))

    plt.scatter(x_a, a, alpha=0.7)
    plt.scatter(x_b, b, alpha=0.7)

    plt.title(var)
    plt.xlabel("Group")
    plt.ylabel(var)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    safe_name = var.replace(" ", "_").replace("/", "_")
    plot_path = os.path.join(PLOTS_DIR, f"boxplot_{safe_name}.png")

    plt.savefig(plot_path, dpi=300)
    plt.close()


# ============================================================
# SIMPLE BARPLOTS FOR BINARY VARIABLES
# ============================================================

for var in BINARY_VARIABLES:
    if var not in df.columns:
        continue

    sub = df[[GROUP_COL, var]].dropna().copy()

    if len(sub) == 0:
        continue

    percentages = []

    for group in ["A", "B"]:
        values = sub.loc[sub[GROUP_COL] == group, var].astype(bool)

        if len(values) == 0:
            percentages.append(np.nan)
        else:
            percentages.append(100 * values.mean())

    plt.figure(figsize=(5, 5))

    plt.bar(["A", "B"], percentages)

    plt.title(var)
    plt.xlabel("Group")
    plt.ylabel("% True")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    safe_name = var.replace(" ", "_").replace("/", "_")
    plot_path = os.path.join(PLOTS_DIR, f"barplot_{safe_name}.png")

    plt.savefig(plot_path, dpi=300)
    plt.close()


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\nDone.")
print(f"Saved descriptive stats to: {OUTPUT_DIR}/descriptive_by_group.csv")
print(f"Saved Mann-Whitney results to: {OUTPUT_DIR}/mann_whitney_results.csv")
print(f"Saved Fisher results to: {OUTPUT_DIR}/fisher_results.csv")
print(f"Saved plots to: {PLOTS_DIR}")

print("\nMann-Whitney results:")
print(mann_results.to_string(index=False))

print("\nFisher exact results:")
print(fisher_results.to_string(index=False))