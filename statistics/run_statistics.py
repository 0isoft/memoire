import os
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, fisher_exact


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "statistics/output"
PLOTS_DIR = "statistics/plots"

GROUP_ORDER = ["A", "B"]

GROUP_LABELS = {
    "A": "Groupe A - algorithme personnalisé",
    "B": "Groupe B - programme assigné",
}

GROUP_COLORS = {
    "A": "blue",
    "B": "red",
}

H1_FEASIBILITY_BINARY = [
    "has_weekly_forms",
    "completed_by_forms",
    "has_final_feedback",
    "has_garmin_data",
]

H1_FEASIBILITY_CONTINUOUS = [
    "n_weeks_reported",
    "forms_coverage",
    "forms_completion_rate",
]

H1_ADHERENCE = [
    "mean_adherence_ratio",
    "mean_sessions_done",
    "forms_coverage",
]

H1_WEEKLY_ACCEPTABILITY = [
    "clarity_mean",
    "intensity_respected_mean",
    "satisfaction_mean",
    "confidence_mean",
    "recommendation_mean",
]

H1_FINAL_ACCEPTABILITY = [
    "mean_final_score",
    "global_use_score",
    "process_score",
    "confidence_score",
    "recommendation_score",
]

H1_ISSUES_BINARY = [
    "any_reported_issue",
    "safety_advice_clear",
]

H1_ISSUES_CONTINUOUS = [
    "n_reported_issues",
]

ISSUE_CATEGORY_COLS = [
    "issue_functional_tests",
    "issue_criteria",
    "issue_program_choice",
    "issue_test_ease",
    "issue_characteristics",
    "issue_symptoms",
    "issue_contraindications",
]

H2_CONTINUOUS = [
    "mean_adherence_ratio",
    "forms_coverage",
    "forms_completion_rate",
    "n_weeks_reported",
    "satisfaction_mean",
    "confidence_mean",
    "recommendation_mean",
    "mean_final_score",
    "global_use_score",
    "process_score",
    "confidence_score",
    "recommendation_score",
]

H2_BINARY = [
    "completed_by_forms",
    "any_reported_issue",
    "safety_advice_clear",
]

GARMIN_CONTINUOUS = [
    "n_garmin_sessions_included",
    "garmin_max_progress",
    "garmin_total_distance_km",
    "garmin_total_running_time_min",
    "garmin_mean_run",
    "garmin_max_run",
    "garmin_last_run",
]

GARMIN_BINARY = [
    "has_garmin_data",
    "reached_25min_ever",
    "reached_30min_ever",
]


# ============================================================
# HELPERS
# ============================================================

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def save_csv(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def save_text(text, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved: {path}")


def clean_bool(series):
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
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
            "nan": np.nan,
            "": np.nan,
        })
    )


def iqr(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan, np.nan, np.nan

    q1 = x.quantile(0.25)
    med = x.quantile(0.50)
    q3 = x.quantile(0.75)

    return med, q1, q3


def format_median_iqr(x, digits=2):
    med, q1, q3 = iqr(x)

    if pd.isna(med):
        return "NA"

    return f"{med:.{digits}f} [{q1:.{digits}f}–{q3:.{digits}f}]"


def cliffs_delta(x, y):
    """
    Cliff's delta effect size.
    Positive value means x tends to be greater than y.
    Here we usually call x = Group B, y = Group A,
    so positive = B higher than A.
    """
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy()
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna().to_numpy()

    if len(x) == 0 or len(y) == 0:
        return np.nan

    greater = 0
    lower = 0

    for xi in x:
        greater += np.sum(xi > y)
        lower += np.sum(xi < y)

    return (greater - lower) / (len(x) * len(y))


def odds_ratio_from_table(table):
    """
    Fisher exact returns inf if cells are zero.
    This function keeps the raw odds ratio.
    """
    try:
        odds_ratio, p = fisher_exact(table, alternative="two-sided")
        return odds_ratio, p
    except Exception:
        return np.nan, np.nan


def available_columns(df, cols):
    return [c for c in cols if c in df.columns]


# ============================================================
# LOAD DATA
# ============================================================

def load_core(path):
    df = pd.read_csv(path, low_memory=False)

    df.columns = df.columns.str.strip()

    if "PID" not in df.columns:
        raise KeyError("Core table must contain PID column.")

    if "group" not in df.columns:
        raise KeyError("Core table must contain group column.")

    df["PID"] = pd.to_numeric(df["PID"], errors="coerce")
    df = df.dropna(subset=["PID"])
    df["PID"] = df["PID"].astype(int)

    df["group"] = df["group"].astype(str).str.strip().str.upper()

    if "program_length" in df.columns:
        df["program_length"] = pd.to_numeric(df["program_length"], errors="coerce")

    # Clean likely boolean columns.
    bool_candidates = [
        "has_program_info",
        "is_study_participant",
        "group_mismatch_signup_vs_program",
        "has_weekly_forms",
        "completed_by_forms",
        "has_final_feedback",
        "flag_duplicate_final_feedback",
        "any_reported_issue",
        "safety_advice_clear",
        "has_garmin_data",
        "reached_30min_ever",
        "has_any_forms_or_feedback",
    ]

    for col in bool_candidates:
        if col in df.columns:
            df[col] = clean_bool(df[col])

    # Numeric conversion for everything except obvious text columns.
    text_cols = [
        "group",
        "signup_group",
        "delivery_mode",
        "tear_grade",
        "status",
        "free_comment",
        "binary_field_comments",
        "followup_status",
        "final_feedback_merge_status",
    ]

    for col in df.columns:
        if col not in text_cols and col != "PID":
            if df[col].dtype == object:
                converted = pd.to_numeric(df[col], errors="ignore")
                df[col] = converted

    # Derived Garmin thresholds.
    if "garmin_max_run" in df.columns:
        df["reached_25min_ever"] = pd.to_numeric(df["garmin_max_run"], errors="coerce") >= 25
        df["reached_30min_ever"] = pd.to_numeric(df["garmin_max_run"], errors="coerce") >= 30

        # Useful "close to target" variable.
        df["distance_to_30min"] = 30 - pd.to_numeric(df["garmin_max_run"], errors="coerce")
        df["within_5min_of_30"] = pd.to_numeric(df["garmin_max_run"], errors="coerce") >= 25

    return df.sort_values(["group", "PID"])


def load_weekly_pid(path):
    if path is None or not os.path.exists(path):
        warnings.warn("Weekly PID file not provided/found. Trajectory plots will be skipped.")
        return None

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    if "PID" not in df.columns or "group" not in df.columns:
        warnings.warn("Weekly PID file missing PID/group columns. Trajectory plots skipped.")
        return None

    df["PID"] = pd.to_numeric(df["PID"], errors="coerce")
    df = df.dropna(subset=["PID"])
    df["PID"] = df["PID"].astype(int)
    df["group"] = df["group"].astype(str).str.strip().str.upper()

    if "progress" in df.columns:
        df["progress"] = pd.to_numeric(df["progress"], errors="coerce")

    if "week" in df.columns:
        df["week"] = pd.to_numeric(df["week"], errors="coerce")

    return df


# ============================================================
# DESCRIPTIVE TABLES
# ============================================================

def summarize_binary(df, variables, label):
    rows = []

    for var in available_columns(df, variables):
        valid = df[var].dropna()
        n_valid = len(valid)

        if n_valid == 0:
            rows.append({
                "section": label,
                "variable": var,
                "n_valid": 0,
                "n_true": np.nan,
                "percent_true": np.nan,
            })
            continue

        true_count = valid.astype(bool).sum()
        percent = true_count / n_valid * 100

        rows.append({
            "section": label,
            "variable": var,
            "n_valid": n_valid,
            "n_true": int(true_count),
            "percent_true": percent,
        })

    return pd.DataFrame(rows)


def summarize_continuous(df, variables, label):
    rows = []

    for var in available_columns(df, variables):
        x = pd.to_numeric(df[var], errors="coerce").dropna()

        med, q1, q3 = iqr(x)

        rows.append({
            "section": label,
            "variable": var,
            "n_valid": len(x),
            "mean": x.mean() if len(x) else np.nan,
            "sd": x.std(ddof=1) if len(x) > 1 else np.nan,
            "median": med,
            "q1": q1,
            "q3": q3,
            "min": x.min() if len(x) else np.nan,
            "max": x.max() if len(x) else np.nan,
            "median_iqr": format_median_iqr(x),
        })

    return pd.DataFrame(rows)


def summarize_score_thresholds(df, variables, threshold=4):
    rows = []

    for var in available_columns(df, variables):
        x = pd.to_numeric(df[var], errors="coerce").dropna()
        n_valid = len(x)

        if n_valid == 0:
            rows.append({
                "variable": var,
                "n_valid": 0,
                "n_at_least_threshold": np.nan,
                "percent_at_least_threshold": np.nan,
                "threshold": threshold,
            })
            continue

        n_high = (x >= threshold).sum()

        rows.append({
            "variable": var,
            "n_valid": n_valid,
            "n_at_least_threshold": int(n_high),
            "percent_at_least_threshold": n_high / n_valid * 100,
            "threshold": threshold,
        })

    return pd.DataFrame(rows)


def summarize_issue_categories(df):
    rows = []

    for var in available_columns(df, ISSUE_CATEGORY_COLS):
        valid = df[var].dropna()

        if len(valid) == 0:
            continue

        vals = clean_bool(valid)
        n_true = vals.sum()

        rows.append({
            "issue_category": var,
            "n_valid": len(vals),
            "n_reported": int(n_true),
            "percent_reported": n_true / len(vals) * 100,
        })

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values("n_reported", ascending=False)

    return out


def sample_description(df):
    continuous = [
        "age",
        "postpartum_weeks",
        "children_count",
        "program_length",
    ]

    rows = []

    for var in available_columns(df, continuous):
        rows.append({
            "variable": var,
            "overall": format_median_iqr(df[var]),
            "group_A": format_median_iqr(df[df["group"] == "A"][var]),
            "group_B": format_median_iqr(df[df["group"] == "B"][var]),
        })

    desc_cont = pd.DataFrame(rows)

    categorical_tables = {}

    for var in available_columns(df, ["delivery_mode", "tear_grade", "status"]):
        tab = (
            df.groupby(["group", var])["PID"]
            .nunique()
            .reset_index(name="n")
            .sort_values(["group", "n"], ascending=[True, False])
        )
        categorical_tables[var] = tab

    return desc_cont, categorical_tables


# ============================================================
# GROUP COMPARISONS
# ============================================================

def compare_groups_continuous(df, variables):
    rows = []

    for var in available_columns(df, variables):
        a = pd.to_numeric(df.loc[df["group"] == "A", var], errors="coerce").dropna()
        b = pd.to_numeric(df.loc[df["group"] == "B", var], errors="coerce").dropna()

        if len(a) == 0 or len(b) == 0:
            rows.append({
                "variable": var,
                "n_A": len(a),
                "n_B": len(b),
                "A_median_iqr": format_median_iqr(a),
                "B_median_iqr": format_median_iqr(b),
                "mannwhitney_U": np.nan,
                "p_value": np.nan,
                "cliffs_delta_B_vs_A": np.nan,
            })
            continue

        try:
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            stat, p = np.nan, np.nan

        delta = cliffs_delta(b, a)

        rows.append({
            "variable": var,
            "n_A": len(a),
            "n_B": len(b),
            "A_median_iqr": format_median_iqr(a),
            "B_median_iqr": format_median_iqr(b),
            "A_mean": a.mean(),
            "B_mean": b.mean(),
            "mannwhitney_U": stat,
            "p_value": p,
            "cliffs_delta_B_vs_A": delta,
        })

    return pd.DataFrame(rows)


def compare_groups_binary(df, variables):
    rows = []

    for var in available_columns(df, variables):
        sub = df[df[var].notna()].copy()

        if len(sub) == 0:
            continue

        sub[var] = clean_bool(sub[var])

        a = sub[sub["group"] == "A"][var].dropna().astype(bool)
        b = sub[sub["group"] == "B"][var].dropna().astype(bool)

        a_true = int(a.sum())
        a_false = int(len(a) - a_true)
        b_true = int(b.sum())
        b_false = int(len(b) - b_true)

        table = [
            [a_true, a_false],
            [b_true, b_false],
        ]

        odds_ratio, p = odds_ratio_from_table(table)

        rows.append({
            "variable": var,
            "n_A": len(a),
            "A_true": a_true,
            "A_percent_true": a_true / len(a) * 100 if len(a) else np.nan,
            "n_B": len(b),
            "B_true": b_true,
            "B_percent_true": b_true / len(b) * 100 if len(b) else np.nan,
            "odds_ratio_A_vs_B": odds_ratio,
            "fisher_p_value": p,
        })

    return pd.DataFrame(rows)


# ============================================================
# PLOTS
# ============================================================

def plot_box_by_group(df, variables, filename, title):
    vars_available = available_columns(df, variables)

    if not vars_available:
        return

    n = len(vars_available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), squeeze=False)
    axes = axes.flatten()

    for ax, var in zip(axes, vars_available):
        data = [
            pd.to_numeric(df.loc[df["group"] == g, var], errors="coerce").dropna()
            for g in GROUP_ORDER
        ]

        ax.boxplot(data, labels=GROUP_ORDER, showmeans=True)

        for i, g in enumerate(GROUP_ORDER, start=1):
            y = pd.to_numeric(df.loc[df["group"] == g, var], errors="coerce").dropna()
            x = np.random.normal(i, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.65)

        ax.set_title(var)
        ax.set_xlabel("Groupe")
        ax.grid(axis="y")

    fig.suptitle(title)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_bar_binary(df, variables, filename, title):
    vars_available = available_columns(df, variables)

    if not vars_available:
        return

    rows = []

    for var in vars_available:
        for g in GROUP_ORDER:
            vals = clean_bool(df.loc[df["group"] == g, var]).dropna()
            percent = vals.mean() * 100 if len(vals) else np.nan

            rows.append({
                "variable": var,
                "group": g,
                "percent": percent,
                "n": len(vals),
            })

    plot_df = pd.DataFrame(rows)

    x = np.arange(len(vars_available))
    width = 0.35

    plt.figure(figsize=(max(10, len(vars_available) * 1.5), 6))

    for offset, g in zip([-width / 2, width / 2], GROUP_ORDER):
        vals = []
        for var in vars_available:
            row = plot_df[(plot_df["variable"] == var) & (plot_df["group"] == g)]
            vals.append(row["percent"].iloc[0] if len(row) else np.nan)

        plt.bar(x + offset, vals, width, label=f"Groupe {g}")

    plt.xticks(x, vars_available, rotation=45, ha="right")
    plt.ylabel("%")
    plt.title(title)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_issue_categories(issue_df):
    if issue_df is None or len(issue_df) == 0:
        return

    plt.figure(figsize=(10, 6))

    plt.barh(issue_df["issue_category"], issue_df["n_reported"])
    plt.xlabel("Nombre de participantes")
    plt.title("Catégories de problèmes rapportés dans le feedback final")
    plt.gca().invert_yaxis()
    plt.grid(axis="x")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "issue_categories_barplot.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_weekly_trajectories(df_weekly, variables):
    if df_weekly is None:
        return

    if "progress" not in df_weekly.columns:
        warnings.warn("Weekly PID table has no progress column. Skipping trajectory plots.")
        return

    for var in available_columns(df_weekly, variables):
        plt.figure(figsize=(10, 6))

        for pid, sub in df_weekly.groupby("PID"):
            sub = sub.sort_values("progress")
            group = sub["group"].iloc[0]
            color = GROUP_COLORS.get(group, "gray")

            plt.plot(
                sub["progress"],
                pd.to_numeric(sub[var], errors="coerce"),
                color=color,
                alpha=0.20,
            )

        for group in GROUP_ORDER:
            sub = df_weekly[df_weekly["group"] == group].copy()
            sub[var] = pd.to_numeric(sub[var], errors="coerce")

            if "progress_bin" in sub.columns:
                grouped = sub.groupby("progress_bin", observed=True)[var].median()
                x_map = {
                    "0-25%": 0.125,
                    "25-50%": 0.375,
                    "50-75%": 0.625,
                    "75-100%": 0.875,
                }
                xs = [x_map.get(str(idx), np.nan) for idx in grouped.index]
                ys = grouped.values
            else:
                grouped = sub.groupby("week")[var].median()
                xs = grouped.index
                ys = grouped.values

            plt.plot(
                xs,
                ys,
                marker="o",
                linewidth=3,
                label=f"{GROUP_LABELS[group]} median",
                color=GROUP_COLORS.get(group),
            )

        plt.title(f"Trajectoires hebdomadaires - {var}")
        plt.xlabel("Progression normalisée")
        plt.ylabel(var)
        plt.xlim(0, 1)
        plt.grid()
        plt.legend()
        plt.tight_layout()

        path = os.path.join(PLOTS_DIR, f"weekly_trajectory_{var}.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Saved: {path}")


# ============================================================
# REPORT
# ============================================================

def generate_markdown_report(
    core,
    sample_cont,
    h1_binary,
    h1_cont,
    score_thresholds,
    issues,
    h2_cont,
    h2_bin,
    garmin_cont,
    garmin_bin,
):
    lines = []

    n_total = core["PID"].nunique()
    n_a = core[core["group"] == "A"]["PID"].nunique()
    n_b = core[core["group"] == "B"]["PID"].nunique()

    lines.append("# Statistical summary report")
    lines.append("")
    lines.append("## Study framing")
    lines.append("")
    lines.append(
        "Group A used the digital decision algorithm to select a personalized running program. "
        "Group B was assigned a program directly. The main analysis therefore evaluates feasibility, "
        "acceptability, adherence, and exploratory differences between algorithm-guided personalization "
        "and direct assignment."
    )
    lines.append("")
    lines.append("## Sample")
    lines.append("")
    lines.append(f"- Total participants: **{n_total}**")
    lines.append(f"- Group A: **{n_a}**")
    lines.append(f"- Group B: **{n_b}**")
    lines.append("")

    if len(sample_cont):
        lines.append("### Baseline continuous variables")
        lines.append("")
        lines.append(sample_cont.to_markdown(index=False))
        lines.append("")

    lines.append("## H1 — Feasibility")
    lines.append("")
    lines.append("### Binary feasibility indicators")
    lines.append("")
    if len(h1_binary):
        lines.append(h1_binary.to_markdown(index=False))
    lines.append("")

    lines.append("### Continuous feasibility indicators")
    lines.append("")
    if len(h1_cont):
        lines.append(h1_cont[[
            "section", "variable", "n_valid", "median_iqr", "mean", "sd", "min", "max"
        ]].to_markdown(index=False))
    lines.append("")

    lines.append("## H1 — Acceptability and satisfaction")
    lines.append("")
    lines.append("Scores are summarized by median [IQR] and by the proportion of participants scoring ≥4/5.")
    lines.append("")
    if len(score_thresholds):
        lines.append(score_thresholds.to_markdown(index=False))
    lines.append("")

    lines.append("## H1 — Reported algorithm/tool issues")
    lines.append("")
    if len(issues):
        lines.append(issues.to_markdown(index=False))
    else:
        lines.append("No issue category columns available.")
    lines.append("")
    lines.append(
        "Interpretation note: high quantitative satisfaction scores can coexist with specific negative comments. "
        "This usually means the tool was broadly acceptable, but certain components require clarification."
    )
    lines.append("")

    lines.append("## H2 — Exploratory comparison: Group A vs Group B")
    lines.append("")
    lines.append(
        "Mann–Whitney U tests were used for continuous/ordinal variables. "
        "Fisher exact tests were used for binary variables. These comparisons are exploratory and not adjusted for multiple testing."
    )
    lines.append("")

    lines.append("### Continuous / ordinal outcomes")
    lines.append("")
    if len(h2_cont):
        lines.append(h2_cont.to_markdown(index=False))
    lines.append("")

    lines.append("### Binary outcomes")
    lines.append("")
    if len(h2_bin):
        lines.append(h2_bin.to_markdown(index=False))
    lines.append("")

    lines.append("## Garmin exploratory analysis")
    lines.append("")
    lines.append(
        "Garmin data were treated as supportive and exploratory because data availability was incomplete and uneven between groups. "
        "Continuous Garmin metrics are preferred over strict binary thresholds such as reaching exactly 30 minutes."
    )
    lines.append("")

    lines.append("### Garmin continuous outcomes")
    lines.append("")
    if len(garmin_cont):
        lines.append(garmin_cont.to_markdown(index=False))
    lines.append("")

    lines.append("### Garmin binary thresholds")
    lines.append("")
    if len(garmin_bin):
        lines.append(garmin_bin.to_markdown(index=False))
    lines.append("")

    lines.append("## Recommended interpretation language")
    lines.append("")
    lines.append(
        "The algorithm-guided pathway appeared feasible if most participants submitted weekly forms, "
        "showed acceptable program coverage, and provided high satisfaction/acceptability scores. "
        "Reported issues should be interpreted as targets for refinement rather than as a contradiction of high global satisfaction."
    )
    lines.append("")
    lines.append(
        "Group comparisons should be interpreted cautiously due to small sample size. "
        "Effect direction and magnitude are more informative than p-values alone."
    )
    lines.append("")

    return "\n".join(lines)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run feasibility, adherence, acceptability, and exploratory group statistics."
    )

    parser.add_argument(
        "--core",
        default="core_participant_table.csv",
        help="Path to core participant table CSV.",
    )

    parser.add_argument(
        "--weekly",
        default="forms_output/forms_weekly_pid_cleaned.csv",
        help="Optional path to weekly PID-level table for trajectory plots.",
    )

    args = parser.parse_args()

    ensure_dirs()

    print("\nLoading core participant table...")
    core = load_core(args.core)

    print(f"Loaded {len(core)} rows, {core['PID'].nunique()} unique participants.")
    print(core.groupby("group")["PID"].nunique())

    weekly = load_weekly_pid(args.weekly)

    # ------------------------------------------------------------
    # Sample description
    # ------------------------------------------------------------
    sample_cont, categorical_tables = sample_description(core)
    save_csv(sample_cont, "sample_description_continuous.csv")

    for name, tab in categorical_tables.items():
        save_csv(tab, f"sample_description_{name}.csv")

    # ------------------------------------------------------------
    # H1 summaries
    # ------------------------------------------------------------
    h1_binary = pd.concat([
        summarize_binary(core, H1_FEASIBILITY_BINARY, "H1 feasibility"),
        summarize_binary(core, H1_ISSUES_BINARY, "H1 issues"),
    ], ignore_index=True)

    h1_cont = pd.concat([
        summarize_continuous(core, H1_FEASIBILITY_CONTINUOUS, "H1 feasibility"),
        summarize_continuous(core, H1_ADHERENCE, "H1 adherence"),
        summarize_continuous(core, H1_WEEKLY_ACCEPTABILITY, "H1 weekly acceptability"),
        summarize_continuous(core, H1_FINAL_ACCEPTABILITY, "H1 final acceptability"),
        summarize_continuous(core, H1_ISSUES_CONTINUOUS, "H1 issues"),
    ], ignore_index=True)

    score_thresholds = summarize_score_thresholds(
        core,
        H1_WEEKLY_ACCEPTABILITY + H1_FINAL_ACCEPTABILITY,
        threshold=4,
    )

    issues = summarize_issue_categories(core)

    save_csv(h1_binary, "h1_binary_summary.csv")
    save_csv(h1_cont, "h1_continuous_summary.csv")
    save_csv(score_thresholds, "h1_score_thresholds_ge4.csv")
    save_csv(issues, "h1_issue_categories.csv")

    # ------------------------------------------------------------
    # H2 group comparisons
    # ------------------------------------------------------------
    h2_cont = compare_groups_continuous(core, H2_CONTINUOUS)
    h2_bin = compare_groups_binary(core, H2_BINARY)

    save_csv(h2_cont, "h2_group_comparisons_continuous.csv")
    save_csv(h2_bin, "h2_group_comparisons_binary.csv")

    # ------------------------------------------------------------
    # Garmin exploratory analysis
    # ------------------------------------------------------------
    garmin_cont = compare_groups_continuous(core, GARMIN_CONTINUOUS)
    garmin_bin = compare_groups_binary(core, GARMIN_BINARY)

    save_csv(garmin_cont, "garmin_group_comparisons_continuous.csv")
    save_csv(garmin_bin, "garmin_group_comparisons_binary.csv")

    garmin_available = core[core.get("has_garmin_data", False).fillna(False).astype(bool)].copy()
    save_csv(garmin_available, "participants_with_garmin_data.csv")

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------
    print("\nGenerating plots...")

    plot_box_by_group(
        core,
        ["forms_coverage", "forms_completion_rate", "mean_adherence_ratio"],
        "h1_h2_adherence_boxplots.png",
        "Adhérence et couverture des formulaires par groupe",
    )

    plot_box_by_group(
        core,
        ["clarity_mean", "satisfaction_mean", "confidence_mean", "recommendation_mean"],
        "weekly_acceptability_boxplots.png",
        "Scores hebdomadaires d'acceptabilité par groupe",
    )

    plot_box_by_group(
        core,
        ["mean_final_score", "global_use_score", "process_score", "confidence_score", "recommendation_score"],
        "final_feedback_scores_boxplots.png",
        "Scores du feedback final par groupe",
    )

    plot_box_by_group(
        core,
        ["garmin_max_run", "garmin_last_run", "garmin_mean_run", "n_garmin_sessions_included"],
        "garmin_continuous_boxplots.png",
        "Variables Garmin exploratoires par groupe",
    )

    plot_bar_binary(
        core,
        ["completed_by_forms", "has_final_feedback", "has_garmin_data", "any_reported_issue", "safety_advice_clear"],
        "binary_indicators_by_group.png",
        "Indicateurs binaires par groupe",
    )

    plot_bar_binary(
        core,
        ["reached_25min_ever", "reached_30min_ever"],
        "garmin_thresholds_by_group.png",
        "Seuils Garmin exploratoires par groupe",
    )

    plot_issue_categories(issues)

    plot_weekly_trajectories(
        weekly,
        [
            "adherence_ratio",
            "sessions_done",
            "satisfaction",
            "confidence",
            "recommendation",
            "fatigue_mean_score",
            "symptom_mean_score",
        ],
    )

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------
    report = generate_markdown_report(
        core=core,
        sample_cont=sample_cont,
        h1_binary=h1_binary,
        h1_cont=h1_cont,
        score_thresholds=score_thresholds,
        issues=issues,
        h2_cont=h2_cont,
        h2_bin=h2_bin,
        garmin_cont=garmin_cont,
        garmin_bin=garmin_bin,
    )

    save_text(report, "statistics_report.md")

    print("\nDone.")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print(f"Plots saved in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()