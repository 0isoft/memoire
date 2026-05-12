# statistics/investigate_feedback_issues.py

import os
import argparse
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_DIR = "statistics/investigation"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")


ISSUE_COLUMNS = [
    "issue_contraindications",
    "issue_functional_tests",
    "issue_characteristics",
    "issue_criteria",
    "issue_symptoms",
    "issue_test_ease",
    "issue_program_choice",
]

ISSUE_LABELS = {
    "issue_contraindications": "Contre-indications",
    "issue_functional_tests": "Tests fonctionnels",
    "issue_characteristics": "Caractéristiques",
    "issue_criteria": "Critères",
    "issue_symptoms": "Symptômes",
    "issue_test_ease": "Facilité des tests",
    "issue_program_choice": "Choix du programme",
}

COMMENT_COLUMNS = [
    "free_comment",
    "binary_field_comments",
    "issue_contraindications_comment",
    "issue_functional_tests_comment",
    "issue_characteristics_comment",
    "issue_criteria_comment",
    "issue_symptoms_comment",
    "issue_test_ease_comment",
    "issue_program_choice_comment",
    "safety_advice_clear_comment",
]

SCORE_COLUMNS = [
    "mean_final_score",
    "global_use_score",
    "process_score",
    "confidence_score",
    "recommendation_score",
]


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def save_csv(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def clean_pid(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace("PP-", "", regex=False)
        .str.replace("pp-", "", regex=False)
        .str.extract(r"(\d+)", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
    )


def clean_bool(series):
    if series is None:
        return pd.Series(dtype="boolean")

    s = series.copy()

    if pd.api.types.is_bool_dtype(s):
        return s.astype("boolean")

    text = (
        s.astype(str)
        .str.strip()
        .str.lower()
    )

    mapped = text.map({
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "oui": True,
        "non": False,
        "nan": pd.NA,
        "none": pd.NA,
        "": pd.NA,
        "<na>": pd.NA,
    })

    return mapped.astype("boolean")


def clean_group(series):
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": np.nan, "": np.nan})
    )


def clean_text(x):
    if pd.isna(x):
        return ""

    x = str(x).strip()

    if x.lower() in {"nan", "none", "<na>"}:
        return ""

    return x


def combine_comments(row):
    comments = []

    for col in COMMENT_COLUMNS:
        if col in row.index:
            txt = clean_text(row[col])
            if txt:
                comments.append(f"{col}: {txt}")

    return "\n\n".join(comments)


def load_core(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    df["PID"] = clean_pid(df["PID"])
    df = df.dropna(subset=["PID"])
    df["PID"] = df["PID"].astype(int)

    if "group" in df.columns:
        df["group"] = clean_group(df["group"])

    return df


def load_feedback(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    if "PID" not in df.columns:
        raise KeyError("Final feedback file must contain a PID column.")

    df["PID"] = clean_pid(df["PID"])
    df = df.dropna(subset=["PID"])
    df["PID"] = df["PID"].astype(int)

    if "group" in df.columns:
        df["group"] = clean_group(df["group"])

    if "program_length" in df.columns:
        df["program_length"] = pd.to_numeric(df["program_length"], errors="coerce")

    for col in SCORE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ISSUE_COLUMNS:
        if col in df.columns:
            df[col] = clean_bool(df[col])

    if "safety_advice_clear" in df.columns:
        df["safety_advice_clear"] = clean_bool(df["safety_advice_clear"])

    if "n_reported_issues" in df.columns:
        df["n_reported_issues"] = pd.to_numeric(df["n_reported_issues"], errors="coerce")

    if "has_final_feedback" in df.columns:
        df["has_final_feedback"] = clean_bool(df["has_final_feedback"])
    else:
        if "n_final_feedback_responses" in df.columns:
            df["has_final_feedback"] = pd.to_numeric(
                df["n_final_feedback_responses"],
                errors="coerce"
            ).fillna(0).gt(0)
        else:
            df["has_final_feedback"] = df["mean_final_score"].notna()

    return df


def build_issue_frequency(feedback):
    rows = []

    valid_feedback = feedback[feedback["has_final_feedback"].fillna(False)].copy()
    n_total = len(valid_feedback)

    for col in ISSUE_COLUMNS:
        if col not in valid_feedback.columns:
            continue

        vals = clean_bool(valid_feedback[col])
        n_valid = vals.notna().sum()
        n_true = vals.fillna(False).sum()

        rows.append({
            "issue_column": col,
            "issue_label": ISSUE_LABELS.get(col, col),
            "n_total_feedback": n_total,
            "n_valid": int(n_valid),
            "n_reported": int(n_true),
            "percent_of_feedback_participants": float(n_true / n_total * 100) if n_total else np.nan,
            "percent_of_valid": float(n_true / n_valid * 100) if n_valid else np.nan,
        })

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values("n_reported", ascending=False)

    return out


def build_issue_frequency_by_group(feedback):
    rows = []

    valid_feedback = feedback[feedback["has_final_feedback"].fillna(False)].copy()

    for group, group_df in valid_feedback.groupby("group"):
        n_group = len(group_df)

        for col in ISSUE_COLUMNS:
            if col not in group_df.columns:
                continue

            vals = clean_bool(group_df[col])
            n_valid = vals.notna().sum()
            n_true = vals.fillna(False).sum()

            rows.append({
                "group": group,
                "issue_column": col,
                "issue_label": ISSUE_LABELS.get(col, col),
                "n_feedback_group": n_group,
                "n_valid": int(n_valid),
                "n_reported": int(n_true),
                "percent_of_group_feedback": float(n_true / n_group * 100) if n_group else np.nan,
                "percent_of_valid": float(n_true / n_valid * 100) if n_valid else np.nan,
            })

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values(["issue_column", "group"])

    return out


def build_comments_for_quotation(feedback):
    df = feedback.copy()

    df["combined_comment"] = df.apply(combine_comments, axis=1)

    # Keep rows with any comment.
    df = df[df["combined_comment"].str.strip() != ""].copy()

    keep = [
        "PID",
        "group",
        "program_length",
        "mean_final_score",
        "global_use_score",
        "process_score",
        "confidence_score",
        "recommendation_score",
        "n_reported_issues",
        "any_reported_issue",
        "combined_comment",
    ]

    for col in ISSUE_COLUMNS:
        if col in feedback.columns:
            keep.append(col)

    keep = [c for c in keep if c in df.columns]

    out = df[keep].copy()

    if "n_reported_issues" in out.columns:
        out = out.sort_values(
            ["n_reported_issues", "mean_final_score"],
            ascending=[False, True],
        )

    return out


def build_score_vs_issues_table(feedback):
    df = feedback[feedback["has_final_feedback"].fillna(False)].copy()

    keep = [
        "PID",
        "group",
        "program_length",
        "mean_final_score",
        "global_use_score",
        "process_score",
        "confidence_score",
        "recommendation_score",
        "n_reported_issues",
        "any_reported_issue",
        "safety_advice_clear",
        "free_comment",
        "binary_field_comments",
    ]

    for col in ISSUE_COLUMNS:
        if col in df.columns:
            keep.append(col)

    keep = [c for c in keep if c in df.columns]

    out = df[keep].copy()

    if "n_reported_issues" in out.columns:
        out = out.sort_values(
            ["n_reported_issues", "mean_final_score"],
            ascending=[False, True],
        )

    return out


def build_interesting_cases(feedback):
    df = feedback[feedback["has_final_feedback"].fillna(False)].copy()

    if "combined_comment" not in df.columns:
        df["combined_comment"] = df.apply(combine_comments, axis=1)

    # These are cases worth inspecting manually.
    low_score = df["mean_final_score"].lt(4) if "mean_final_score" in df.columns else False
    many_issues = df["n_reported_issues"].ge(2) if "n_reported_issues" in df.columns else False
    high_score_but_complaint = (
        df["mean_final_score"].ge(4.5)
        & df["combined_comment"].str.strip().ne("")
    ) if "mean_final_score" in df.columns else False

    df["case_low_final_score"] = low_score
    df["case_many_reported_issues"] = many_issues
    df["case_high_score_but_comment"] = high_score_but_complaint

    mask = (
        df["case_low_final_score"]
        | df["case_many_reported_issues"]
        | df["case_high_score_but_comment"]
    )

    keep = [
        "PID",
        "group",
        "program_length",
        "mean_final_score",
        "global_use_score",
        "process_score",
        "confidence_score",
        "recommendation_score",
        "n_reported_issues",
        "any_reported_issue",
        "case_low_final_score",
        "case_many_reported_issues",
        "case_high_score_but_comment",
        "combined_comment",
    ]

    for col in ISSUE_COLUMNS:
        if col in df.columns:
            keep.append(col)

    keep = [c for c in keep if c in df.columns]

    return df.loc[mask, keep].sort_values(
        ["case_many_reported_issues", "mean_final_score"],
        ascending=[False, True],
    )


def plot_issue_frequency(issue_freq):
    if len(issue_freq) == 0:
        return

    plot_df = issue_freq.sort_values("n_reported", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["issue_label"], plot_df["n_reported"])
    plt.xlabel("Nombre de participantes ayant signalé le problème")
    plt.ylabel("Catégorie")
    plt.title("Problèmes rapportés dans le feedback final")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "final_feedback_issue_frequency.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


def plot_final_score_vs_issues(feedback):
    df = feedback[feedback["has_final_feedback"].fillna(False)].copy()

    if "mean_final_score" not in df.columns or "n_reported_issues" not in df.columns:
        return

    df["mean_final_score"] = pd.to_numeric(df["mean_final_score"], errors="coerce")
    df["n_reported_issues"] = pd.to_numeric(df["n_reported_issues"], errors="coerce")

    df = df.dropna(subset=["mean_final_score", "n_reported_issues"])

    if len(df) == 0:
        return

    plt.figure(figsize=(8, 6))

    for group in ["A", "B"]:
        sub = df[df["group"] == group]
        plt.scatter(
            sub["n_reported_issues"],
            sub["mean_final_score"],
            label=f"Groupe {group}",
            alpha=0.8,
        )

    plt.xlabel("Nombre de problèmes rapportés")
    plt.ylabel("Score final moyen")
    plt.title("Score final vs nombre de problèmes rapportés")
    plt.ylim(0, 5.2)
    plt.grid()
    plt.legend()
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "final_score_vs_reported_issues.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


def write_markdown_issue_report(issue_freq, issue_by_group, quote_table, interesting_cases):
    path = os.path.join(OUTPUT_DIR, "feedback_issue_investigation_report.md")

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Final feedback issue investigation\n\n")

        f.write("## Issue frequency\n\n")
        if len(issue_freq):
            f.write(issue_freq.to_markdown(index=False))
            f.write("\n\n")
        else:
            f.write("No issue columns were found.\n\n")

        f.write("## Issue frequency by group\n\n")
        if len(issue_by_group):
            f.write(issue_by_group.to_markdown(index=False))
            f.write("\n\n")
        else:
            f.write("No issue-by-group table available.\n\n")

        f.write("## Comments potentially useful for quotation\n\n")
        if len(quote_table):
            for _, row in quote_table.iterrows():
                pid = row.get("PID", "")
                group = row.get("group", "")
                score = row.get("mean_final_score", np.nan)
                n_issues = row.get("n_reported_issues", np.nan)
                comment = row.get("combined_comment", "")

                f.write(f"### PID {pid} — Groupe {group}\n\n")
                f.write(f"- Score final moyen: {score}\n")
                f.write(f"- Nombre de problèmes rapportés: {n_issues}\n\n")
                f.write("> ")
                wrapped = textwrap.fill(str(comment).replace("\n", " "), width=100)
                f.write(wrapped.replace("\n", "\n> "))
                f.write("\n\n")
        else:
            f.write("No comments available.\n\n")

        f.write("## Interesting cases\n\n")
        if len(interesting_cases):
            f.write(interesting_cases.to_markdown(index=False))
            f.write("\n\n")
        else:
            f.write("No interesting cases found.\n\n")

    print(f"Saved: {path}")


def print_sanity(feedback, issue_freq, quote_table):
    print("\n--- Final feedback sanity ---")
    print(f"Rows loaded: {len(feedback)}")
    print(f"Unique PIDs: {feedback['PID'].nunique()}")

    if "group" in feedback.columns:
        print("\nParticipants by group:")
        print(feedback.groupby("group")["PID"].nunique())

    if "has_final_feedback" in feedback.columns:
        print("\nHas final feedback:")
        print(feedback["has_final_feedback"].value_counts(dropna=False))

    print("\nIssue columns found:")
    for col in ISSUE_COLUMNS:
        print(f"{col}: {'yes' if col in feedback.columns else 'NO'}")

    print("\nTop issue categories:")
    if len(issue_freq):
        print(issue_freq[["issue_label", "n_reported", "percent_of_feedback_participants"]].to_string(index=False))
    else:
        print("None")

    print("\nComments available for quotation:")
    print(len(quote_table))


def main():
    parser = argparse.ArgumentParser(
        description="Secondary investigation of final feedback issues and comments."
    )

    parser.add_argument(
        "--core",
        default="statistics/core_participant_table.csv",
        help="Path to core participant table.",
    )

    parser.add_argument(
        "--feedback",
        default="final_feedback_output/final_feedback_participant_summary.csv",
        help="Path to final feedback participant summary.",
    )

    args = parser.parse_args()

    ensure_dirs()

    print("Loading final feedback directly...")
    feedback = load_feedback(args.feedback)

    # Optional merge with core, useful if final feedback lacks some metadata.
    if args.core and os.path.exists(args.core):
        print("Loading core table for backup metadata...")
        core = load_core(args.core)

        backup_cols = [
            "PID",
            "group",
            "program_length",
            "age",
            "postpartum_weeks",
            "children_count",
            "mean_adherence_ratio",
            "forms_coverage",
            "satisfaction_mean",
            "confidence_mean",
            "recommendation_mean",
        ]

        backup_cols = [c for c in backup_cols if c in core.columns]

        feedback = feedback.merge(
            core[backup_cols],
            on="PID",
            how="left",
            suffixes=("", "_core"),
        )

        # Fill group/program_length from core if missing.
        for col in ["group", "program_length"]:
            core_col = f"{col}_core"
            if col in feedback.columns and core_col in feedback.columns:
                feedback[col] = feedback[col].combine_first(feedback[core_col])
                feedback = feedback.drop(columns=[core_col])

    issue_freq = build_issue_frequency(feedback)
    issue_by_group = build_issue_frequency_by_group(feedback)
    quote_table = build_comments_for_quotation(feedback)
    score_vs_issues = build_score_vs_issues_table(feedback)
    interesting_cases = build_interesting_cases(feedback)

    save_csv(issue_freq, "final_feedback_issue_frequency.csv")
    save_csv(issue_by_group, "final_feedback_issue_frequency_by_group.csv")
    save_csv(quote_table, "final_feedback_comments_for_quotes.csv")
    save_csv(score_vs_issues, "final_feedback_scores_vs_issues.csv")
    save_csv(interesting_cases, "final_feedback_interesting_cases.csv")

    plot_issue_frequency(issue_freq)
    plot_final_score_vs_issues(feedback)

    write_markdown_issue_report(
        issue_freq=issue_freq,
        issue_by_group=issue_by_group,
        quote_table=quote_table,
        interesting_cases=interesting_cases,
    )

    print_sanity(feedback, issue_freq, quote_table)

    print("\nDone.")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print(f"Plots saved in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()