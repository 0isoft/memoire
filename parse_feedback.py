#!/usr/bin/env python3
"""
parse-final-feedback.py

Cleans and aggregates the final review / feedback Google Form.

Inputs:
- program_length.csv with columns:
    PID, program_length, group

- final feedback CSV with columns like:
    Horodateur, PID, Contre-indications, Tests fonctionnels, ...
    Utilisation globale, Déroulement, Confiance, Recommandation,
    Remarques libres, Statut Suivi

Outputs:
- final_feedback_output/final_feedback_cleaned.csv
- final_feedback_output/final_feedback_participant_summary.csv

This parser intentionally does NOT use Garmin data.
"""

import os
import re
import argparse
import unicodedata
import warnings

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "final_feedback_output"

BINARY_FEEDBACK_COLUMNS = {
    "issue_contraindications": {
        "label": "Contre-indications",
        "match": ["contre-indications", "contre indications"],
    },
    "issue_functional_tests": {
        "label": "Tests fonctionnels",
        "match": ["tests fonctionnels", "tests fonctionnels"],
    },
    "issue_characteristics": {
        "label": "Caractéristiques",
        "match": ["caracteristiques", "caractéristiques"],
    },
    "issue_criteria": {
        "label": "Critères",
        "match": ["criteres", "critères"],
    },
    "issue_symptoms": {
        "label": "Symptômes",
        "match": ["symptomes", "symptômes"],
    },
    "issue_test_ease": {
        "label": "Facilité des tests",
        "match": ["facilite des tests", "facilité des tests"],
    },
    "issue_program_choice": {
        "label": "Choix du programme",
        "match": ["choix du programme"],
    },
    "safety_advice_clear": {
        "label": "Conseils de sécurité",
        "match": ["conseils de securite", "conseils de sécurité"],
    },
}

SCORE_COLUMNS = {
    "global_use_score": {
        "label": "Utilisation globale",
        "match": ["utilisation globale"],
    },
    "process_score": {
        "label": "Déroulement",
        "match": ["deroulement", "déroulement"],
    },
    "confidence_score": {
        "label": "Confiance",
        "match": ["confiance"],
    },
    "recommendation_score": {
        "label": "Recommandation",
        "match": ["recommandation"],
    },
}

TEXT_COLUMNS = {
    "free_comment": {
        "label": "Remarques libres",
        "match": ["remarques libres"],
    },
    "followup_status": {
        "label": "Statut Suivi",
        "match": ["statut suivi"],
    },
}


# ============================================================
# HELPERS
# ============================================================

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x).strip().lower()
    x = unicodedata.normalize("NFKD", x)
    x = "".join(ch for ch in x if not unicodedata.combining(ch))

    x = x.replace("’", "'")
    x = x.replace("œ", "oe")
    x = x.replace("æ", "ae")

    x = re.sub(r"\s+", " ", x)
    return x.strip()


def normalize_column_name(x):
    x = normalize_text(x)

    for char in ["[", "]", ":", "?", ",", ";", "/", "(", ")", "-", "_"]:
        x = x.replace(char, " ")

    x = re.sub(r"\s+", " ", x)
    return x.strip()


def clean_pid(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace("PP-", "", regex=False)
        .str.replace("pp-", "", regex=False)
        .str.extract(r"(\d+)", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
    )


def clean_group(series):
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": np.nan, "": np.nan})
    )


def to_numeric_clean(series):
    return pd.to_numeric(series, errors="coerce")


def parse_datetime_flexible(series):
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def find_column(df, candidates, required=False):
    normalized_cols = {col: normalize_column_name(col) for col in df.columns}

    for candidate in candidates:
        candidate_norm = normalize_column_name(candidate)

        for original_col, normalized_col in normalized_cols.items():
            if candidate_norm == normalized_col:
                return original_col

        for original_col, normalized_col in normalized_cols.items():
            if candidate_norm in normalized_col:
                return original_col

    if required:
        raise KeyError(
            f"Could not find required column matching one of: {candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def clean_free_text(series):
    return (
        series.astype(str)
        .str.strip()
        .replace({"nan": np.nan, "": np.nan})
    )


def encode_yes_no_or_comment(value):
    """
    Encodes final-feedback binary columns.

    Expected answers are mostly:
    - Non
    - Oui

    But some participants wrote comments directly, e.g.
    "j'ai eu du mal à comprendre running woman"

    Returns:
    - value_bool:
        True if explicit Oui
        False if explicit Non
        NaN if free-text / unclear / missing

    - comment:
        free-text content if the cell was not a clean Oui/Non.
    """

    if pd.isna(value):
        return np.nan, np.nan

    raw = str(value).strip()
    norm = normalize_text(raw)

    if norm == "":
        return np.nan, np.nan

    if norm in {"oui", "yes", "y"}:
        return True, np.nan

    if norm in {"non", "no", "n"}:
        return False, np.nan

    return np.nan, raw


def save_csv(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


# ============================================================
# LOAD PROGRAM LENGTH
# ============================================================

def load_program_length(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = ["PID", "program_length", "group"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(f"program_length file missing columns: {missing}")

    df = df[required].copy()

    df["PID"] = clean_pid(df["PID"])
    df["program_length"] = to_numeric_clean(df["program_length"])
    df["group"] = clean_group(df["group"])

    df = df.dropna(subset=["PID", "program_length", "group"])
    df["PID"] = df["PID"].astype(int)
    df["program_length"] = df["program_length"].astype(int)

    df = df.drop_duplicates(subset=["PID"], keep="first")

    return df.sort_values(["group", "PID"])


# ============================================================
# LOAD FINAL FEEDBACK
# ============================================================

def load_final_feedback(path, df_prog):
    df_raw = pd.read_csv(path, low_memory=False)
    df_raw.columns = df_raw.columns.str.strip()

    pid_col = find_column(df_raw, ["PID", "Votre PID"], required=True)
    timestamp_col = find_column(df_raw, ["Horodateur", "Timestamp"], required=False)

    out = pd.DataFrame()

    out["PID"] = clean_pid(df_raw[pid_col])

    if timestamp_col:
        out["timestamp"] = parse_datetime_flexible(df_raw[timestamp_col])
    else:
        out["timestamp"] = pd.NaT

    # Binary / yes-no-ish feedback questions.
    all_binary_comment_cols = []

    for clean_name, spec in BINARY_FEEDBACK_COLUMNS.items():
        col = find_column(df_raw, spec["match"], required=False)

        if col is None:
            out[clean_name] = np.nan
            out[f"{clean_name}_comment"] = np.nan
            warnings.warn(f"Final feedback binary column not found: {clean_name}")
            continue

        encoded = df_raw[col].apply(encode_yes_no_or_comment)

        out[clean_name] = encoded.apply(lambda x: x[0])
        out[f"{clean_name}_comment"] = encoded.apply(lambda x: x[1])

        all_binary_comment_cols.append(f"{clean_name}_comment")

    # Numeric scores.
    for clean_name, spec in SCORE_COLUMNS.items():
        col = find_column(df_raw, spec["match"], required=False)

        if col is None:
            out[clean_name] = np.nan
            warnings.warn(f"Final feedback score column not found: {clean_name}")
        else:
            out[clean_name] = to_numeric_clean(df_raw[col])

    # Free text columns.
    for clean_name, spec in TEXT_COLUMNS.items():
        col = find_column(df_raw, spec["match"], required=False)

        if col is None:
            out[clean_name] = np.nan
            warnings.warn(f"Final feedback text column not found: {clean_name}")
        else:
            out[clean_name] = clean_free_text(df_raw[col])

    # Basic cleaning.
    out = out.dropna(subset=["PID"])
    out["PID"] = out["PID"].astype(int)

    # If duplicate final responses exist, keep the latest one.
    out = out.sort_values(["PID", "timestamp"])
    out["n_final_feedback_responses"] = out.groupby("PID")["PID"].transform("count")
    out["flag_duplicate_final_feedback"] = out["n_final_feedback_responses"] > 1
    out = out.groupby("PID", as_index=False).tail(1)

    # Merge official participant information.
    out = out.merge(df_prog, on="PID", how="left")

    out["has_program_info"] = out["program_length"].notna()
    out["is_study_participant"] = out["has_program_info"]

    # Derived scores.
    issue_cols = [
        c for c in BINARY_FEEDBACK_COLUMNS.keys()
        if c != "safety_advice_clear"
    ]

    existing_issue_cols = [c for c in issue_cols if c in out.columns]

    # True = participant reported a problem/issue.
    out["n_reported_issues"] = out[existing_issue_cols].sum(axis=1, skipna=True)
    out["any_reported_issue"] = out["n_reported_issues"] > 0

    score_cols = list(SCORE_COLUMNS.keys())
    existing_score_cols = [c for c in score_cols if c in out.columns]

    out["mean_final_score"] = out[existing_score_cols].mean(axis=1, skipna=True)

    # Consolidate weird free-text answers inside yes/no fields.
    if all_binary_comment_cols:
        out["binary_field_comments"] = out[all_binary_comment_cols].apply(
            lambda row: " | ".join(
                str(v).strip()
                for v in row
                if pd.notna(v) and str(v).strip() != ""
            ),
            axis=1,
        )
        out["binary_field_comments"] = out["binary_field_comments"].replace({"": np.nan})
    else:
        out["binary_field_comments"] = np.nan

    # Put core columns first.
    core_cols = [
        "PID",
        "group",
        "program_length",
        "timestamp",
        "has_program_info",
        "is_study_participant",
        "n_final_feedback_responses",
        "flag_duplicate_final_feedback",
        "n_reported_issues",
        "any_reported_issue",
        "mean_final_score",
        "global_use_score",
        "process_score",
        "confidence_score",
        "recommendation_score",
        "safety_advice_clear",
        "free_comment",
        "binary_field_comments",
        "followup_status",
    ]

    remaining_cols = [c for c in out.columns if c not in core_cols]
    out = out[core_cols + remaining_cols]

    return out.sort_values(["group", "PID"], na_position="last")


# ============================================================
# PARTICIPANT SUMMARY
# ============================================================

def build_participant_summary(df_feedback, df_prog):
    """
    Keeps ALL official study participants.

    Participants without final feedback remain present with:
    has_final_feedback = False
    """

    feedback_keep = df_feedback.copy()

    summary = df_prog.merge(
        feedback_keep,
        on=["PID", "group", "program_length"],
        how="left",
    )

    summary["has_final_feedback"] = summary["timestamp"].notna()

    return summary.sort_values(["group", "PID"])


# ============================================================
# REPORTING
# ============================================================

def print_sanity_checks(df_prog, df_feedback, participant_summary):
    print("\n--- Program participants ---")
    print(df_prog.groupby("group")["PID"].nunique())

    print("\n--- Final feedback responses with program info ---")
    print(df_feedback[df_feedback["has_program_info"]].groupby("group")["PID"].nunique())

    print("\n--- Final feedback responses without program info ---")
    no_program = df_feedback[~df_feedback["has_program_info"]]
    if len(no_program) == 0:
        print("None")
    else:
        print(no_program[["PID", "timestamp"]].to_string(index=False))

    print("\n--- Final feedback coverage by group ---")
    print(participant_summary.groupby(["group", "has_final_feedback"])["PID"].nunique())

    print("\n--- Participants without final feedback ---")
    missing = participant_summary[~participant_summary["has_final_feedback"]]
    if len(missing) == 0:
        print("None")
    else:
        print(missing[["PID", "group", "program_length"]].to_string(index=False))

    print("\n--- Mean final scores by group ---")
    score_cols = [
        "global_use_score",
        "process_score",
        "confidence_score",
        "recommendation_score",
        "mean_final_score",
    ]

    existing = [c for c in score_cols if c in participant_summary.columns]

    print(
        participant_summary[participant_summary["has_final_feedback"]]
        .groupby("group")[existing]
        .mean(numeric_only=True)
        .round(3)
    )

    print("\n--- Reported issues by group ---")
    if "n_reported_issues" in participant_summary.columns:
        print(
            participant_summary[participant_summary["has_final_feedback"]]
            .groupby("group")["n_reported_issues"]
            .agg(["count", "mean", "median", "sum"])
            .round(3)
        )

    print("\n--- Free comments ---")
    comments = participant_summary[
        participant_summary["free_comment"].notna()
        | participant_summary["binary_field_comments"].notna()
    ]

    if len(comments) == 0:
        print("None")
    else:
        print(
            comments[[
                "PID",
                "group",
                "free_comment",
                "binary_field_comments",
                "followup_status",
            ]]
            .sort_values(["group", "PID"])
            .to_string(index=False)
        )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Clean and aggregate final review / feedback Google Form."
    )

    parser.add_argument(
        "--program",
        default="program_length.csv",
        help="Path to program_length.csv",
    )

    parser.add_argument(
        "--feedback",
        default="final_feedback.csv",
        help="Path to final feedback CSV",
    )

    args = parser.parse_args()

    ensure_output_dir()

    print("\nLoading program_length...")
    df_prog = load_program_length(args.program)

    print("Loading final feedback form...")
    df_feedback = load_final_feedback(args.feedback, df_prog)

    print("Building participant summary...")
    participant_summary = build_participant_summary(df_feedback, df_prog)

    save_csv(df_feedback, "final_feedback_cleaned.csv")
    save_csv(participant_summary, "final_feedback_participant_summary.csv")

    print_sanity_checks(
        df_prog=df_prog,
        df_feedback=df_feedback,
        participant_summary=participant_summary,
    )

    print("\nDone.")
    print(f"Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()