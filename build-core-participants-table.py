import os
import argparse
import numpy as np
import pandas as pd


# ============================================================
# Helpers
# ============================================================

def clean_pid(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace("PP-", "", regex=False)
        .str.replace("pp-", "", regex=False)
        .str.extract(r"(\d+)", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
    )


def to_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def safe_read_csv(path, required=False):
    if path is None:
        return None

    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        print(f"Warning: file not found, skipping: {path}")
        return None

    return pd.read_csv(path, low_memory=False)


def first_existing_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def ensure_pid(df, pid_col="PID"):
    if df is None:
        return None

    if pid_col not in df.columns:
        raise KeyError(f"Missing PID column: {pid_col}")

    df = df.copy()
    df["PID"] = clean_pid(df[pid_col])
    df = df.dropna(subset=["PID"])
    df["PID"] = df["PID"].astype(int)
    return df


def bool_to_nullable(series):
    if series is None:
        return series
    return series.astype("boolean")


# ============================================================
# Load base participant list
# ============================================================

def load_program_participants(path):
    df = safe_read_csv(path, required=True)
    df.columns = df.columns.str.strip()

    required = ["PID", "group", "program_length"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")

    df = df[required].copy()
    df = ensure_pid(df)

    df["group"] = df["group"].astype(str).str.strip().str.upper()
    df["program_length"] = pd.to_numeric(df["program_length"], errors="coerce")

    df = df.dropna(subset=["PID", "group", "program_length"])
    df["program_length"] = df["program_length"].astype(int)

    df = df.drop_duplicates(subset=["PID"], keep="first")

    return df.sort_values(["group", "PID"])


# ============================================================
# Signup metadata
# ============================================================

def load_signup_summary(path):
    df = safe_read_csv(path)
    if df is None:
        return None

    df.columns = df.columns.str.strip()
    df = ensure_pid(df)

    keep = [
        "PID",
        "signup_group",
        "has_program_info",
        "is_study_participant",
        "group_mismatch_signup_vs_program",
        "age",
        "postpartum_weeks",
        "children_count",
        "delivery_mode",
        "tear_grade",
        "status",
    ]

    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    df = to_numeric(df, ["age", "postpartum_weeks", "children_count"])

    df = df.drop_duplicates(subset=["PID"], keep="first")

    return df


# ============================================================
# Weekly forms participant summary
# ============================================================

def load_forms_participant_summary(path):
    df = safe_read_csv(path)
    if df is None:
        return None

    df.columns = df.columns.str.strip()
    df = ensure_pid(df)

    keep = [
        "PID",
        "has_weekly_forms",
        "first_reported_week",
        "last_reported_week",
        "n_weeks_reported",
        "max_progress_reported",
        "forms_completion_rate",
        "forms_coverage",
        "completed_by_forms",
        "mean_adherence_ratio",
        "mean_sessions_done",
        "mean_fatigue_score",
        "mean_symptom_score",
        "mean_sleep_hours",
        "mean_sleep_quality",
        "clarity_mean",
        "intensity_respected_mean",
        "satisfaction_mean",
        "confidence_mean",
        "recommendation_mean",
    ]

    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    numeric_cols = [
        "first_reported_week",
        "last_reported_week",
        "n_weeks_reported",
        "max_progress_reported",
        "forms_completion_rate",
        "forms_coverage",
        "mean_adherence_ratio",
        "mean_sessions_done",
        "mean_fatigue_score",
        "mean_symptom_score",
        "mean_sleep_hours",
        "mean_sleep_quality",
        "clarity_mean",
        "intensity_respected_mean",
        "satisfaction_mean",
        "confidence_mean",
        "recommendation_mean",
    ]

    df = to_numeric(df, numeric_cols)

    for col in ["has_weekly_forms", "completed_by_forms"]:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    df = df.drop_duplicates(subset=["PID"], keep="first")

    return df


# ============================================================
# Final feedback summary
# ============================================================

def load_final_feedback_summary(path):
    print("\n--- DEBUG load_final_feedback_summary ---")
    print(f"Received path: {repr(path)}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Absolute path: {os.path.abspath(path) if path else None}")
    print(f"Exists? {os.path.exists(path) if path else False}")

    if path:
        parent = os.path.dirname(os.path.abspath(path)) or "."
        print(f"Parent directory exists? {os.path.exists(parent)}")
        if os.path.exists(parent):
            print("Nearby CSV files:")
            for f in os.listdir(parent):
                if "feedback" in f.lower() or "final" in f.lower():
                    print("  ", f)

    if path is None or not os.path.exists(path):
        print(f"Final feedback file not found: {path}")
        return None

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    print(f"Loaded final feedback rows: {len(df)}")
    print(f"Loaded final feedback columns: {list(df.columns)}")

    df["PID"] = (
        df["PID"]
        .astype(str)
        .str.strip()
        .str.replace("PP-", "", regex=False)
        .str.replace("pp-", "", regex=False)
        .str.extract(r"(\d+)", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    df = df.dropna(subset=["PID"])
    df["PID"] = df["PID"].astype(int)

    if "has_final_feedback" not in df.columns:
        if "n_final_feedback_responses" in df.columns:
            df["has_final_feedback"] = df["n_final_feedback_responses"].fillna(0).gt(0)
        elif "mean_final_score" in df.columns:
            df["has_final_feedback"] = df["mean_final_score"].notna()
        else:
            df["has_final_feedback"] = True

    print(f"Final feedback unique PIDs after cleaning: {df['PID'].nunique()}")

    return df


# ============================================================
# Garmin participant summary
# ============================================================

def build_garmin_participant_summary(path):
    df = safe_read_csv(path)
    if df is None:
        return None

    df.columns = df.columns.str.strip()

    # Some Garmin outputs contain both id and PID.
    if "PID" not in df.columns and "id" in df.columns:
        df["PID"] = df["id"]

    df = ensure_pid(df)

    # Only use included Garmin data if the column exists.
    # This does NOT exclude participants from the final table.
    if "include_in_garmin_analysis" in df.columns:
        included = df["include_in_garmin_analysis"].astype(str).str.lower().isin(["true", "1", "yes"])
        df_included = df[included].copy()
    else:
        df_included = df.copy()

    numeric_cols = [
        "program_week",
        "progress",
        "total_duration_min",
        "valid_duration_min",
        "max_continuous_run_min",
        "estimated_running_time_min",
        "distance_km",
        "avg_speed_kmh",
        "running_time_ratio",
        "valid_time_ratio",
    ]

    df_included = to_numeric(df_included, numeric_cols)

    if len(df_included) == 0:
        return pd.DataFrame(columns=[
            "PID",
            "has_garmin_data",
            "n_garmin_sessions_included",
            "garmin_first_week",
            "garmin_last_week",
            "garmin_max_progress",
            "garmin_total_distance_km",
            "garmin_total_running_time_min",
            "garmin_mean_run",
            "garmin_max_run",
            "garmin_last_run",
            "reached_30min_ever",
        ])

    agg = {
        "max_continuous_run_min": ["count", "mean", "max"],
    }

    if "program_week" in df_included.columns:
        agg["program_week"] = ["min", "max"]

    if "progress" in df_included.columns:
        agg["progress"] = "max"

    if "distance_km" in df_included.columns:
        agg["distance_km"] = "sum"

    if "estimated_running_time_min" in df_included.columns:
        agg["estimated_running_time_min"] = "sum"

    summary = df_included.groupby("PID").agg(agg)

    summary.columns = [
        "_".join([str(x) for x in col if x])
        for col in summary.columns.to_flat_index()
    ]

    summary = summary.reset_index()

    rename_map = {
        "max_continuous_run_min_count": "n_garmin_sessions_included",
        "max_continuous_run_min_mean": "garmin_mean_run",
        "max_continuous_run_min_max": "garmin_max_run",
        "program_week_min": "garmin_first_week",
        "program_week_max": "garmin_last_week",
        "progress_max": "garmin_max_progress",
        "distance_km_sum": "garmin_total_distance_km",
        "estimated_running_time_min_sum": "garmin_total_running_time_min",
    }

    summary = summary.rename(columns=rename_map)

    # Last Garmin run by chronological order if possible.
    sort_cols = ["PID"]
    if "date" in df_included.columns:
        df_included["date_parsed"] = pd.to_datetime(df_included["date"], errors="coerce")
        sort_cols.append("date_parsed")
    elif "start_time" in df_included.columns:
        df_included["date_parsed"] = pd.to_datetime(df_included["start_time"], errors="coerce")
        sort_cols.append("date_parsed")
    elif "program_week" in df_included.columns:
        sort_cols.append("program_week")

    last_run = (
        df_included.sort_values(sort_cols)
        .groupby("PID")
        .tail(1)[["PID", "max_continuous_run_min"]]
        .rename(columns={"max_continuous_run_min": "garmin_last_run"})
    )

    summary = summary.merge(last_run, on="PID", how="left")

    summary["has_garmin_data"] = summary["n_garmin_sessions_included"].fillna(0).gt(0)
    summary["reached_30min_ever"] = summary["garmin_max_run"] >= 30

    return summary


# ============================================================
# Build core table
# ============================================================


def merge_simple(core, df, name):
    core = core.copy()

    if df is None:
        print(f"\nSkipping {name}: file not loaded")
        return core

    df = df.copy()

    core["PID"] = pd.to_numeric(core["PID"], errors="coerce").astype("Int64")
    df["PID"] = pd.to_numeric(df["PID"], errors="coerce").astype("Int64")

    print(f"\nMerging {name}:")
    print(f"  rows: {len(df)}")
    print(f"  unique PIDs in source: {df['PID'].nunique()}")
    print(f"  PID overlap with core: {len(set(core['PID'].dropna()) & set(df['PID'].dropna()))}")

    df = df.drop_duplicates("PID", keep="first")

    # Keep only columns that do not already exist in core.
    # This prevents later merges from nuking previous data.
    keep_cols = ["PID"] + [c for c in df.columns if c != "PID" and c not in core.columns]

    dropped_cols = [c for c in df.columns if c != "PID" and c in core.columns]
    if dropped_cols:
        print(f"  skipped overlapping columns from {name}: {dropped_cols}")

    return core.merge(df[keep_cols], on="PID", how="left")

def merge_final_feedback(core, final_feedback):
    core = core.copy()

    if final_feedback is None or len(final_feedback) == 0:
        core["has_final_feedback"] = False
        return core

    final_feedback = final_feedback.copy()

    # Clean merge keys aggressively
    core["PID"] = pd.to_numeric(core["PID"], errors="coerce").astype("Int64")
    final_feedback["PID"] = pd.to_numeric(final_feedback["PID"], errors="coerce").astype("Int64")

    core["group"] = core["group"].astype(str).str.strip().str.upper()
    final_feedback["group"] = final_feedback["group"].astype(str).str.strip().str.upper()

    if "program_length" in core.columns:
        core["program_length"] = pd.to_numeric(core["program_length"], errors="coerce").astype("Int64")

    if "program_length" in final_feedback.columns:
        final_feedback["program_length"] = pd.to_numeric(final_feedback["program_length"], errors="coerce").astype("Int64")

    # Remove existing final-feedback columns from core if they exist
    feedback_cols = [
    "has_final_feedback",
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

    # Issue categories
    "issue_contraindications",
    "issue_functional_tests",
    "issue_characteristics",
    "issue_criteria",
    "issue_symptoms",
    "issue_test_ease",
    "issue_program_choice",

    # Optional comment fields
    "issue_contraindications_comment",
    "issue_functional_tests_comment",
    "issue_characteristics_comment",
    "issue_criteria_comment",
    "issue_symptoms_comment",
    "issue_test_ease_comment",
    "issue_program_choice_comment",
    "safety_advice_clear_comment",
]

    existing_feedback_cols = [c for c in feedback_cols if c in core.columns]
    core = core.drop(columns=existing_feedback_cols)

    # Keep only useful feedback columns
    keep_cols = ["PID"] + [c for c in feedback_cols if c in final_feedback.columns]

    final_feedback_small = final_feedback[keep_cols].drop_duplicates("PID", keep="first")

    core = core.merge(
        final_feedback_small,
        on="PID",
        how="left",
        indicator="final_feedback_merge_status",
    )

    # Recompute flag from actual merged data, not from the source/dummy flag
    if "n_final_feedback_responses" in core.columns:
        core["has_final_feedback"] = core["n_final_feedback_responses"].fillna(0).gt(0)
    elif "mean_final_score" in core.columns:
        core["has_final_feedback"] = core["mean_final_score"].notna()
    else:
        core["has_final_feedback"] = core["final_feedback_merge_status"].eq("both")

    return core
def build_core_table(
    program_path,
    signup_path,
    forms_summary_path,
    final_feedback_path,
    garmin_path,
):
    core = load_program_participants(program_path)

    signup = load_signup_summary(signup_path)
    forms = load_forms_participant_summary(forms_summary_path)
    final_feedback = load_final_feedback_summary(final_feedback_path)
    garmin = build_garmin_participant_summary(garmin_path)

    print("\n--- Core participants from program file ---")
    print(f"Rows: {len(core)}")
    print(core.groupby("group")["PID"].nunique())

    core = merge_simple(core, signup, "signup")
    core = merge_simple(core, forms, "weekly forms participant summary")
    core = merge_simple(core, garmin, "Garmin participant summary")

    # Merge final feedback LAST so nothing can overwrite/delete it afterwards.
    if final_feedback is not None:
        print("\n--- Final feedback loaded ---")
        print(f"Rows: {len(final_feedback)}")
        print(f"Unique PIDs: {final_feedback['PID'].nunique()}")

        available_preview_cols = [
            c for c in [
                "PID",
                "group",
                "program_length",
                "n_final_feedback_responses",
                "has_final_feedback",
                "mean_final_score",
            ]
            if c in final_feedback.columns
        ]

        print(final_feedback[available_preview_cols].head().to_string(index=False))

        core = merge_final_feedback(core, final_feedback)

        print("\n--- Final feedback merge status ---")
        if "final_feedback_merge_status" in core.columns:
            print(core["final_feedback_merge_status"].value_counts(dropna=False))

        print("\n--- Has final feedback immediately after merge ---")
        print(core.groupby(["group", "has_final_feedback"])["PID"].nunique())
    else:
        print("\nSkipping final feedback: file not loaded")
        core["has_final_feedback"] = False

    # Fill booleans sensibly.
    if "has_weekly_forms" in core.columns:
        core["has_weekly_forms"] = core["has_weekly_forms"].fillna(False).astype(bool)
    else:
        core["has_weekly_forms"] = False

    if "completed_by_forms" in core.columns:
        core["completed_by_forms"] = core["completed_by_forms"].fillna(False).astype(bool)

    if "has_garmin_data" in core.columns:
        core["has_garmin_data"] = core["has_garmin_data"].fillna(False).astype(bool)
    else:
        core["has_garmin_data"] = False

    if "reached_30min_ever" in core.columns:
        core["reached_30min_ever"] = core["reached_30min_ever"].fillna(False).astype(bool)

    if "any_reported_issue" in core.columns:
        core["any_reported_issue"] = core["any_reported_issue"].fillna(False).astype(bool)

    if "n_final_feedback_responses" in core.columns:
        core["has_final_feedback"] = core["n_final_feedback_responses"].fillna(0).gt(0)
    else:
        core["has_final_feedback"] = False

    core["has_any_forms_or_feedback"] = (
        core["has_weekly_forms"].astype(bool)
        | core["has_final_feedback"].astype(bool)
    )

    preferred_order = [
        "PID",
        "group",
        "program_length",

        "signup_group",
        "has_program_info",
        "is_study_participant",
        "group_mismatch_signup_vs_program",
        "age",
        "postpartum_weeks",
        "children_count",
        "delivery_mode",
        "tear_grade",
        "status",

        "has_weekly_forms",
        "first_reported_week",
        "last_reported_week",
        "n_weeks_reported",
        "max_progress_reported",
        "forms_completion_rate",
        "forms_coverage",
        "completed_by_forms",
        "mean_adherence_ratio",
        "mean_sessions_done",
        "clarity_mean",
        "intensity_respected_mean",
        "satisfaction_mean",
        "confidence_mean",
        "recommendation_mean",
        "mean_fatigue_score",
        "mean_symptom_score",
        "mean_sleep_hours",
        "mean_sleep_quality",

        "has_final_feedback",
        "timestamp",
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
        "issue_contraindications",
        "issue_functional_tests",
        "issue_characteristics",
        "issue_criteria",
        "issue_symptoms",
        "issue_test_ease",
        "issue_program_choice",

        "has_garmin_data",
        "n_garmin_sessions_included",
        "garmin_first_week",
        "garmin_last_week",
        "garmin_max_progress",
        "garmin_total_distance_km",
        "garmin_total_running_time_min",
        "garmin_mean_run",
        "garmin_max_run",
        "garmin_last_run",
        "reached_30min_ever",

        "has_any_forms_or_feedback",
    ]

    ordered = [c for c in preferred_order if c in core.columns]
    remaining = [c for c in core.columns if c not in ordered]
    core = core[ordered + remaining]

    return core.sort_values(["group", "PID"])


# ============================================================
# Sanity checks
# ============================================================

def print_sanity_checks(core):
    print("\n--- Core participant table ---")
    print(f"Total participants: {core['PID'].nunique()}")

    print("\n--- Participants by group ---")
    print(core.groupby("group")["PID"].nunique())

    if "has_weekly_forms" in core.columns:
        print("\n--- Has weekly forms by group ---")
        print(core.groupby(["group", "has_weekly_forms"])["PID"].nunique())

    if "has_final_feedback" in core.columns:
        print("\n--- Has final feedback by group ---")
        print(core.groupby(["group", "has_final_feedback"])["PID"].nunique())

    if "has_garmin_data" in core.columns:
        print("\n--- Has Garmin data by group ---")
        print(core.groupby(["group", "has_garmin_data"])["PID"].nunique())

    print("\n--- Participants missing weekly forms ---")
    if "has_weekly_forms" in core.columns:
        missing = core[~core["has_weekly_forms"]]
        if len(missing):
            print(missing[["PID", "group", "program_length"]].to_string(index=False))
        else:
            print("None")

    print("\n--- Participants missing final feedback ---")
    if "has_final_feedback" in core.columns:
        missing = core[~core["has_final_feedback"]]
        if len(missing):
            print(missing[["PID", "group", "program_length"]].to_string(index=False))
        else:
            print("None")

    print("\n--- Participants missing Garmin data ---")
    if "has_garmin_data" in core.columns:
        missing = core[~core["has_garmin_data"]]
        if len(missing):
            print(missing[["PID", "group", "program_length"]].to_string(index=False))
        else:
            print("None")

    print("\n--- Preview ---")
    preview_cols = [
        "PID",
        "group",
        "program_length",
        "has_weekly_forms",
        "forms_coverage",
        "mean_adherence_ratio",
        "satisfaction_mean",
        "confidence_mean",
        "recommendation_mean",
        "has_final_feedback",
        "mean_final_score",
        "any_reported_issue",
        "has_garmin_data",
        "garmin_max_run",
    ]

    preview_cols = [c for c in preview_cols if c in core.columns]
    print(core[preview_cols].to_string(index=False))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build one core participant-level analysis table from cleaned CSV outputs."
    )

    parser.add_argument(
        "--program",
        default="forms_output/program_length_cleaned.csv",
        help="Path to cleaned program length CSV.",
    )

    parser.add_argument(
        "--signup",
        default="forms_output/forms_signup_cleaned.csv",
        help="Path to cleaned signup CSV.",
    )

    parser.add_argument(
        "--forms-summary",
        default="forms_output/forms_participant_summary.csv",
        help="Path to participant-level weekly forms summary CSV.",
    )

    parser.add_argument(
        "--final-feedback",
        default="final_feedback_output/final_feedback_participant_summary.csv",
        help="Path to final feedback participant summary CSV.",
    )

    parser.add_argument(
        "--garmin",
        default="all-sessions-cleaned.csv",
        help="Path to cleaned Garmin sessions CSV.",
    )

    parser.add_argument(
        "--output",
        default="core_participant_table.csv",
        help="Output CSV path.",
    )

    args = parser.parse_args()

    core = build_core_table(
        program_path=args.program,
        signup_path=args.signup,
        forms_summary_path=args.forms_summary,
        final_feedback_path=args.final_feedback,
        garmin_path=args.garmin,
    )

    core.to_csv(args.output, index=False)
    print(f"\nSaved core participant table: {args.output}")

    print_sanity_checks(core)


if __name__ == "__main__":
    main()