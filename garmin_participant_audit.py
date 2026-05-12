import os
import pandas as pd

GARMIN_FOLDER = "/home/octavian/projects/memoire/nolio-2"
ALL_PATH = "all-sessions.csv"
CLEAN_PATH = "all-sessions-cleaned.csv"
PROGRAM_PATH = "program_length.csv"


def folder_pids(path):
    pids = []
    for entry in os.scandir(path):
        if entry.is_dir():
            try:
                pids.append(int(entry.name))
            except ValueError:
                pass
    return sorted(pids)


def main():
    folders = pd.DataFrame({"id": folder_pids(GARMIN_FOLDER)})

    df_all = pd.read_csv(ALL_PATH)
    df_clean = pd.read_csv(CLEAN_PATH)
    df_prog = pd.read_csv(PROGRAM_PATH)

    df_all["id"] = pd.to_numeric(df_all["id"], errors="coerce")
    df_clean["id"] = pd.to_numeric(df_clean["id"], errors="coerce")
    df_prog["PID"] = pd.to_numeric(df_prog["PID"], errors="coerce")
    df_prog["group"] = df_prog["group"].astype(str).str.strip()

    df_prog = df_prog.rename(columns={"PID": "id"})

    # Basic participant-level counts from all parsed sessions
    all_summary = (
        df_all.groupby("id")
        .agg(
            parsed_sessions=("session", "count"),
            running_sessions=("activity_type", lambda x: (x == "running").sum()),
            walking_sessions=("activity_type", lambda x: (x == "walking").sum()),
            strength_sessions=("activity_type", lambda x: (x == "strength").sum()),
            unknown_sessions=("activity_type", lambda x: (x == "unknown").sum()),
            ok_sessions=("garmin_quality", lambda x: (x == "ok").sum()),
            suspect_sessions=("garmin_quality", lambda x: (x == "suspect").sum()),
            included_sessions=("include_in_garmin_analysis", lambda x: x.astype(bool).sum()),
            in_program_sessions=("in_program", lambda x: x.astype(bool).sum()),
            first_date=("date", "min"),
            last_date=("date", "max"),
            max_run_raw=("max_continuous_run_min", "max"),
        )
        .reset_index()
    )

    # Clean/session-level participants
    clean_summary = (
        df_clean.groupby("id")
        .agg(
            clean_sessions=("session", "count"),
            clean_weeks=("program_week", "nunique"),
            max_run_clean=("max_continuous_run_min", "max"),
            median_run_clean=("max_continuous_run_min", "median"),
        )
        .reset_index()
    )

    audit = (
        folders
        .merge(df_prog[["id", "group", "program_length"]], on="id", how="left")
        .merge(all_summary, on="id", how="left")
        .merge(clean_summary, on="id", how="left")
    )

    # Fill counts
    count_cols = [
        "parsed_sessions",
        "running_sessions",
        "walking_sessions",
        "strength_sessions",
        "unknown_sessions",
        "ok_sessions",
        "suspect_sessions",
        "included_sessions",
        "in_program_sessions",
        "clean_sessions",
        "clean_weeks",
    ]

    for col in count_cols:
        if col in audit.columns:
            audit[col] = audit[col].fillna(0).astype(int)

    audit["has_folder"] = True
    audit["has_parsed_data"] = audit["parsed_sessions"] > 0
    audit["has_clean_analysis_data"] = audit["clean_sessions"] > 0

    def exclusion_reason(row):
        if not row["has_parsed_data"]:
            return "no parsed FIT/TCX sessions"
        if row["running_sessions"] == 0:
            return "no sessions classified as running"
        if row["in_program_sessions"] == 0:
            return "no sessions inside normalized program window"
        if row["ok_sessions"] == 0:
            return "all sessions flagged suspect"
        if row["included_sessions"] == 0:
            return "no running+in_program+ok sessions"
        return "included"

    audit["audit_status"] = audit.apply(exclusion_reason, axis=1)

    audit = audit.sort_values(["group", "id"])

    print("\n=== Folder participants ===")
    print(len(folders), folders["id"].tolist())

    print("\n=== Participants by group / clean inclusion ===")
    print(pd.crosstab(audit["group"], audit["has_clean_analysis_data"], dropna=False))

    print("\n=== Audit status ===")
    print(audit["audit_status"].value_counts(dropna=False))

    print("\n=== Full audit table ===")
    cols = [
        "id", "group", "program_length",
        "parsed_sessions",
        "running_sessions",
        "walking_sessions",
        "strength_sessions",
        "unknown_sessions",
        "ok_sessions",
        "suspect_sessions",
        "in_program_sessions",
        "included_sessions",
        "clean_sessions",
        "clean_weeks",
        "max_run_raw",
        "max_run_clean",
        "audit_status",
    ]

    print(audit[cols].to_string(index=False))

    audit.to_csv("garmin_participant_audit.csv", index=False)
    print("\nSaved: garmin_participant_audit.csv")


if __name__ == "__main__":
    main()