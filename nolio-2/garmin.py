import os
import re
import pandas as pd
import xml.etree.ElementTree as ET

from datetime import datetime, timezone
from fitparse import FitFile


# ============================================================
# CONFIG
# ============================================================

GARMIN_FOLDER = "/home/octavian/projects/memoire/nolio-2"
PROGRAM_LENGTH_PATH = "program_length.csv"

OUTPUT_ALL_HYPHEN = "all-sessions.csv"
OUTPUT_ALL_UNDERSCORE = "all_sessions.csv"      # compatibility with old scripts
OUTPUT_CLEAN = "all-sessions-cleaned.csv"

# Known subject-level exclusions.
# Keep empty by default; do not hard-delete participants here.
# If needed later: KNOWN_OUTLIER_IDS = [6302]
KNOWN_OUTLIER_IDS = []

# Running detection thresholds
MAX_DT = 25                 # seconds; larger gaps break continuity
MIN_DT = 0.5                # ignore duplicate/near-duplicate timestamps
RUN_MIN_SPEED = 1.5         # m/s; permissive beginner running/jogging threshold
RUN_MAX_SPEED = 6.0         # m/s; ~21.6 km/h, upper plausible running speed
MIN_DISTANCE_STEP = 1.0     # meters; below this treated as no movement
TOLERANCE = 8               # seconds of non-running tolerated inside a streak

# Cleaning / quality thresholds
MIN_CONTINUOUS_RUN_FOR_ANALYSIS = 1.0       # min
MAX_CONTINUOUS_RUN_FOR_ANALYSIS = 60.0      # min; above this is suspicious
MIN_VALID_TIME_RATIO = 0.60
MAX_GAP_BREAKS = 10
MAX_TOTAL_DURATION_MIN = 180.0              # session longer than 3h is suspicious
MIN_DISTANCE_KM_FOR_RUNNING = 0.20


# ============================================================
# HELPERS
# ============================================================

def strip_ns(tag):
    return tag.split("}")[-1]


def parse_datetime_safe(text):
    if text is None:
        return None

    text = str(text).strip()

    if not text:
        return None

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def infer_activity_type(filename):
    name = filename.lower()

    # French + English
    if (
        "running" in name
        or "course" in name
        or "course a pied" in name
        or "course à pied" in name
        or "run" in name
    ):
        return "running"

    if "walking" in name or "marche" in name or "walk" in name:
        return "walking"

    if (
        "strength" in name
        or "renforcement" in name
        or "muscu" in name
        or "fitness" in name
    ):
        return "strength"

    if "other" in name or "autre" in name:
        return "other"

    return "unknown"


def clean_group_column(df):
    if "group" in df.columns:
        df["group"] = df["group"].astype(str).str.strip()
    return df


def find_program_start_column(df_prog):
    """
    Tries to detect a real program start date if present.
    If none exists, we fall back to first Garmin date per participant.
    """
    possible = [
        "program_start_date",
        "start_date",
        "first_program_date",
        "date_start",
        "date_debut",
        "début",
        "debut",
    ]

    lower_map = {c.lower().strip(): c for c in df_prog.columns}

    for candidate in possible:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


# ============================================================
# GARMIN METRICS
# ============================================================

def compute_run_metrics(trackpoints):
    """
    Input:
        trackpoints = list of (datetime, cumulative_distance_m)

    Output:
        dict of duration, running, quality and diagnostic metrics.

    Philosophy:
        - We do NOT blindly trust Garmin continuity.
        - We compute both the outcome and the reasons why a session may be suspicious.
    """

    if not trackpoints or len(trackpoints) < 2:
        return {
            "total_duration_min": 0.0,
            "valid_duration_min": 0.0,
            "unknown_gap_time_min": 0.0,
            "max_continuous_run_min": 0.0,
            "estimated_running_time_min": 0.0,
            "distance_km": 0.0,
            "avg_speed_mps": 0.0,
            "avg_speed_kmh": 0.0,
            "median_segment_speed_mps": None,
            "p95_segment_speed_mps": None,
            "valid_time_ratio": 0.0,
            "running_time_ratio": 0.0,
            "n_trackpoints": len(trackpoints) if trackpoints else 0,
            "n_segments": 0,
            "n_valid_segments": 0,
            "n_gap_breaks": 0,
            "n_speed_breaks": 0,
            "n_zero_distance_segments": 0,
            "n_negative_distance_segments": 0,
        }

    # Ensure chronological order
    trackpoints = sorted(trackpoints, key=lambda x: x[0])

    current_streak = 0.0
    max_streak = 0.0
    slow_time = 0.0

    total_time = 0.0
    valid_time = 0.0
    running_time = 0.0
    unknown_gap_time = 0.0

    n_segments = 0
    n_valid_segments = 0
    n_gap_breaks = 0
    n_speed_breaks = 0
    n_zero_distance_segments = 0
    n_negative_distance_segments = 0

    speeds = []

    for i in range(1, len(trackpoints)):
        t1, d1 = trackpoints[i - 1]
        t2, d2 = trackpoints[i]

        dt = (t2 - t1).total_seconds()
        dd = d2 - d1

        if dt <= 0:
            continue

        n_segments += 1
        total_time += dt

        if dt <= MIN_DT:
            continue

        if dd < 0:
            n_negative_distance_segments += 1
            max_streak = max(max_streak, current_streak)
            current_streak = 0.0
            slow_time = 0.0
            continue

        if dt > MAX_DT:
            unknown_gap_time += dt
            n_gap_breaks += 1

            max_streak = max(max_streak, current_streak)
            current_streak = 0.0
            slow_time = 0.0
            continue

        n_valid_segments += 1
        valid_time += dt

        if dd < MIN_DISTANCE_STEP:
            speed = 0.0
            n_zero_distance_segments += 1
        else:
            speed = dd / dt
            speeds.append(speed)

        if RUN_MIN_SPEED <= speed <= RUN_MAX_SPEED:
            current_streak += dt
            running_time += dt
            slow_time = 0.0
        else:
            slow_time += dt

            if slow_time <= TOLERANCE:
                current_streak += dt
            else:
                n_speed_breaks += 1
                max_streak = max(max_streak, current_streak)
                current_streak = 0.0
                slow_time = 0.0

    max_streak = max(max_streak, current_streak)

    total_distance_m = trackpoints[-1][1] if trackpoints else 0.0
    distance_km = total_distance_m / 1000.0

    avg_speed_mps = total_distance_m / total_time if total_time > 0 else 0.0
    avg_speed_kmh = avg_speed_mps * 3.6

    valid_time_ratio = valid_time / total_time if total_time > 0 else 0.0
    running_time_ratio = running_time / valid_time if valid_time > 0 else 0.0

    if speeds:
        speed_series = pd.Series(speeds)
        median_segment_speed_mps = float(speed_series.median())
        p95_segment_speed_mps = float(speed_series.quantile(0.95))
    else:
        median_segment_speed_mps = None
        p95_segment_speed_mps = None

    return {
        "total_duration_min": total_time / 60.0,
        "valid_duration_min": valid_time / 60.0,
        "unknown_gap_time_min": unknown_gap_time / 60.0,
        "max_continuous_run_min": max_streak / 60.0,
        "estimated_running_time_min": running_time / 60.0,
        "distance_km": distance_km,
        "avg_speed_mps": avg_speed_mps,
        "avg_speed_kmh": avg_speed_kmh,
        "median_segment_speed_mps": median_segment_speed_mps,
        "p95_segment_speed_mps": p95_segment_speed_mps,
        "valid_time_ratio": valid_time_ratio,
        "running_time_ratio": running_time_ratio,
        "n_trackpoints": len(trackpoints),
        "n_segments": n_segments,
        "n_valid_segments": n_valid_segments,
        "n_gap_breaks": n_gap_breaks,
        "n_speed_breaks": n_speed_breaks,
        "n_zero_distance_segments": n_zero_distance_segments,
        "n_negative_distance_segments": n_negative_distance_segments,
    }


# ============================================================
# FILE PARSERS
# ============================================================

def parse_tcx(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    trackpoints = []
    start_time = None

    for elem in root.iter():
        tag = strip_ns(elem.tag)

        if tag == "Id" and start_time is None:
            parsed = parse_datetime_safe(elem.text)
            if parsed is not None:
                start_time = parsed

        if tag == "Trackpoint":
            time = None
            dist = None

            for child in elem:
                child_tag = strip_ns(child.tag)

                if child_tag == "Time":
                    time = parse_datetime_safe(child.text)

                elif child_tag == "DistanceMeters":
                    try:
                        dist = float(child.text)
                    except Exception:
                        dist = None

            if time is not None and dist is not None:
                trackpoints.append((time, dist))

    if start_time is None and trackpoints:
        start_time = trackpoints[0][0]

    metrics = compute_run_metrics(trackpoints)
    return start_time, metrics


def parse_fit(file_path):
    fitfile = FitFile(file_path)

    trackpoints = []
    start_time = None

    for msg in fitfile.get_messages("record"):
        time = None
        dist = None

        for data in msg:
            if data.name == "timestamp":
                if data.value is not None:
                    time = data.value.replace(tzinfo=timezone.utc)

                    if start_time is None:
                        start_time = time

            elif data.name == "distance":
                if data.value is not None:
                    try:
                        dist = float(data.value)
                    except Exception:
                        dist = None

        if time is not None and dist is not None:
            trackpoints.append((time, dist))

    if start_time is None and trackpoints:
        start_time = trackpoints[0][0]

    metrics = compute_run_metrics(trackpoints)
    return start_time, metrics


# ============================================================
# PROGRAM ALIGNMENT + CLEANING
# ============================================================

def load_program_info(path):
    df_prog = pd.read_csv(path)
    df_prog.columns = df_prog.columns.str.strip()

    if "PID" not in df_prog.columns:
        raise ValueError("program_length.csv must contain a PID column.")

    df_prog["PID"] = pd.to_numeric(df_prog["PID"], errors="coerce")

    if "program_length" not in df_prog.columns:
        raise ValueError("program_length.csv must contain a program_length column.")

    df_prog["program_length"] = pd.to_numeric(
        df_prog["program_length"],
        errors="coerce"
    )

    df_prog = clean_group_column(df_prog)

    start_col = find_program_start_column(df_prog)

    if start_col is not None:
        df_prog["program_start_date"] = pd.to_datetime(
            df_prog[start_col],
            errors="coerce"
        ).dt.date
    else:
        df_prog["program_start_date"] = pd.NaT

    keep_cols = ["PID", "program_length"]

    if "group" in df_prog.columns:
        keep_cols.append("group")

    keep_cols.append("program_start_date")

    return df_prog[keep_cols].drop_duplicates("PID")


def align_program_weeks(df, df_prog):
    df = df.copy()

    df = df.merge(
        df_prog,
        left_on="id",
        right_on="PID",
        how="left"
    )

    # Force date columns into proper pandas datetime64, not Python date objects
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")

    if "program_start_date" in df.columns:
        df["program_start_date_dt"] = pd.to_datetime(
            df["program_start_date"],
            errors="coerce"
        )
    else:
        df["program_start_date_dt"] = pd.NaT

    # First Garmin date per participant, using clean datetime dtype
    df["first_garmin_date_dt"] = (
        df.groupby("id")["date_dt"]
        .transform("min")
    )

    # Use real program start date if available, otherwise first Garmin date
    df["alignment_start_date_dt"] = df["program_start_date_dt"]

    missing_start = df["alignment_start_date_dt"].isna()

    df.loc[missing_start, "alignment_start_date_dt"] = df.loc[
        missing_start,
        "first_garmin_date_dt"
    ]

    df["week_alignment_source"] = "program_start_date"
    df.loc[missing_start, "week_alignment_source"] = "first_garmin_date"

    df["days_since_alignment_start"] = (
        df["date_dt"] - df["alignment_start_date_dt"]
    ).dt.days

    df["program_week"] = (df["days_since_alignment_start"] // 7) + 1
    df["program_week"] = pd.to_numeric(df["program_week"], errors="coerce")

    df["program_length"] = pd.to_numeric(df["program_length"], errors="coerce")

    df["progress"] = df["program_week"] / df["program_length"]

    df["progress_bin"] = pd.cut(
        df["progress"],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=["0-25%", "25-50%", "50-75%", "75-100%"],
        include_lowest=True
    )

    df["in_program"] = (
        (df["progress"] >= 0) &
        (df["progress"] <= 1.0)
    )

    # Nice export columns as dates, but keep internal datetime clean
    df["date"] = df["date_dt"].dt.date
    df["first_garmin_date"] = df["first_garmin_date_dt"].dt.date
    df["alignment_start_date"] = df["alignment_start_date_dt"].dt.date

    if "program_start_date_dt" in df.columns:
        df["program_start_date"] = df["program_start_date_dt"].dt.date

    return df

def add_exclusion_reason(df):
    df = df.copy()

    reasons = []

    for _, row in df.iterrows():
        r = []

        if row.get("known_outlier_pid", False):
            r.append("known_outlier_pid")

        if row.get("flag_not_running_activity", False):
            r.append("not_running_activity")

        if row.get("flag_too_short_run", False):
            r.append("too_short_run")

        if row.get("flag_implausible_continuous_run", False):
            r.append("implausible_continuous_run")

        if row.get("flag_low_valid_time_ratio", False):
            r.append("low_valid_time_ratio")

        if row.get("flag_many_gaps", False):
            r.append("many_gaps")

        if row.get("flag_too_long_session", False):
            r.append("too_long_session")

        if row.get("flag_low_distance_running", False):
            r.append("low_distance_running")

        if row.get("flag_no_program_info", False):
            r.append("no_program_info")

        if row.get("flag_outside_program", False):
            r.append("outside_program")

        if row.get("garmin_quality") != "ok":
            r.append("garmin_quality_suspect")

        reasons.append(";".join(r) if r else "included_or_no_issue")

    df["exclusion_reason"] = reasons

    return df

def add_quality_flags(df):
    df = df.copy()

    df["known_outlier_pid"] = df["id"].isin(KNOWN_OUTLIER_IDS)

    df["flag_not_running_activity"] = ~df["activity_type"].isin(["running", "unknown"])

    df["flag_too_short_run"] = (
        df["max_continuous_run_min"] <= MIN_CONTINUOUS_RUN_FOR_ANALYSIS
    )

    df["flag_implausible_continuous_run"] = (
        df["max_continuous_run_min"] > MAX_CONTINUOUS_RUN_FOR_ANALYSIS
    )

    df["flag_low_valid_time_ratio"] = (
        df["valid_time_ratio"] < MIN_VALID_TIME_RATIO
    )

    df["flag_many_gaps"] = (
        df["n_gap_breaks"] > MAX_GAP_BREAKS
    )

    df["flag_too_long_session"] = (
        df["total_duration_min"] > MAX_TOTAL_DURATION_MIN
    )

    df["flag_low_distance_running"] = (
        (df["activity_type"] == "running") &
        (df["distance_km"] < MIN_DISTANCE_KM_FOR_RUNNING)
    )

    df["flag_no_program_info"] = (
        df["program_length"].isna()
    )

    df["flag_outside_program"] = (
        df["in_program"] == False
    )

    quality_flag_cols = [
        "known_outlier_pid",
        "flag_not_running_activity",
        "flag_too_short_run",
        "flag_implausible_continuous_run",
        "flag_low_valid_time_ratio",
        "flag_many_gaps",
        "flag_too_long_session",
        "flag_low_distance_running",
        "flag_no_program_info",
    ]

    df["garmin_quality"] = "ok"

    df.loc[
        df[quality_flag_cols].any(axis=1),
        "garmin_quality"
    ] = "suspect"

    # Analysis inclusion rule:
    # - running activity
    # - inside program
    # - quality ok
    # - not known outlier
    df["include_in_garmin_analysis"] = (
        (df["activity_type"] == "running") &
        (df["in_program"]) &
        (df["garmin_quality"] == "ok") &
        (~df["known_outlier_pid"])
    )

    return df



def assign_session_numbers(df):
    df = df.sort_values(["id", "start_time"]).copy()

    df["session_global_all"] = df.groupby("id").cumcount() + 1

    df["session_in_week_all"] = (
        df.groupby(["id", "program_week"])
        .cumcount() + 1
    )

    # Running-only numbering
    df["session_global_running"] = pd.NA
    df["session_in_week_running"] = pd.NA

    running_mask = df["activity_type"] == "running"

    df.loc[running_mask, "session_global_running"] = (
        df.loc[running_mask]
        .groupby("id")
        .cumcount() + 1
    )

    df.loc[running_mask, "session_in_week_running"] = (
        df.loc[running_mask]
        .groupby(["id", "program_week"])
        .cumcount() + 1
    )

    return df


# ============================================================
# FOLDER PROCESSING
# ============================================================

def process_file(path, filename, folder_id):
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".fit":
        start_time, metrics = parse_fit(path)
    elif ext == ".tcx":
        start_time, metrics = parse_tcx(path)
    else:
        return None

    row = {
        "id": int(folder_id),
        "session": filename,
        "file_ext": ext.replace(".", ""),
        "activity_type": infer_activity_type(filename),
        "start_time": start_time,
        "date": start_time.date() if start_time else None,
    }

    row.update(metrics)

    return row


def process_garmin_folder(folder):
    rows = []

    for entry in os.scandir(folder):
        if not entry.is_dir():
            continue

        folder_id = entry.name
        folder_path = entry.path

        try:
            int(folder_id)
        except ValueError:
            print(f"Skipping non-PID folder: {folder_id}")
            continue

        print(f"Processing folder: {folder_id}")

        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)

            if not os.path.isfile(path):
                continue

            if not filename.lower().endswith((".fit", ".tcx")):
                continue

            try:
                row = process_file(path, filename, folder_id)

                if row is not None:
                    rows.append(row)

            except Exception as e:
                print(f"Error with {folder_id}/{filename}: {e}")

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    print("CURRENT DIR:", os.getcwd())

    df = process_garmin_folder(GARMIN_FOLDER)

    if df.empty:
        print("No Garmin sessions found.")
        return

    df_prog = load_program_info(PROGRAM_LENGTH_PATH)

    df = align_program_weeks(df, df_prog)
    df = add_quality_flags(df)
    df = add_exclusion_reason(df)
    df = assign_session_numbers(df)

    # Sort cleanly
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    df = df.sort_values(["id", "start_time", "session"])

    # Round numeric metrics for readability
    round_cols = [
        "total_duration_min",
        "valid_duration_min",
        "unknown_gap_time_min",
        "max_continuous_run_min",
        "estimated_running_time_min",
        "distance_km",
        "avg_speed_mps",
        "avg_speed_kmh",
        "median_segment_speed_mps",
        "p95_segment_speed_mps",
        "valid_time_ratio",
        "running_time_ratio",
        "progress",
    ]

    for col in round_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(3)

    # Preferred column order
    preferred_cols = [
        "id",
        "PID",
        "group",
        "program_length",
        "session",
        "file_ext",
        "activity_type",
        "date",
        "start_time",
        "first_garmin_date",
        "program_start_date",
        "alignment_start_date",
        "week_alignment_source",
        "days_since_alignment_start",
        "program_week",
        "progress",
        "progress_bin",
        "in_program",
        "total_duration_min",
        "valid_duration_min",
        "unknown_gap_time_min",
        "max_continuous_run_min",
        "estimated_running_time_min",
        "distance_km",
        "avg_speed_mps",
        "avg_speed_kmh",
        "median_segment_speed_mps",
        "p95_segment_speed_mps",
        "valid_time_ratio",
        "running_time_ratio",
        "n_trackpoints",
        "n_segments",
        "n_valid_segments",
        "n_gap_breaks",
        "n_speed_breaks",
        "n_zero_distance_segments",
        "n_negative_distance_segments",
        "known_outlier_pid",
        "flag_not_running_activity",
        "flag_too_short_run",
        "flag_implausible_continuous_run",
        "flag_low_valid_time_ratio",
        "flag_many_gaps",
        "flag_too_long_session",
        "flag_low_distance_running",
        "flag_no_program_info",
        "flag_outside_program",
        "garmin_quality",
        "include_in_garmin_analysis",
        "session_global_all",
        "session_in_week_all",
        "session_global_running",
        "session_in_week_running",
    ]

    existing_preferred_cols = [c for c in preferred_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_preferred_cols]
    df = df[existing_preferred_cols + remaining_cols]

    # Save all sessions with flags
    df.to_csv(OUTPUT_ALL_HYPHEN, index=False)
    df.to_csv(OUTPUT_ALL_UNDERSCORE, index=False)

    # Save cleaned analysis-ready running sessions
    df_clean = df[df["include_in_garmin_analysis"]].copy()
    df_clean.to_csv(OUTPUT_CLEAN, index=False)

    print(f"\nSaved all sessions: {OUTPUT_ALL_HYPHEN}")
    print(f"Saved compatibility copy: {OUTPUT_ALL_UNDERSCORE}")
    print(f"Saved cleaned sessions: {OUTPUT_CLEAN}")

    print("\n--- Session counts ---")
    print("All parsed sessions:", len(df))
    print("Clean analysis sessions:", len(df_clean))

    print("\n--- Activity types ---")
    print(df["activity_type"].value_counts(dropna=False))

    print("\n--- Garmin quality ---")
    print(df["garmin_quality"].value_counts(dropna=False))

    print("\n--- Included in analysis by group ---")
    if "group" in df.columns:
        print(pd.crosstab(df["group"], df["include_in_garmin_analysis"]))

    print("\n--- Top suspicious continuous runs ---")
    cols_to_show = [
        "id",
        "group",
        "session",
        "program_week",
        "progress",
        "max_continuous_run_min",
        "total_duration_min",
        "distance_km",
        "valid_time_ratio",
        "n_gap_breaks",
        "garmin_quality",
        "include_in_garmin_analysis",
    ]
    cols_to_show = [c for c in cols_to_show if c in df.columns]

    print(
        df.sort_values("max_continuous_run_min", ascending=False)
        [cols_to_show]
        .head(20)
        .to_string(index=False)
    )

    print("\n--- Exclusion reasons ---")
    print(
        df[~df["include_in_garmin_analysis"]]["exclusion_reason"]
        .value_counts()
        .head(30)
    )

    print("\n--- Exclusion reasons by group ---")
    print(
        df[~df["include_in_garmin_analysis"]]
        .groupby(["group", "exclusion_reason"])
        .size()
        .sort_values(ascending=False)
        .head(40)
    )

    print("\n--- OK but excluded ---")
    print(
        df[
            (df["garmin_quality"] == "ok") &
            (~df["include_in_garmin_analysis"])
        ][[
            "id", "group", "session", "activity_type",
            "program_week", "progress", "in_program",
            "max_continuous_run_min", "total_duration_min",
            "known_outlier_pid",
            "flag_not_running_activity",
            "flag_outside_program",
            "flag_too_long_session",
            "flag_implausible_continuous_run",
            "exclusion_reason"
        ]]
        .head(80)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()