import os
import re
import unicodedata
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "forms_output"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

PROGRESS_BINS = [0, 0.25, 0.5, 0.75, 1.0]
PROGRESS_LABELS = ["0-25%", "25-50%", "50-75%", "75-100%"]

GROUP_COLORS = {
    "A": "blue",
    "B": "red",
}

WEEKLY_QUESTION_SPECS = {
    "sessions_done": {
        "label": "J'ai pu réaliser les séances prévues",
        "match": [
            "j'ai pu realiser les seances prevues",
            "j ai pu realiser les seances prevues",
        ],
        "ylim": (0, 5),
    },
    "clarity": {
        "label": "Les explications fournies étaient claires",
        "match": [
            "les explications fournis etaient claires",
            "les explications fournies etaient claires",
        ],
        "ylim": (0, 5),
    },
    "intensity_respected": {
        "label": "J'ai respecté l'intensité / les consignes",
        "match": [
            "j'ai respecte l'intensite/les consignes",
            "j ai respecte l intensite les consignes",
            "j'ai respecte l'intensite les consignes",
        ],
        "ylim": (0, 5),
    },
    "satisfaction": {
        "label": "Satisfaction globale de la séance",
        "match": [
            "satisfaction globale de la seance",
        ],
        "ylim": (0, 5),
    },
    "confidence": {
        "label": "Je me sens en confiance pour poursuivre le programme",
        "match": [
            "je me sens en confiance pour poursuivre le programme",
        ],
        "ylim": (0, 5),
    },
    "recommendation": {
        "label": "Je recommanderais ces séances à une amie",
        "match": [
            "je recommanderais ces seances a une amie",
        ],
        "ylim": (0, 5),
    },
}

FATIGUE_QUESTION_SPECS = {
    "fatigue_wakeup": {
        "label": "Fatiguée au réveil",
        "match": ["je me sens fatiguee au reveil"],
    },
    "fatigue_persistent": {
        "label": "Fatigue persistante",
        "match": ["je ressens une fatigue persistante"],
    },
    "sleepiness_daytime": {
        "label": "Forte envie de dormir en journée",
        "match": ["j ai une forte envie de dormir en journee"],
    },
    "malaise_general": {
        "label": "Mal-être général",
        "match": ["je me sens mal en general"],
    },
    "exhaustion": {
        "label": "Épuisement",
        "match": ["je me sens epuisee"],
    },
    "irritability": {
        "label": "Irritabilité",
        "match": ["je me sens irritable"],
    },
    "agitation": {
        "label": "Agitation",
        "match": ["je me sens agitee"],
    },
    "anxiety": {
        "label": "Anxiété",
        "match": ["je me sens anxieuse"],
    },
    "depressed": {
        "label": "Humeur déprimée",
        "match": ["je me sens deprimee"],
    },
    "errors": {
        "label": "Erreurs fréquentes",
        "match": ["je tendance a faire beaucoup d erreurs", "j ai tendance a faire beaucoup d erreurs"],
    },
    "motivation_lack": {
        "label": "Manque de motivation",
        "match": ["je manque de motivation"],
    },
    "concentration_difficulty": {
        "label": "Difficulté de concentration",
        "match": ["j ai du mal a me concentrer"],
    },
}

LIKERT_TEXT_MAP = {
    "jamais": 0,
    "rarement": 1,
    "parfois": 2,
    "souvent": 3,
    "tres souvent": 4,
    "très souvent": 4,
    "toujours": 4,
}

SIGNUP_COLUMN_SPECS = {
    "PID": ["colonne 1"],
    "signup_group": ["groupe"],
    "age": ["votre age"],
    "postpartum_weeks": ["a combien de semaines de post-partum etes-vous"],
    "children_count": ["combien d'enfants avec vous", "combien d enfants avec vous"],
    "delivery_mode": ["mode d'accouchement de votre dernier accouchement"],
    "tear_grade": ["avez-vous eu une dechirure"],
    "status": ["statut"],
}


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_output_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


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
    x = x.strip()

    return x


def normalize_column_name(x):
    x = normalize_text(x)
    x = x.replace("[", " ")
    x = x.replace("]", " ")
    x = x.replace(":", " ")
    x = x.replace("?", " ")
    x = x.replace(",", " ")
    x = x.replace(";", " ")
    x = x.replace("/", " ")
    x = x.replace("(", " ")
    x = x.replace(")", " ")
    x = x.replace("-", " ")
    x = x.replace("_", " ")
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


def to_numeric_clean(series):
    return pd.to_numeric(series, errors="coerce")


def clean_group(series):
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": np.nan, "": np.nan})
    )


def clean_children_count(series):
    s = series.astype(str).str.strip().str.lower()

    s = (
        s.str.replace("4 ou plus", "4", regex=False)
        .str.replace("4+", "4", regex=False)
        .str.extract(r"(\d+)", expand=False)
    )

    return pd.to_numeric(s, errors="coerce")


def parse_datetime_flexible(series):
    return pd.to_datetime(series, errors="coerce", dayfirst=True, utc=False)


def parse_sleep_hours(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text == "":
        return np.nan

    # Handles Google Forms duration such as "04:00:00"
    if re.match(r"^\d{1,2}:\d{2}:\d{2}$", text):
        h, m, s = text.split(":")
        return int(h) + int(m) / 60 + int(s) / 3600

    # Handles "4", "4.5", "4,5"
    text = text.replace(",", ".")
    match = re.search(r"\d+(\.\d+)?", text)

    if match:
        return float(match.group(0))

    return np.nan


def map_likert_text(value):
    if pd.isna(value):
        return np.nan

    text = normalize_text(value)

    if text == "":
        return np.nan

    if text in LIKERT_TEXT_MAP:
        return LIKERT_TEXT_MAP[text]

    numeric = pd.to_numeric(text.replace(",", "."), errors="coerce")
    return numeric


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


def find_columns_containing(df, required_fragments=None, optional_fragments=None):
    required_fragments = required_fragments or []
    optional_fragments = optional_fragments or []

    required_fragments = [normalize_column_name(x) for x in required_fragments]
    optional_fragments = [normalize_column_name(x) for x in optional_fragments]

    found = []

    for col in df.columns:
        c = normalize_column_name(col)

        if all(fragment in c for fragment in required_fragments):
            if not optional_fragments or any(fragment in c for fragment in optional_fragments):
                found.append(col)

    return found


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

    return df


# ============================================================
# LOAD SIGNUP FORM
# ============================================================

def load_signup(path, df_prog):
    df_raw = pd.read_csv(path, low_memory=False)
    df_raw.columns = df_raw.columns.str.strip()

    out = pd.DataFrame()

    for clean_name, candidates in SIGNUP_COLUMN_SPECS.items():
        col = find_column(df_raw, candidates, required=(clean_name == "PID"))

        if col is None:
            out[clean_name] = np.nan
            warnings.warn(f"Signup column not found for: {clean_name}")
        else:
            out[clean_name] = df_raw[col]

    out["PID"] = clean_pid(out["PID"])
    out = out.dropna(subset=["PID"])
    out["PID"] = out["PID"].astype(int)

    out["signup_group"] = clean_group(out["signup_group"])
    out["age"] = to_numeric_clean(out["age"])
    out["postpartum_weeks"] = to_numeric_clean(out["postpartum_weeks"])
    out["children_count"] = clean_children_count(out["children_count"])

    out["delivery_mode"] = out["delivery_mode"].astype(str).str.strip().replace({"nan": np.nan})
    out["tear_grade"] = out["tear_grade"].astype(str).str.strip().replace({"nan": np.nan})
    out["status"] = out["status"].astype(str).str.strip().replace({"nan": np.nan})

    # Keep only one signup row per PID.
    out = out.drop_duplicates(subset=["PID"], keep="first")

    # Merge official program info.
    out = out.merge(df_prog, on="PID", how="left")

    out["group_mismatch_signup_vs_program"] = (
        out["signup_group"].notna()
        & out["group"].notna()
        & (out["signup_group"] != out["group"])
    )

    out["has_program_info"] = out["program_length"].notna()
    out["is_study_participant"] = out["has_program_info"]

    cols = [
        "PID",
        "group",
        "signup_group",
        "program_length",
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

    return out[cols].sort_values(["group", "PID"], na_position="last")


# ============================================================
# LOAD WEEKLY CHECK-IN FORM
# ============================================================

def load_weekly(path, df_prog, df_signup):
    df_raw = pd.read_csv(path, low_memory=False)
    df_raw.columns = df_raw.columns.str.strip()

    pid_col = find_column(df_raw, ["Votre PID", "PID"], required=True)
    week_col = find_column(
        df_raw,
        ["A quelle semaine du programme êtes-vous arrivés", "semaine du programme"],
        required=True,
    )

    timestamp_col = find_column(df_raw, ["Horodateur", "Timestamp"], required=False)
    limitation_col = find_column(
        df_raw,
        [
            "Si vous n'avez pas pu effectuer au moins 2 entrainements cette semaine",
            "raison(s) qui ont limité votre activité",
        ],
        required=False,
    )
    comments_col = find_column(df_raw, ["Commentaires libres"], required=False)

    out = pd.DataFrame()

    out["PID"] = clean_pid(df_raw[pid_col])
    out["week"] = to_numeric_clean(df_raw[week_col])

    if timestamp_col:
        out["timestamp"] = parse_datetime_flexible(df_raw[timestamp_col])
    else:
        out["timestamp"] = pd.NaT

    if limitation_col:
        out["activity_limitation_reason"] = df_raw[limitation_col].astype(str).str.strip()
    else:
        out["activity_limitation_reason"] = ""

    if comments_col:
        out["free_comment"] = df_raw[comments_col].astype(str).str.strip()
    else:
        out["free_comment"] = ""

    # Main weekly satisfaction/adherence questions.
    for clean_name, spec in WEEKLY_QUESTION_SPECS.items():
        col = find_column(df_raw, spec["match"], required=False)

        if col is None:
            out[clean_name] = np.nan
            warnings.warn(f"Weekly question column not found: {clean_name}")
        else:
            out[clean_name] = to_numeric_clean(df_raw[col])

    # Fatigue / psychological state questions.
    for clean_name, spec in FATIGUE_QUESTION_SPECS.items():
        col = find_column(df_raw, spec["match"], required=False)

        if col is None:
            out[clean_name] = np.nan
        else:
            out[clean_name] = df_raw[col].apply(map_likert_text)

    # Sleep variables.
    sleep_hours_col = find_column(
        df_raw,
        ["Combien d'heure dormez-vous par nuit", "Combien d heures dormez vous par nuit"],
        required=False,
    )

    sleep_quality_col = find_column(
        df_raw,
        ["Comment évalueriez vous la qualité de votre sommeil"],
        required=False,
    )

    nocturnal_wakeups_col = find_column(
        df_raw,
        ["Avez-vous eu des réveils nocturnes"],
        required=False,
    )

    restorative_sleep_col = find_column(
        df_raw,
        ["Trouvez-vous votre sommeil réparateur", "fatigue au réveil"],
        required=False,
    )

    if sleep_hours_col:
        out["sleep_hours"] = df_raw[sleep_hours_col].apply(parse_sleep_hours)
    else:
        out["sleep_hours"] = np.nan

    if sleep_quality_col:
        out["sleep_quality"] = to_numeric_clean(df_raw[sleep_quality_col])
    else:
        out["sleep_quality"] = np.nan

    if nocturnal_wakeups_col:
        out["nocturnal_wakeups"] = df_raw[nocturnal_wakeups_col].apply(map_likert_text)
    else:
        out["nocturnal_wakeups"] = np.nan

    if restorative_sleep_col:
        out["restorative_sleep"] = df_raw[restorative_sleep_col].apply(map_likert_text)
    else:
        out["restorative_sleep"] = np.nan

    # Symptom/pain columns across Jour 1, Jour 2, optional days.
    symptom_cols = extract_symptom_scores(df_raw)
    out = pd.concat([out, symptom_cols], axis=1)

    # Basic cleaning.
    out = out.dropna(subset=["PID", "week"])
    out["PID"] = out["PID"].astype(int)
    out["week"] = out["week"].astype(float)

    # Merge program info and signup metadata.
    out = out.merge(df_prog, on="PID", how="left")
    signup_cols = [
        "PID",
        "age",
        "postpartum_weeks",
        "children_count",
        "delivery_mode",
        "tear_grade",
        "status",
        "is_study_participant",
    ]
    out = out.merge(df_signup[signup_cols], on="PID", how="left")

    # Normalization.
    out["progress"] = out["week"] / out["program_length"]
    out["progress_bin"] = pd.cut(
        out["progress"],
        bins=PROGRESS_BINS,
        labels=PROGRESS_LABELS,
        include_lowest=True,
    )

    out["has_program_info"] = out["program_length"].notna()
    out["in_program"] = out["progress"].between(0, 1, inclusive="both")

    # Data-quality / analysis flags.
    out["flag_no_program_info"] = ~out["has_program_info"]
    out["flag_week_missing"] = out["week"].isna()
    out["flag_outside_program"] = ~out["in_program"]

    out["include_in_forms_analysis"] = (
        out["has_program_info"]
        & out["week"].notna()
        & out["in_program"]
    )

    # Duplicate responses for same PID/week.
    duplicate_counts = (
        out.groupby(["PID", "week"])["PID"]
        .transform("count")
    )
    out["n_responses_same_pid_week"] = duplicate_counts
    out["flag_duplicate_pid_week"] = out["n_responses_same_pid_week"] > 1

    # Derived variables.
    out["adherence_ratio"] = (out["sessions_done"] / 5).clip(lower=0, upper=1)

    fatigue_cols = list(FATIGUE_QUESTION_SPECS.keys())
    existing_fatigue_cols = [c for c in fatigue_cols if c in out.columns]
    out["fatigue_mean_score"] = out[existing_fatigue_cols].mean(axis=1, skipna=True)

    symptom_score_cols = [
        c for c in out.columns
        if c.startswith("symptom_") and c.endswith("_mean")
    ]
    out["symptom_mean_score"] = out[symptom_score_cols].mean(axis=1, skipna=True)

    # Put core columns first.
    core_cols = [
        "PID",
        "group",
        "program_length",
        "week",
        "progress",
        "progress_bin",
        "timestamp",
        "include_in_forms_analysis",
        "in_program",
        "has_program_info",
        "flag_no_program_info",
        "flag_outside_program",
        "flag_duplicate_pid_week",
        "n_responses_same_pid_week",
        "age",
        "postpartum_weeks",
        "children_count",
        "delivery_mode",
        "tear_grade",
        "status",
        "sessions_done",
        "adherence_ratio",
        "clarity",
        "intensity_respected",
        "satisfaction",
        "confidence",
        "recommendation",
        "fatigue_mean_score",
        "symptom_mean_score",
        "sleep_hours",
        "sleep_quality",
        "nocturnal_wakeups",
        "restorative_sleep",
        "activity_limitation_reason",
        "free_comment",
    ]

    remaining_cols = [c for c in out.columns if c not in core_cols]
    out = out[core_cols + remaining_cols]

    return out.sort_values(["group", "PID", "week", "timestamp"], na_position="last")


def extract_symptom_scores(df_raw):
    """
    Extracts day-level symptom/pain columns from Google Forms.

    The source columns look like:
    Jour 1 [Douleur au genou]
    Jour 2 [Douleur lombaire]
    Jour 3 (Optionnel) [Douleur musculaire]
    etc.

    Output gives:
    - symptom_day1_mean
    - symptom_day2_mean
    - symptom_day3_mean
    - symptom_day4_mean
    - symptom_all_days_mean
    - symptom_any_nonzero
    """

    normalized_cols = {col: normalize_column_name(col) for col in df_raw.columns}

    day_patterns = {
        "day1": ["jour 1"],
        "day2": ["jour 2"],
        "day3": ["jour 3"],
        "day4": ["jour 4"],
    }

    symptom_keywords = [
        "incontinence",
        "envie pressante",
        "douleur",
    ]

    out = pd.DataFrame(index=df_raw.index)
    day_score_cols = []

    for day_name, fragments in day_patterns.items():
        cols = []

        for original_col, norm_col in normalized_cols.items():
            is_day = all(fragment in norm_col for fragment in fragments)
            is_symptom = any(keyword in norm_col for keyword in symptom_keywords)

            if is_day and is_symptom:
                cols.append(original_col)

        clean_day_cols = []

        for col in cols:
            clean_col = f"raw_{day_name}_{slugify_column(col)}"
            out[clean_col] = to_numeric_clean(df_raw[col])
            clean_day_cols.append(clean_col)

        mean_col = f"symptom_{day_name}_mean"
        max_col = f"symptom_{day_name}_max"

        if clean_day_cols:
            out[mean_col] = out[clean_day_cols].mean(axis=1, skipna=True)
            out[max_col] = out[clean_day_cols].max(axis=1, skipna=True)
        else:
            out[mean_col] = np.nan
            out[max_col] = np.nan

        day_score_cols.append(mean_col)

    out["symptom_all_days_mean"] = out[day_score_cols].mean(axis=1, skipna=True)
    out["symptom_any_nonzero"] = out[day_score_cols].fillna(0).gt(0).any(axis=1)

    return out


def slugify_column(col):
    col = normalize_column_name(col)
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = col.strip("_")
    return col[:80]


# ============================================================
# AGGREGATION
# ============================================================

def build_weekly_pid_table(df_weekly):
    """
    One row per PID/week.
    If a participant submitted multiple responses for the same week,
    numeric variables are averaged.
    """

    df = df_weekly[df_weekly["include_in_forms_analysis"]].copy()

    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    # Do not average identifiers as outcomes.
    for c in ["PID", "program_length"]:
        if c in numeric_cols:
            numeric_cols.remove(c)

    agg_dict = {c: "mean" for c in numeric_cols}

    # Stable categorical columns.
    cat_cols = [
        "group",
        "progress_bin",
        "delivery_mode",
        "tear_grade",
        "status",
    ]

    for c in cat_cols:
        if c in df.columns:
            agg_dict[c] = "first"

    # Keep timestamps and comments in basic form.
    agg_dict["timestamp"] = "min"
    agg_dict["activity_limitation_reason"] = lambda x: " | ".join(
        sorted(set(v for v in x.astype(str) if v and v != "nan"))
    )
    agg_dict["free_comment"] = lambda x: " | ".join(
        sorted(set(v for v in x.astype(str) if v and v != "nan"))
    )

    weekly_pid = (
        df.groupby(["PID", "program_length", "week"], as_index=False)
        .agg(agg_dict)
    )

    weekly_pid["progress"] = weekly_pid["week"] / weekly_pid["program_length"]

    weekly_pid["progress_bin"] = pd.cut(
        weekly_pid["progress"],
        bins=PROGRESS_BINS,
        labels=PROGRESS_LABELS,
        include_lowest=True,
    )

    return weekly_pid.sort_values(["group", "PID", "week"])


def build_participant_summary(df_weekly_pid, df_signup, df_prog):
    df = df_weekly_pid.copy()

    question_cols = list(WEEKLY_QUESTION_SPECS.keys())
    fatigue_cols = list(FATIGUE_QUESTION_SPECS.keys())

    agg = {
        "group": "first",
        "program_length": "first",
        "week": ["min", "max", "nunique"],
        "progress": "max",
        "adherence_ratio": "mean",
        "sessions_done": "mean",
        "fatigue_mean_score": "mean",
        "symptom_mean_score": "mean",
        "sleep_hours": "mean",
        "sleep_quality": "mean",
    }

    for col in question_cols:
        if col in df.columns:
            agg[col] = "mean"

    for col in fatigue_cols:
        if col in df.columns:
            agg[col] = "mean"

    participant = df.groupby("PID").agg(agg)

    participant.columns = [
        "_".join([str(x) for x in col if x])
        for col in participant.columns.to_flat_index()
    ]

    participant = participant.reset_index()

    participant = participant.rename(columns={
        "group_first": "group",
        "program_length_first": "program_length",
        "week_min": "first_reported_week",
        "week_max": "last_reported_week",
        "week_nunique": "n_weeks_reported",
        "progress_max": "max_progress_reported",
        "adherence_ratio_mean": "mean_adherence_ratio",
        "sessions_done_mean": "mean_sessions_done",
        "fatigue_mean_score_mean": "mean_fatigue_score",
        "symptom_mean_score_mean": "mean_symptom_score",
        "sleep_hours_mean": "mean_sleep_hours",
        "sleep_quality_mean": "mean_sleep_quality",
    })

    participant["completed_by_forms"] = (
        participant["last_reported_week"] >= participant["program_length"]
    )

    participant["forms_completion_rate"] = (
        participant["last_reported_week"] / participant["program_length"]
    )

    participant["forms_coverage"] = (
        participant["n_weeks_reported"] / participant["program_length"]
    )

    signup_keep = [
        "PID",
        "age",
        "postpartum_weeks",
        "children_count",
        "delivery_mode",
        "tear_grade",
        "status",
    ]

    participant = participant.merge(
        df_signup[signup_keep],
        on="PID",
        how="left",
    )

    # Add participants who have program info but no weekly responses.
    all_participants = df_prog[["PID", "group", "program_length"]].copy()

    participant = all_participants.merge(
        participant,
        on=["PID", "group", "program_length"],
        how="left",
    )

    participant["has_weekly_forms"] = participant["n_weeks_reported"].notna()
    participant["n_weeks_reported"] = participant["n_weeks_reported"].fillna(0)

    return participant.sort_values(["group", "PID"])


# ============================================================
# PLOTS
# ============================================================

def plot_active_participants(df_weekly_pid):
    counts_week = (
        df_weekly_pid.groupby(["week", "group"])["PID"]
        .nunique()
        .reset_index(name="n_participants")
    )

    plt.figure(figsize=(10, 6))

    for group in ["A", "B"]:
        subset = counts_week[counts_week["group"] == group]

        plt.plot(
            subset["week"],
            subset["n_participants"],
            marker="o",
            label=f"Groupe {group}",
            color=GROUP_COLORS.get(group),
        )

    plt.title("Nombre de participantes actives par semaine")
    plt.xlabel("Semaine du programme")
    plt.ylabel("Nombre de participantes")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "forms_active_participants_by_week.png"), dpi=300)
    plt.close()

    counts_progress = (
        df_weekly_pid.groupby(["progress_bin", "group"], observed=True)["PID"]
        .nunique()
        .reset_index(name="n_participants")
    )

    plt.figure(figsize=(10, 6))

    for group in ["A", "B"]:
        subset = counts_progress[counts_progress["group"] == group]

        plt.plot(
            subset["progress_bin"].astype(str),
            subset["n_participants"],
            marker="o",
            label=f"Groupe {group}",
            color=GROUP_COLORS.get(group),
        )

    plt.title("Nombre de participantes actives par progression normalisée")
    plt.xlabel("Progression dans le programme")
    plt.ylabel("Nombre de participantes")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "forms_active_participants_by_progress.png"), dpi=300)
    plt.close()


def plot_group_question_curves(df_weekly_pid, statistic="mean"):
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")

    for col, spec in WEEKLY_QUESTION_SPECS.items():
        if col not in df_weekly_pid.columns:
            continue

        plt.figure(figsize=(10, 6))

        for group in ["A", "B"]:
            subset = df_weekly_pid[df_weekly_pid["group"] == group]

            if statistic == "mean":
                grouped = subset.groupby("progress_bin", observed=True)[col].mean()
                stat_label = "moyenne"
            else:
                grouped = subset.groupby("progress_bin", observed=True)[col].median()
                stat_label = "médiane"

            plt.plot(
                grouped.index.astype(str),
                grouped.values,
                marker="o",
                linewidth=2.5,
                label=f"Groupe {group}",
                color=GROUP_COLORS.get(group),
            )

        plt.title(f"{spec['label']} ({stat_label})")
        plt.xlabel("Progression dans le programme")
        plt.ylabel("Score")
        if spec.get("ylim"):
            plt.ylim(*spec["ylim"])
        plt.grid()
        plt.legend()
        plt.tight_layout()

        filename = f"forms_{col}_{statistic}_by_group_normalized.png"
        plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
        plt.close()


def plot_individual_question_trajectories(df_weekly_pid):
    for col, spec in WEEKLY_QUESTION_SPECS.items():
        if col not in df_weekly_pid.columns:
            continue

        plt.figure(figsize=(10, 6))

        for pid, group_df in df_weekly_pid.groupby("PID"):
            group_df = group_df.sort_values("progress")
            group = group_df["group"].iloc[0]

            plt.plot(
                group_df["progress"],
                group_df[col],
                alpha=0.25,
                color=GROUP_COLORS.get(group, "gray"),
            )

        # Overlay group medians by bin.
        x_map = {
            "0-25%": 0.125,
            "25-50%": 0.375,
            "50-75%": 0.625,
            "75-100%": 0.875,
        }

        for group in ["A", "B"]:
            subset = df_weekly_pid[df_weekly_pid["group"] == group]

            med = subset.groupby("progress_bin", observed=True)[col].median()
            x = [x_map[str(k)] for k in med.index]

            plt.plot(
                x,
                med.values,
                marker="o",
                linewidth=3,
                label=f"Groupe {group} (médiane)",
                color=GROUP_COLORS.get(group),
            )

        plt.title(f"Trajectoires individuelles - {spec['label']}")
        plt.xlabel("Progression normalisée dans le programme")
        plt.ylabel("Score")
        if spec.get("ylim"):
            plt.ylim(*spec["ylim"])
        plt.xlim(0, 1)
        plt.grid()
        plt.legend()
        plt.tight_layout()

        filename = f"forms_{col}_individual_trajectories.png"
        plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
        plt.close()


def plot_fatigue_and_symptom_curves(df_weekly_pid):
    extra_specs = {
        "fatigue_mean_score": {
            "label": "Score moyen de fatigue / état général",
            "ylim": (0, 4),
        },
        "symptom_mean_score": {
            "label": "Score moyen des symptômes / douleurs",
            "ylim": (0, 5),
        },
        "sleep_hours": {
            "label": "Durée moyenne de sommeil",
            "ylim": None,
        },
        "sleep_quality": {
            "label": "Qualité du sommeil",
            "ylim": (0, 5),
        },
    }

    for col, spec in extra_specs.items():
        if col not in df_weekly_pid.columns:
            continue

        plt.figure(figsize=(10, 6))

        for group in ["A", "B"]:
            subset = df_weekly_pid[df_weekly_pid["group"] == group]
            grouped = subset.groupby("progress_bin", observed=True)[col].mean()

            plt.plot(
                grouped.index.astype(str),
                grouped.values,
                marker="o",
                linewidth=2.5,
                label=f"Groupe {group}",
                color=GROUP_COLORS.get(group),
            )

        plt.title(spec["label"])
        plt.xlabel("Progression dans le programme")
        plt.ylabel("Score / valeur")
        if spec["ylim"]:
            plt.ylim(*spec["ylim"])
        plt.grid()
        plt.legend()
        plt.tight_layout()

        filename = f"forms_{col}_mean_by_group_normalized.png"
        plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
        plt.close()


# ============================================================
# SANITY CHECKS / REPORTING
# ============================================================

def print_sanity_checks(df_prog, df_signup, df_weekly, df_weekly_pid, participant_summary):
    print("\n--- Program participants ---")
    print(df_prog.groupby("group")["PID"].nunique())

    print("\n--- Signup participants with program info ---")
    print(df_signup[df_signup["has_program_info"]].groupby("group")["PID"].nunique())

    print("\n--- Weekly raw responses ---")
    print(f"Total weekly responses: {len(df_weekly)}")

    print("\n--- Weekly responses included in forms analysis ---")
    print(df_weekly.groupby(["group", "include_in_forms_analysis"])["PID"].count())

    print("\n--- Participants with included weekly forms ---")
    print(df_weekly_pid.groupby("group")["PID"].nunique())

    print("\n--- Participants with no included weekly forms ---")
    no_forms = participant_summary[~participant_summary["has_weekly_forms"]]
    if len(no_forms) == 0:
        print("None")
    else:
        print(no_forms[["PID", "group", "program_length"]].to_string(index=False))

    print("\n--- Duplicate PID/week responses ---")
    dups = df_weekly[df_weekly["flag_duplicate_pid_week"]]
    if len(dups) == 0:
        print("None")
    else:
        print(
            dups[[
                "PID",
                "group",
                "week",
                "timestamp",
                "n_responses_same_pid_week",
            ]]
            .sort_values(["group", "PID", "week"])
            .to_string(index=False)
        )

    print("\n--- Group mismatch signup vs program_length ---")
    mismatches = df_signup[df_signup["group_mismatch_signup_vs_program"]]
    if len(mismatches) == 0:
        print("None")
    else:
        print(
            mismatches[[
                "PID",
                "signup_group",
                "group",
                "program_length",
            ]].to_string(index=False)
        )

    print("\n--- Participant summary preview ---")
    print(participant_summary.head(20).to_string(index=False))


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Clean and aggregate Google Forms signup + weekly reports."
    )

    parser.add_argument(
        "--program",
        default="program_length.csv",
        help="Path to program_length.csv",
    )

    parser.add_argument(
        "--signup",
        default="start.csv",
        help="Path to signup CSV",
    )

    parser.add_argument(
        "--weekly",
        default="data2.csv",
        help="Path to weekly Google Forms CSV",
    )

    args = parser.parse_args()

    ensure_output_dirs()

    print("\nLoading program_length...")
    df_prog = load_program_length(args.program)

    print("Loading signup form...")
    df_signup = load_signup(args.signup, df_prog)

    print("Loading weekly form...")
    df_weekly = load_weekly(args.weekly, df_prog, df_signup)

    print("Building participant-week table...")
    df_weekly_pid = build_weekly_pid_table(df_weekly)

    print("Building participant summary...")
    participant_summary = build_participant_summary(df_weekly_pid, df_signup, df_prog)

    save_csv(df_prog, "program_length_cleaned.csv")
    save_csv(df_signup, "forms_signup_cleaned.csv")
    save_csv(df_weekly, "forms_weekly_cleaned.csv")
    save_csv(df_weekly_pid, "forms_weekly_pid_cleaned.csv")
    save_csv(participant_summary, "forms_participant_summary.csv")

    print("\nGenerating plots...")
    plot_active_participants(df_weekly_pid)
    plot_group_question_curves(df_weekly_pid, statistic="mean")
    plot_group_question_curves(df_weekly_pid, statistic="median")
    plot_individual_question_trajectories(df_weekly_pid)
    plot_fatigue_and_symptom_curves(df_weekly_pid)

    print_sanity_checks(
        df_prog=df_prog,
        df_signup=df_signup,
        df_weekly=df_weekly,
        df_weekly_pid=df_weekly_pid,
        participant_summary=participant_summary,
    )

    print("\nDone.")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print(f"Plots saved in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()