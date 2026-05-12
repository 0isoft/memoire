import os
import pandas as pd
from datetime import datetime
from fitparse import FitFile
import xml.etree.ElementTree as ET
from datetime import timezone


folder = "/home/octavian/projects/memoire/nolio-2"
print("CURRENT DIR:", os.getcwd())
sessions = []

def compute_max_continuous_run(trackpoints):
    MAX_DT = 25
    MIN_DT = 0.5

    RUN_MIN_SPEED = 1.5
    RUN_MAX_SPEED = 6.0

    MIN_DISTANCE_STEP = 1.0
    TOLERANCE = 8

    current_streak = 0
    max_streak = 0
    slow_time = 0

    for i in range(1, len(trackpoints)):
        t1, d1 = trackpoints[i-1]
        t2, d2 = trackpoints[i]

        dt = (t2 - t1).total_seconds()
        dd = d2 - d1

        if dt <= MIN_DT:
            continue

        if dt > MAX_DT:
            max_streak = max(max_streak, current_streak)
            current_streak = 0
            slow_time = 0
            continue

        if dd < MIN_DISTANCE_STEP:
            speed = 0
        else:
            speed = dd / dt

        if RUN_MIN_SPEED <= speed <= RUN_MAX_SPEED:
            current_streak += dt
            slow_time = 0
        else:
            slow_time += dt

            if slow_time <= TOLERANCE:
                current_streak += dt
            else:
                max_streak = max(max_streak, current_streak)
                current_streak = 0
                slow_time = 0

    max_streak = max(max_streak, current_streak)

    total_time = sum(
        max(0, (trackpoints[i][0] - trackpoints[i-1][0]).total_seconds())
        for i in range(1, len(trackpoints))
    )

    return total_time / 60, max_streak / 60

# --- TCX PARSER ---
def strip_ns(tag):
    return tag.split('}')[-1]

def parse_tcx(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    trackpoints = []
    start_time = None

    for elem in root.iter():
        tag = strip_ns(elem.tag)

        if tag == "Id" and start_time is None:
            start_time = datetime.fromisoformat(elem.text.replace("Z", "+00:00"))

        if tag == "Trackpoint":
            time = None
            dist = None

            for child in elem:
                child_tag = strip_ns(child.tag)

                if child_tag == "Time":
                    time = datetime.fromisoformat(child.text.replace("Z", "+00:00"))
                elif child_tag == "DistanceMeters":
                    dist = float(child.text)

            if time and dist is not None:
                trackpoints.append((time, dist))

    
    total_time_min, max_run_min = compute_max_continuous_run(trackpoints)

    total_distance = trackpoints[-1][1] if trackpoints else 0

    return start_time, total_time_min, max_run_min, total_distance/1000

def parse_fit(file_path):
    fitfile = FitFile(file_path)

    trackpoints = []
    start_time = None

    for msg in fitfile.get_messages("record"):
        time = None
        dist = None

        for data in msg:
            if data.name == "timestamp":
                time = data.value.replace(tzinfo=timezone.utc)
                if start_time is None:
                    start_time = time
            elif data.name == "distance":
                dist = data.value

        if time and dist is not None:
            trackpoints.append((time, dist))

    
    total_time_min, max_run_min = compute_max_continuous_run(trackpoints)
    total_distance = trackpoints[-1][1] if trackpoints else 0


    return start_time, total_time_min, max_run_min, total_distance/1000

def process_folder(folder_path, folder_id):
    sessions = []

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        if not os.path.isfile(path):
            continue

        try:
            if file.lower().endswith(".fit"):
                start_time, total_time_min, max_run_min, dist_km = parse_fit(path)
            elif file.lower().endswith(".tcx"):
                start_time, total_time_min, max_run_min, dist_km = parse_tcx(path)
            else:
                continue

            sessions.append({
                "id": int(folder_id),
                "session": file,
                "date": start_time.date() if start_time else None,
                "start_time": start_time,
                "total_duration_min": round(total_time_min, 2),
                "max_continuous_run_min": round(max_run_min, 2),
                "distance_km": round(dist_km, 2)
            })

        except Exception as e:
            print(f"Error with {file}: {e}")

    return sessions


all_sessions = []

for entry in os.scandir(folder):
    if entry.is_dir():
        folder_id = entry.name
        folder_path = entry.path

        print(f"Processing folder: {folder_id}")

        folder_sessions = process_folder(folder_path, folder_id)
        all_sessions.extend(folder_sessions)


df = pd.DataFrame(all_sessions)

df = df.sort_values(by="start_time")

print(df)

df.to_csv("all_sessions.csv", index=False)

