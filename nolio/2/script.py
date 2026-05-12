import os
import pandas as pd
from datetime import datetime
from fitparse import FitFile
import xml.etree.ElementTree as ET

folder = "/home/octavian/projects/memoire/nolio/2"
print("CURRENT DIR:", os.getcwd())
sessions = []

# --- TCX PARSER ---
def strip_ns(tag):
    return tag.split('}')[-1]

def parse_tcx(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    trackpoints = []

    for elem in root.iter():
        if strip_ns(elem.tag) == "Trackpoint":
            time = None
            dist = None

            for child in elem:
                tag = strip_ns(child.tag)

                if tag == "Time":
                    time = datetime.fromisoformat(child.text.replace("Z", "+00:00"))
                elif tag == "DistanceMeters":
                    dist = float(child.text)

            if time and dist is not None:
                trackpoints.append((time, dist))

    moving_time = 0
    total_time = 0

    for i in range(1, len(trackpoints)):
        t1, d1 = trackpoints[i-1]
        t2, d2 = trackpoints[i]

        dt = (t2 - t1).total_seconds()
        dd = d2 - d1

        if dt <= 0:
            continue

        speed = dd / dt  # m/s

        total_time += dt

        if speed > 0.5:  # threshold
            moving_time += dt

    total_distance = trackpoints[-1][1] if trackpoints else 0

    return total_time/60, moving_time/60, total_distance/1000

# --- FIT PARSER ---
def parse_fit(file_path):
    fitfile = FitFile(file_path)

    trackpoints = []

    for msg in fitfile.get_messages("record"):
        time = None
        dist = None

        for data in msg:
            if data.name == "timestamp":
                time = data.value
            elif data.name == "distance":
                dist = data.value

        if time and dist is not None:
            trackpoints.append((time, dist))

    moving_time = 0
    total_time = 0

    for i in range(1, len(trackpoints)):
        t1, d1 = trackpoints[i-1]
        t2, d2 = trackpoints[i]

        dt = (t2 - t1).total_seconds()
        dd = d2 - d1

        if dt <= 0:
            continue

        speed = dd / dt  # m/s

        total_time += dt

        if speed > 0.5:  # same threshold
            moving_time += dt

    total_distance = trackpoints[-1][1] if trackpoints else 0

    return total_time/60, moving_time/60, total_distance/1000


# --- MAIN LOOP ---
for file in os.listdir(folder):
    path = os.path.join(folder, file)

    try:
        if file.lower().endswith(".tcx"):
            elapsed_min, moving_min, dist_km = parse_tcx(path)


        elif file.lower().endswith(".fit"):
            elapsed_min, moving_min, dist_km = parse_fit(path)

        else:
            continue

        sessions.append({
        "session": file,
        "elapsed_min": round(elapsed_min, 2),
        "moving_min": round(moving_min, 2),
        "distance_km": round(dist_km, 2),
        "moving_ratio": round(moving_min / elapsed_min, 2) if elapsed_min else 0
    })

    except Exception as e:
        print(f"Error with {file}: {e}")

# --- DATAFRAME ---
df = pd.DataFrame(sessions)

print(df)

# Optional: save
df.to_csv("running_sessions.csv", index=False)