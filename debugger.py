import pandas as pd

df = pd.read_csv("forms_output/forms_weekly_pid_cleaned.csv")

questions = [
    "sessions_done",
    "clarity",
    "intensity_respected",
    "satisfaction",
    "confidence",
    "recommendation",
]

print("Participants in weekly PID table:", df["PID"].nunique())
print(df.groupby("group")["PID"].nunique())

print("\nParticipants with at least one non-NaN value per question:")
for q in questions:
    usable = df[df[q].notna()]
    print(q, usable["PID"].nunique(), usable.groupby("group")["PID"].nunique().to_dict())