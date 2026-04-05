import json
from pathlib import Path

import pandas as pd

SUBSET_CSV = Path("verification_subset_300.csv")
DATA_DIR = Path("verification_data")
OUTPUT_JSON = Path("labelstudio_tasks.json")

subset = pd.read_csv(SUBSET_CSV).sort_values("event_idx").reset_index(drop=True)

tasks = []

for _, row in subset.iterrows():
    event_idx = int(row["event_idx"])
    csv_file = DATA_DIR / f"event_{event_idx:05d}.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"Missing waveform file: {csv_file}")

    predicted_t0 = None
    if "fault_start_s" in subset.columns and pd.notna(row["fault_start_s"]):
        predicted_t0 = float(row["fault_start_s"])

    group = str(row["verification_group"]) if pd.notna(row["verification_group"]) else ""

    tasks.append(
        {
            "data": {
                "csv": f"/data/local-files/?d=fault_detection_framework/verification_data/event_{event_idx:05d}.csv",
                "event_idx": event_idx,
                "predicted_t0": predicted_t0,
                "group": group,
            }
        }
    )

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(tasks, f, indent=2, ensure_ascii=False, allow_nan=False)

print(f"Created {OUTPUT_JSON} with {len(tasks)} tasks")