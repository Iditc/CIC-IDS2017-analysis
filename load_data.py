"""
Step 1 — Data Loading
Loads all CIC-IDS2017 CSV files, cleans them, and returns a unified DataFrame.
"""

import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")

# Map each source file to the attack day it represents
DAY_LABELS = {
    "Monday-WorkingHours":                              "Monday (Benign)",
    "Tuesday-WorkingHours":                             "Tuesday (Brute Force)",
    "Wednesday-workingHours":                           "Wednesday (DoS/DDoS)",
    "Thursday-WorkingHours-Morning-WebAttacks":         "Thursday Morning (Web Attacks)",
    "Thursday-WorkingHours-Afternoon-Infilteration":    "Thursday Afternoon (Infiltration)",
    "Friday-WorkingHours-Morning":                      "Friday Morning (PortScan)",
    "Friday-WorkingHours-Afternoon-DDos":               "Friday Afternoon (DDoS)",
    "Friday-WorkingHours-Afternoon-PortScan":           "Friday Afternoon (PortScan)",
}


def _source_day(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0].replace(".pcap_ISCX", "")
    for key, label in DAY_LABELS.items():
        if key.lower() in base.lower():
            return label
    return base


def load_dataset(data_dir: str = DATA_DIR) -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}\n"
            "Run download_data.py first."
        )

    frames = []
    for path in csv_files:
        print(f"Loading {os.path.basename(path)}...")
        df = pd.read_csv(path, low_memory=False)
        df["source_file"] = _source_day(path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Normalize column names: strip whitespace
    combined.columns = combined.columns.str.strip()

    # Replace inf/-inf with NaN, then drop those rows
    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    rows_before = len(combined)
    combined.dropna(inplace=True)
    dropped = rows_before - len(combined)
    if dropped:
        print(f"Dropped {dropped:,} rows with NaN/Inf values ({dropped/rows_before:.1%})")

    # Normalize label column
    combined["Label"] = combined["Label"].str.strip()

    print(f"\nDataset loaded: {len(combined):,} rows × {len(combined.columns)} columns")
    return combined


if __name__ == "__main__":
    df = load_dataset()
    print(df["Label"].value_counts())
