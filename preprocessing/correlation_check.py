"""
Check correlation between selected features.
"""

import glob
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")

FEATURES = [
    "Flow IAT Max",
    "Fwd IAT Max",
    "Idle Min",
    "Idle Mean",
    "Idle Max",
]


def load_raw(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()
    return df


def main() -> None:
    print("Loading dataset...")
    df = load_raw(DATA_DIR)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)

    corr = df[FEATURES].corr().round(3)

    print("\nCorrelation Matrix:")
    print(corr.to_string())

    print("\nHigh correlations (|r| > 0.9):")
    found = False
    for i in range(len(FEATURES)):
        for j in range(i + 1, len(FEATURES)):
            r = corr.iloc[i, j]
            if abs(r) > 0.9:
                print(f"  {FEATURES[i]}  <-->  {FEATURES[j]}  :  r = {r}")
                found = True
    if not found:
        print("  None found.")


if __name__ == "__main__":
    main()
