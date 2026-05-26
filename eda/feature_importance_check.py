"""
Check correlation of Flow Bytes/s and Flow Packets/s with the Label.
Uses point-biserial correlation (numeric feature vs binary label).
"""

import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")

FEATURES_TO_CHECK = ["Flow Bytes/s", "Flow Packets/s"]


def load_raw(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].str.strip()
    return df


def main() -> None:
    print("Loading dataset...")
    df = load_raw(DATA_DIR)

    # Binary label: 1 = attack, 0 = benign
    df["is_attack"] = (df["Label"] != "BENIGN").astype(int)

    # Drop rows with inf or NaN in the features we check
    df = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df.dropna(subset=FEATURES_TO_CHECK)

    print(f"\nRows after dropping NaN/Inf: {len(df_clean):,} "
          f"(dropped {len(df) - len(df_clean):,})\n")

    print(f"{'Feature':<25} {'Correlation':>12} {'p-value':>12} {'Importance'}")
    print("-" * 65)

    for feature in FEATURES_TO_CHECK:
        corr, pval = pointbiserialr(df_clean[feature], df_clean["is_attack"])
        importance = "HIGH" if abs(corr) > 0.3 else "MEDIUM" if abs(corr) > 0.1 else "LOW"
        print(f"{feature:<25} {corr:>12.4f} {pval:>12.2e} {importance}")

    print("\nInterpretation:")
    print("  |correlation| > 0.3  -> HIGH importance")
    print("  |correlation| > 0.1  -> MEDIUM importance")
    print("  |correlation| < 0.1  -> LOW importance (safe to drop or ignore missing rows)")


if __name__ == "__main__":
    main()
