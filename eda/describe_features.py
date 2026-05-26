"""
Basic statistical description of all kept features.
Prints: mean, std, min, max, skewness, and % of outliers (IQR method).
"""

import glob
import os
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
FEATURES_TO_KEEP_PATH = os.path.join(OUTPUT_DIR, "features_to_keep.csv")


def load_raw(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()
    return df


def outlier_pct(series: pd.Series) -> float:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).mean() * 100


def main() -> None:
    to_keep = pd.read_csv(FEATURES_TO_KEEP_PATH)["feature"].tolist()

    print("Loading dataset...")
    df = load_raw(DATA_DIR)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=to_keep)

    print(f"\nDataset: {len(df):,} rows  |  {len(to_keep)} features\n")

    fmt = "{:<40} {:>12} {:>12} {:>12} {:>12} {:>10} {:>10}"
    print(fmt.format("Feature", "Mean", "Std", "Min", "Max", "Skewness", "Outliers%"))
    print("-" * 110)

    rows = []
    for feat in to_keep:
        s = df[feat]
        rows.append({
            "feature":   feat,
            "mean":      s.mean(),
            "std":       s.std(),
            "min":       s.min(),
            "max":       s.max(),
            "skewness":  s.skew(),
            "outliers%": outlier_pct(s),
        })

    stats = pd.DataFrame(rows).sort_values("skewness", ascending=False, key=abs)

    for _, r in stats.iterrows():
        print(fmt.format(
            r["feature"][:40],
            f"{r['mean']:.2f}",
            f"{r['std']:.2f}",
            f"{r['min']:.2f}",
            f"{r['max']:.2f}",
            f"{r['skewness']:.2f}",
            f"{r['outliers%']:.1f}%",
        ))

    print("\nTop 10 most skewed features:")
    print(stats.head(10)[["feature", "skewness", "outliers%"]].to_string(index=False))


if __name__ == "__main__":
    main()
