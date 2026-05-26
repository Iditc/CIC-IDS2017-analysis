"""
Data Cleaning — Step 1: Remove problematic rows.

  Step 1 — Duplicate rows:
      Identify and remove exact duplicate rows (all feature values identical).
      Keep the first occurrence of each duplicate group.

  Step 2 — NaN and Infinity analysis:
      Count NaN and Infinity values per class before removal.
      Save a bar chart showing the distribution across classes.
      Then drop all affected rows.

Output:
  - Prints counts and percentages to terminal.
  - Saves output/nan_inf_per_class.png
  - Saves cleaned dataset to data/processed/clean.parquet
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
FEATURES_TO_KEEP_PATH = os.path.join(OUTPUT_DIR, "features_to_keep.csv")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_raw(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].str.strip()
    return df


def apply_feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only features from features_to_keep.csv plus Label."""
    if not os.path.exists(FEATURES_TO_KEEP_PATH):
        print("Warning: features_to_keep.csv not found — using all features.")
        return df
    features = pd.read_csv(FEATURES_TO_KEEP_PATH)["feature"].tolist()
    cols = [c for c in features + ["Label"] if c in df.columns]
    return df[cols]


# ── Step 1: Duplicates ────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows, keeping the first occurrence."""
    total = len(df)
    df_clean = df.drop_duplicates(keep="first")
    removed = total - len(df_clean)

    print(f"\n  Total rows        : {total:,}")
    print(f"  Duplicate rows    : {removed:,} ({removed / total * 100:.2f}%)")
    print(f"  Rows after dedup  : {len(df_clean):,}")

    # Duplicates per class
    dupes = df[df.duplicated(keep="first")]
    if removed > 0:
        print(f"\n  Duplicates per class:")
        for label, count in dupes["Label"].value_counts().items():
            pct = count / df["Label"].value_counts()[label] * 100
            print(f"    {label:<45} {count:>7,}  ({pct:.2f}%)")

    return df_clean


# ── Step 2: NaN and Infinity analysis ────────────────────────────────────────

def analyze_and_plot_nan_inf(df: pd.DataFrame) -> pd.DataFrame:
    """Count NaN and Inf rows per class, plot a diagram, then drop them."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # A row is problematic if it contains at least one NaN or Inf value
    has_inf = np.isinf(df[numeric_cols]).any(axis=1)
    has_nan = df[numeric_cols].isnull().any(axis=1)
    is_problematic = has_inf | has_nan

    total_problematic = is_problematic.sum()
    print(f"\n  Total rows with NaN or Inf : {total_problematic:,} "
          f"({total_problematic / len(df) * 100:.3f}%)")
    print(f"    - Rows with Inf only      : {(has_inf & ~has_nan).sum():,}")
    print(f"    - Rows with NaN only      : {(has_nan & ~has_inf).sum():,}")
    print(f"    - Rows with both          : {(has_inf & has_nan).sum():,}")

    # Count per class
    class_counts = df["Label"].value_counts()
    problematic_per_class = df[is_problematic]["Label"].value_counts()

    stats = pd.DataFrame({
        "total":       class_counts,
        "problematic": problematic_per_class,
    }).fillna(0)
    stats["problematic"] = stats["problematic"].astype(int)
    stats["pct"] = (stats["problematic"] / stats["total"] * 100).round(3)
    stats = stats.sort_values("pct", ascending=False)

    print(f"\n  NaN/Inf rows per class:")
    print(f"  {'Class':<45} {'Problematic':>12} {'% of class':>12}")
    print("  " + "-" * 72)
    for label, row in stats.iterrows():
        print(f"  {label:<45} {int(row['problematic']):>12,} {row['pct']:>11.3f}%")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: absolute counts
    ax1 = axes[0]
    colors = ["#d9534f" if v > 0 else "#5bc0de" for v in stats["problematic"]]
    bars = ax1.barh(stats.index, stats["problematic"], color=colors, edgecolor="white")
    ax1.set_title("NaN / Inf Rows per Class — Absolute Count")
    ax1.set_xlabel("Number of problematic rows")
    for bar, val in zip(bars, stats["problematic"]):
        if val > 0:
            ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{val:,}", va="center", fontsize=8)

    # Right: percentage of class
    ax2 = axes[1]
    colors2 = ["#d9534f" if v > 0 else "#5bc0de" for v in stats["pct"]]
    bars2 = ax2.barh(stats.index, stats["pct"], color=colors2, edgecolor="white")
    ax2.set_title("NaN / Inf Rows per Class — % of Class")
    ax2.set_xlabel("% of class total")
    for bar, val in zip(bars2, stats["pct"]):
        if val > 0:
            ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}%", va="center", fontsize=8)

    plt.suptitle("Problematic Rows (NaN / Infinity) Before Removal", fontsize=12, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "nan_inf_per_class.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out_path}")

    # Drop problematic rows
    df_clean = df[~is_problematic].reset_index(drop=True)
    print(f"\n  Rows after dropping NaN/Inf : {len(df_clean):,}")

    return df_clean


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading dataset...")
    df = load_raw(DATA_DIR)
    df = apply_feature_selection(df)
    print(f"Loaded: {len(df):,} rows × {len(df.columns)} columns")

    # ── Step 1: Remove duplicates ─────────────────────────────────────────────
    print("\nStep 1 — Removing duplicate rows:")
    df = remove_duplicates(df)

    # ── Step 2: NaN / Infinity analysis and removal ───────────────────────────
    print("\nStep 2 — NaN and Infinity rows:")
    df = analyze_and_plot_nan_inf(df)

    # ── Save cleaned dataset ──────────────────────────────────────────────────
    out_path = os.path.join(PROCESSED_DIR, "clean.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\nCleaned dataset saved: {out_path}")
    print(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nClass distribution after cleaning:")
    counts = df["Label"].value_counts()
    pct = counts / len(df) * 100
    for label, count in counts.items():
        print(f"  {label:<45} {count:>9,}  ({pct[label]:.2f}%)")


if __name__ == "__main__":
    main()
