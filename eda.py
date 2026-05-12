"""
Step 2 — Exploratory Data Analysis
Prints key statistics and saves plots to ./output/eda/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "eda")


def run_eda(df: pd.DataFrame) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Basic info ────────────────────────────────────────────────────────
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage   : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"Duplicate rows : {df.duplicated().sum():,}")

    # ── 2. Label distribution ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION")
    print("=" * 60)
    label_counts = df["Label"].value_counts()
    label_pct = label_counts / len(df) * 100
    label_summary = pd.DataFrame({"count": label_counts, "percent": label_pct.round(2)})
    print(label_summary.to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    label_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Traffic Label Distribution (CIC-IDS2017)")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    for bar in ax.patches:
        ax.annotate(f"{int(bar.get_height()):,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "label_distribution.png"), dpi=150)
    plt.close()
    print(f"\n[saved] label_distribution.png")

    # ── 3. Feature types & missing values ───────────────────────────────────
    print("\n" + "=" * 60)
    print("FEATURE TYPES")
    print("=" * 60)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    other_cols = [c for c in df.columns if c not in numeric_cols]
    print(f"Numeric features : {len(numeric_cols)}")
    print(f"Other columns    : {other_cols}")

    # ── 4. Basic statistics for numeric features ─────────────────────────────
    print("\n" + "=" * 60)
    print("NUMERIC FEATURE STATISTICS (sample)")
    print("=" * 60)
    stats = df[numeric_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    stats.columns = ["mean", "std", "min", "median", "max"]
    print(stats.round(2).to_string())

    # ── 5. Correlation heatmap (top 20 features by variance) ─────────────────
    top_var_cols = df[numeric_cols].var().nlargest(20).index.tolist()
    corr = df[top_var_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, linewidths=0.3,
                annot=False, square=True)
    ax.set_title("Correlation Heatmap — Top 20 Features by Variance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()
    print(f"\n[saved] correlation_heatmap.png")

    # ── 6. Feature distributions by label (top 6 discriminative features) ────
    # Use mean absolute difference between benign and attack as proxy
    benign = df[df["Label"] == "BENIGN"][numeric_cols]
    attack = df[df["Label"] != "BENIGN"][numeric_cols]
    diff = (attack.mean() - benign.mean()).abs().nlargest(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(diff):
        ax = axes[i]
        benign[col].clip(upper=benign[col].quantile(0.99)).plot.hist(
            bins=50, ax=ax, alpha=0.6, label="BENIGN", density=True, color="green")
        attack[col].clip(upper=attack[col].quantile(0.99)).plot.hist(
            bins=50, ax=ax, alpha=0.6, label="ATTACK", density=True, color="red")
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=8)
    plt.suptitle("Top 6 Most Discriminative Features (Benign vs Attack)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_features_distribution.png"), dpi=150)
    plt.close()
    print(f"[saved] top_features_distribution.png")

    # ── 7. Samples per source file ───────────────────────────────────────────
    if "source_file" in df.columns:
        print("\n" + "=" * 60)
        print("SAMPLES PER DAY/FILE")
        print("=" * 60)
        print(df.groupby("source_file")["Label"].value_counts().to_string())

    print(f"\nEDA complete. Plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    df = load_dataset()
    run_eda(df)
