"""
Correlation of all kept features with the Label.
Uses point-biserial correlation (numeric feature vs binary label: 0=BENIGN, 1=ATTACK).
Produces a heatmap sorted by absolute correlation.
"""

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
FEATURES_TO_KEEP_PATH = os.path.join(OUTPUT_DIR, "features_to_keep.csv")


def load_raw(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].str.strip()
    return df


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load kept features list
    to_keep = pd.read_csv(FEATURES_TO_KEEP_PATH)["feature"].tolist()

    print("Loading dataset...")
    df = load_raw(DATA_DIR)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=to_keep)

    # Binary label: 1 = attack, 0 = benign
    df["is_attack"] = (df["Label"] != "BENIGN").astype(int)

    # Compute point-biserial correlation for each kept feature
    print(f"Computing correlations for {len(to_keep)} features...")
    results = []
    for feat in to_keep:
        r, pval = pointbiserialr(df[feat], df["is_attack"])
        results.append({"feature": feat, "correlation": r, "abs_correlation": abs(r), "pvalue": pval})

    corr_df = (
        pd.DataFrame(results)
        .sort_values("abs_correlation", ascending=False)
        .reset_index(drop=True)
    )

    # Print table
    print(f"\n{'Feature':<40} {'Correlation':>12} {'|r|':>8}")
    print("-" * 65)
    for _, row in corr_df.iterrows():
        print(f"{row['feature']:<40} {row['correlation']:>12.4f} {row['abs_correlation']:>8.4f}")

    # Heatmap — single column, features sorted by |r|
    fig, ax = plt.subplots(figsize=(4, max(8, len(to_keep) * 0.28)))

    heatmap_data = corr_df.set_index("feature")[["correlation"]]

    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        annot=True, fmt=".2f",
        linewidths=0.3,
        cbar_kws={"label": "Point-Biserial Correlation with Label"},
        annot_kws={"size": 7},
    )
    ax.set_title("Feature Correlation with Label\n(0=BENIGN, 1=ATTACK)", fontsize=11, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "label_correlation_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved: {out_path}")

    # Summary by threshold
    print("\nDistribution by |r| threshold:")
    for t in [0.05, 0.1, 0.2, 0.3]:
        n = (corr_df["abs_correlation"] >= t).sum()
        print(f"  |r| >= {t}: {n} features ({n/len(corr_df)*100:.0f}%)")

    low = (corr_df["abs_correlation"] < 0.05).sum()
    print(f"  |r| <  0.05: {low} features — candidates to drop")


if __name__ == "__main__":
    main()
