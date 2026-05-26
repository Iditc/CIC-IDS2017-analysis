"""
Basic Feature Selection — Three-pass pipeline.

  Pass 1 — Zero-variance:
      Drop any numeric feature whose value never changes across all rows.
      Such features carry no information for any classifier.

  Pass 2 — High inter-feature correlation:
      Drop one of every pair of features with |r| >= CORR_THRESHOLD.
      Once a feature is marked for dropping it is skipped in all
      further comparisons, avoiding redundant checks.

  Pass 3 — Low label correlation:
      From the surviving features, compute point-biserial correlation
      with the binary label (BENIGN=0 / ATTACK=1).
      If LABEL_THRESHOLD is set, features below it are dropped.
      If LABEL_THRESHOLD is None, only the distribution is printed
      so the user can decide on a threshold before committing.

Output:
  - Prints every dropped feature and the reason to the terminal.
  - Saves output/features_to_keep.csv
  - Saves output/dropped_features_heatmap.png   (Pass 2)
  - Saves output/label_correlation_heatmap.png  (Pass 3)
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pointbiserialr

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Pass 2: drop feature B if |corr(A, B)| >= CORR_THRESHOLD
CORR_THRESHOLD = 0.9

# Pass 3: drop feature if |corr(feature, label)| < LABEL_THRESHOLD
# Set to None to inspect the distribution before deciding on a threshold.
LABEL_THRESHOLD = None


# ── Data loading ──────────────────────────────────────────────────────────────

def load_raw(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].str.strip()
    return df


# ── Pass 1: Zero-variance detection ──────────────────────────────────────────

def find_zero_variance(df: pd.DataFrame) -> list[str]:
    """Return numeric features that have only one unique value across all rows.

    Known zero-variance features in CIC-IDS2017:
    - Bulk features (Fwd/Bwd Avg Bytes/Packets/Rate): CICFlowMeter found no
      bulk transfers in this capture, so all values are 0.
    - Bwd PSH Flags / Bwd URG Flags: no backward PSH or URG packets were
      observed in any flow, so all values are 0.
    """
    numeric = df.select_dtypes(include=[np.number])
    return [col for col in numeric.columns if numeric[col].nunique() <= 1]


# ── Pass 2: High inter-feature correlation ────────────────────────────────────

def inter_feature_correlation_filter(
    df: pd.DataFrame,
    features: list[str],
    threshold: float,
) -> tuple[list[str], dict[str, str]]:
    """Drop one feature from every highly correlated pair.

    Iterates over all pairs (A, B) where A precedes B in the feature list.
    If |corr(A, B)| >= threshold and neither is already dropped, B is dropped.
    Skips any feature already marked for dropping to avoid redundant checks.

    Returns:
        to_keep     — features that survived the filter.
        drop_reason — maps each dropped feature to the feature it correlated with.
    """
    print(f"  Computing correlation matrix for {len(features)} features...")
    corr = df[features].corr().abs()

    to_drop: set[str] = set()
    drop_reason: dict[str, str] = {}

    for i, feat_a in enumerate(features):
        if feat_a in to_drop:
            # Already marked for removal — no need to check its pairs.
            continue
        for feat_b in features[i + 1:]:
            if feat_b in to_drop:
                # Already marked for removal — skip.
                continue
            r = corr.loc[feat_a, feat_b]
            if r >= threshold:
                to_drop.add(feat_b)
                drop_reason[feat_b] = feat_a
                print(f"  DROP  {feat_b:<40}  (r={r:.3f} with {feat_a})")

    to_keep = [f for f in features if f not in to_drop]
    return to_keep, drop_reason


# ── Pass 3: Label correlation ─────────────────────────────────────────────────

def label_correlation_filter(
    df: pd.DataFrame,
    features: list[str],
    threshold: float | None,
) -> tuple[list[str], dict[str, str], pd.DataFrame]:
    """Compute point-biserial correlation between each feature and the binary label.

    Binary label: 0 = BENIGN, 1 = ATTACK.
    Point-biserial correlation measures how strongly a numeric feature is
    associated with the binary outcome — equivalent to Pearson r for this case.

    If threshold is set, features with |r| < threshold are dropped.
    If threshold is None, only the distribution is printed (inspection mode).

    Returns:
        to_keep     — surviving features (all features if threshold is None).
        drop_reason — maps each dropped feature to its |r| value as a string.
        corr_df     — full correlation table sorted by |r| descending.
    """
    print(f"  Computing point-biserial correlation for {len(features)} features...")

    df = df.copy()
    df["is_attack"] = (df["Label"] != "BENIGN").astype(int)

    results = []
    for feat in features:
        r, pval = pointbiserialr(df[feat], df["is_attack"])
        results.append({
            "feature":         feat,
            "correlation":     r,
            "abs_correlation": abs(r),
            "pvalue":          pval,
        })

    corr_df = (
        pd.DataFrame(results)
        .sort_values("abs_correlation", ascending=False)
        .reset_index(drop=True)
    )

    # Print full table
    print(f"\n  {'Feature':<40} {'r':>8} {'|r|':>8}")
    print("  " + "-" * 60)
    for _, row in corr_df.iterrows():
        print(f"  {row['feature']:<40} {row['correlation']:>8.4f} {row['abs_correlation']:>8.4f}")

    # Print distribution summary to help choose a threshold
    print("\n  Distribution by |r| threshold:")
    for t in [0.05, 0.1, 0.2, 0.3]:
        n = (corr_df["abs_correlation"] >= t).sum()
        print(f"    |r| >= {t:.2f} : {n:>3} features ({n / len(corr_df) * 100:.0f}%)")

    drop_reason: dict[str, str] = {}

    if threshold is None:
        # Inspection mode — do not drop anything yet.
        # Review the distribution above and set LABEL_THRESHOLD to activate dropping.
        print(f"\n  LABEL_THRESHOLD is None — no features dropped in Pass 3.")
        print(f"  Set LABEL_THRESHOLD in the script to activate dropping.")
        return features, drop_reason, corr_df

    # Drop features whose absolute correlation with the label is below the threshold
    to_drop = corr_df[corr_df["abs_correlation"] < threshold]["feature"].tolist()
    for feat in to_drop:
        r_val = corr_df.loc[corr_df["feature"] == feat, "abs_correlation"].values[0]
        drop_reason[feat] = f"low label correlation |r|={r_val:.4f} < {threshold}"
        print(f"  DROP  {feat:<40}  ({drop_reason[feat]})")

    to_keep = [f for f in features if f not in to_drop]
    return to_keep, drop_reason, corr_df


# ── Heatmaps ──────────────────────────────────────────────────────────────────

def plot_inter_feature_heatmap(
    corr_full: pd.DataFrame,
    drop_reason: dict[str, str],
    output_dir: str,
) -> None:
    """Heatmap: dropped features (rows) vs all involved features (columns).

    Helps verify that the dropped features are indeed highly correlated
    with the features that replaced them.
    """
    if not drop_reason:
        print("  No features dropped in Pass 2 — skipping heatmap.")
        return

    dropped = sorted(drop_reason.keys())
    involved = sorted(set(drop_reason.values()))
    all_cols = sorted(set(dropped + involved))

    # Keep only features present in the correlation matrix
    dropped = [f for f in dropped if f in corr_full.index]
    all_cols = [f for f in all_cols if f in corr_full.columns]

    subset = corr_full.loc[dropped, all_cols]

    fig, ax = plt.subplots(figsize=(max(8, len(all_cols) * 0.5), max(6, len(dropped) * 0.4)))
    sns.heatmap(
        subset, ax=ax, cmap="Reds", vmin=0, vmax=1,
        annot=True, fmt=".2f", linewidths=0.3,
        cbar_kws={"label": "Absolute Correlation"},
    )
    ax.set_title(
        f"Pass 2 — Dropped Features vs Correlated Partners  (threshold={CORR_THRESHOLD})",
        fontsize=11, pad=12,
    )
    ax.set_xlabel("All involved features")
    ax.set_ylabel("Dropped features")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "dropped_features_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_label_correlation_heatmap(corr_df: pd.DataFrame, output_dir: str) -> None:
    """Heatmap: one column showing each feature's correlation with the label.

    Features are sorted by absolute correlation (highest at the top).
    Positive values mean higher feature values are associated with attacks.
    Negative values mean higher feature values are associated with benign traffic.
    """
    fig, ax = plt.subplots(figsize=(4, max(8, len(corr_df) * 0.28)))
    sns.heatmap(
        corr_df.set_index("feature")[["correlation"]],
        ax=ax,
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", linewidths=0.3,
        cbar_kws={"label": "Point-Biserial Correlation with Label"},
        annot_kws={"size": 7},
    )
    ax.set_title(
        "Pass 3 — Feature Correlation with Label\n(BENIGN=0, ATTACK=1)",
        fontsize=11, pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "label_correlation_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")
    df = load_raw(DATA_DIR)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    all_features = list(df.select_dtypes(include=[np.number]).columns)
    all_drop_reasons: dict[str, str] = {}

    # ── Pass 1 ────────────────────────────────────────────────────────────────
    print("\nPass 1 — Zero-variance features:")
    zero_var = find_zero_variance(df)
    for feat in zero_var:
        all_drop_reasons[feat] = "zero variance — constant value across all rows"
        print(f"  DROP  {feat:<40}  (zero variance)")
    features_p1 = [f for f in all_features if f not in zero_var]

    # ── Pass 2 ────────────────────────────────────────────────────────────────
    print(f"\nPass 2 — Inter-feature correlation (threshold={CORR_THRESHOLD}):")
    corr_full = df[features_p1].corr().abs()
    features_p2, drop_p2 = inter_feature_correlation_filter(df, features_p1, CORR_THRESHOLD)
    all_drop_reasons.update(drop_p2)

    # ── Pass 3 ────────────────────────────────────────────────────────────────
    label_str = str(LABEL_THRESHOLD) if LABEL_THRESHOLD else "None (inspection only)"
    print(f"\nPass 3 — Label correlation (threshold={label_str}):")
    features_final, drop_p3, corr_df = label_correlation_filter(df, features_p2, LABEL_THRESHOLD)
    all_drop_reasons.update(drop_p3)

    # ── Summary ───────────────────────────────────────────────────────────────
    to_drop_all = [f for f in all_features if f not in features_final]
    print(f"\n{'=' * 60}")
    print(f"Original features : {len(all_features)}")
    print(f"  Pass 1 (zero variance)       : -{len(zero_var)}")
    print(f"  Pass 2 (inter-feature corr.) : -{len(drop_p2)}")
    print(f"  Pass 3 (label correlation)   : -{len(drop_p3)}")
    print(f"Final features    : {len(features_final)}")

    print(f"\nTO DROP ({len(to_drop_all)}):")
    for f in sorted(to_drop_all):
        print(f"  - {f:<40}  ({all_drop_reasons.get(f, '')})")

    print(f"\nTO KEEP ({len(features_final)}):")
    for f in sorted(features_final):
        print(f"  - {f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    plot_inter_feature_heatmap(corr_full, drop_p2, OUTPUT_DIR)
    plot_label_correlation_heatmap(corr_df, OUTPUT_DIR)

    pd.Series(sorted(features_final)).to_csv(
        os.path.join(OUTPUT_DIR, "features_to_keep.csv"),
        index=False,
        header=["feature"],
    )
    print(f"\nSaved: output/features_to_keep.csv")


if __name__ == "__main__":
    main()
