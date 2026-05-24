"""
Feature Selection — Step 1: Remove Redundant and Useless Features.

Two removal passes:
  Pass 1 — Zero-variance features:
      Automatically detect any numeric feature whose value never changes
      (i.e. only one unique value across all rows). Such features carry
      no information and cannot help any classifier.

  Pass 2 — High-correlation filter:
      For every pair of features, if |r| >= THRESHOLD, drop the second
      one. Once a feature is marked for dropping it is skipped in all
      further comparisons, avoiding redundant checks.

Output:
  - Prints every dropped feature and the reason to the terminal.
  - Saves output/features_to_keep.csv
  - Saves output/dropped_features_heatmap.png
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Features with |r| >= THRESHOLD are considered redundant; the second one is dropped.
THRESHOLD = 0.9


# ── Data loading ─────────────────────────────────────────────────────────────

def load_raw(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()
    return df


# ── Pass 1: Zero-variance detection ──────────────────────────────────────────

def find_zero_variance(df: pd.DataFrame) -> list[str]:
    """Return a list of numeric features that have only one unique value.

    These features are useless for classification:
    - Bulk features (Fwd/Bwd Avg Bytes/Packets/Rate): CICFlowMeter found no
      bulk transfers in this capture, so all values are 0.
    - Bwd PSH Flags / Bwd URG Flags: no backward PSH or URG packets were
      observed in any flow, so all values are 0.
    """
    numeric = df.select_dtypes(include=[np.number])
    zero_var = [col for col in numeric.columns if numeric[col].nunique() <= 1]
    return zero_var


# ── Pass 2: High-correlation filter ──────────────────────────────────────────

def correlation_filter(
    df: pd.DataFrame,
    features: list[str],
    threshold: float,
) -> tuple[list[str], dict[str, str]]:
    """Remove features that are highly correlated with an already-kept feature.

    For each pair (A, B) where A comes before B in the feature list:
      - If both are still in to_keep and |corr(A, B)| >= threshold, drop B.
      - Once B is dropped, it is skipped in all subsequent comparisons.

    Returns:
        to_keep   — features that survived the filter.
        drop_reason — mapping from dropped feature to the feature it correlated with.
    """
    print(f"Computing correlation matrix for {len(features)} features...")
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


# ── Heatmap ───────────────────────────────────────────────────────────────────

def plot_dropped_heatmap(
    corr_full: pd.DataFrame,
    drop_reason: dict[str, str],
    output_dir: str,
) -> None:
    """Save a heatmap showing dropped features vs all features involved.

    Rows   = dropped features (sorted alphabetically).
    Columns = all features that appear as either dropped or as the reason for dropping.
    Cell values = absolute correlation between the two features.
    """
    if not drop_reason:
        print("No features dropped — no heatmap to generate.")
        return

    dropped = sorted(drop_reason.keys())
    involved = sorted(set(drop_reason.values()))
    all_cols = sorted(set(dropped + involved))

    # Keep only features that exist in the correlation matrix
    # (zero-variance features are excluded from corr_full)
    dropped_in_corr = [f for f in dropped if f in corr_full.index]
    all_cols_in_corr = [f for f in all_cols if f in corr_full.columns]

    if not dropped_in_corr:
        print("No correlation-dropped features to plot.")
        return

    subset = corr_full.loc[dropped_in_corr, all_cols_in_corr]

    fig_h = max(6, len(dropped_in_corr) * 0.4)
    fig_w = max(8, len(all_cols_in_corr) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        subset,
        ax=ax,
        cmap="Reds",
        vmin=0, vmax=1,
        annot=True, fmt=".2f",
        linewidths=0.3,
        cbar_kws={"label": "Absolute Correlation"},
    )
    ax.set_title(
        f"Dropped Features vs Their Correlated Partners  (threshold={THRESHOLD})",
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
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")
    df = load_raw(DATA_DIR)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    numeric = df.select_dtypes(include=[np.number])
    all_features = list(numeric.columns)
    drop_reason: dict[str, str] = {}

    # ── Pass 1: zero-variance ─────────────────────────────────────────────────
    print("\nPass 1 — Zero-variance features:")
    zero_var = find_zero_variance(df)
    for feat in zero_var:
        drop_reason[feat] = "zero variance — constant value across all rows"
        print(f"  DROP  {feat:<40}  (zero variance)")

    features_after_pass1 = [f for f in all_features if f not in zero_var]

    # ── Pass 2: high-correlation filter ──────────────────────────────────────
    print("\nPass 2 — High-correlation filter (threshold={:.2f}):".format(THRESHOLD))
    corr_full = df[features_after_pass1].corr().abs()
    to_keep, corr_drop_reason = correlation_filter(df, features_after_pass1, THRESHOLD)
    drop_reason.update(corr_drop_reason)

    to_drop = [f for f in all_features if f not in to_keep]

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"Original  : {len(all_features)} features")
    print(f"  Pass 1 (zero variance) : {len(zero_var)} dropped")
    print(f"  Pass 2 (correlation)   : {len(corr_drop_reason)} dropped")
    print(f"To keep   : {len(to_keep)} features")
    print(f"To drop   : {len(to_drop)} features")

    print(f"\nTO DROP ({len(to_drop)}):")
    for f in sorted(to_drop):
        print(f"  - {f:<40}  ({drop_reason.get(f, '')})")

    print(f"\nTO KEEP ({len(to_keep)}):")
    for f in sorted(to_keep):
        print(f"  - {f}")

    # ── Outputs ───────────────────────────────────────────────────────────────
    plot_dropped_heatmap(corr_full, corr_drop_reason, OUTPUT_DIR)

    pd.Series(sorted(to_keep)).to_csv(
        os.path.join(OUTPUT_DIR, "features_to_keep.csv"),
        index=False,
        header=["feature"],
    )
    print(f"\nSaved: output/features_to_keep.csv")


if __name__ == "__main__":
    main()
