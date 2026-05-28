"""
Shared utilities for model evaluation and result management.

Every model script calls:
  - remap_labels()            — merges Web Attack subclasses into one label
  - save_model_results()      — saves confusion matrix PNG + per-class CSV
                                to output/models/<model_name>/
  - save_feature_importance() — saves top-N feature importance bar chart + CSV
  - update_comparison()       — appends one row to output/models/comparison.csv
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_OUTPUT_DIR = os.path.join(ROOT_DIR, "output", "models")
COMPARISON_CSV = os.path.join(MODELS_OUTPUT_DIR, "comparison.csv")

# Classes tracked explicitly in the comparison table
FOCUS_CLASSES = [
    "Bot",
    "Web Attack",       # merged from XSS + Brute Force + Sql Injection
    "Heartbleed",
    "Infiltration",
]


# ── Label remapping ───────────────────────────────────────────────────────────

def remap_labels(y: pd.Series) -> pd.Series:
    """Merge the three Web Attack subclasses into a single 'Web Attack' label.

    The three subclasses (Brute Force, XSS, Sql Injection) are too small
    and too similar to each other for the model to distinguish reliably.
    Merging them gives the model more samples to learn from and reduces
    the number of classes from 15 to 13.
    """
    return y.apply(lambda label: "Web Attack" if "Web Attack" in str(label) else label)


# ── Model results ─────────────────────────────────────────────────────────────

def save_model_results(
    model_name: str,
    y_test: pd.Series,
    y_pred: np.ndarray,
    per_class_df: pd.DataFrame,
    class_names: list[str],
) -> None:
    """Save confusion matrix PNG and per-class CSV to output/models/<model_name>/."""
    out_dir = os.path.join(MODELS_OUTPUT_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred, labels=class_names, normalize="true")

    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(
        cm,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.3,
        vmin=0,
        vmax=1,
        annot_kws={"size": 7},
    )
    ax.set_title(
        f"Confusion Matrix — {model_name}  (row-normalized, diagonal = Recall)",
        fontsize=12,
        pad=15,
    )
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {cm_path}")

    # ── Per-class CSV ─────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "results.csv")
    per_class_df.to_csv(csv_path)
    print(f"  Saved: {csv_path}")


def save_feature_importance(
    model_name: str,
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 20,
) -> None:
    """Save a bar chart and CSV of the top-N most important features.

    Uses the mean decrease in impurity (MDI) from the Random Forest,
    which is stored in model.feature_importances_ after training.
    Features are sorted by importance descending.
    """
    out_dir = os.path.join(MODELS_OUTPUT_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "feature_importance.csv")
    importance_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # ── Bar chart (top N) ─────────────────────────────────────────────────────
    top = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    bars = ax.barh(top["feature"][::-1], top["importance"][::-1],
                   color="steelblue", edgecolor="white")

    # Label each bar with its value
    for bar, val in zip(bars, top["importance"][::-1]):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=12, pad=12)
    ax.set_xlabel("Mean Decrease in Impurity (MDI)")
    plt.tight_layout()

    chart_path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {chart_path}")

    # Print top 10 to terminal
    print(f"\n  Top 10 features:")
    for _, row in importance_df.head(10).iterrows():
        bar = "█" * int(row["importance"] * 300)
        print(f"    {row['feature']:<35} {row['importance']:.4f}  {bar}")


def update_comparison(
    model_name: str,
    y_test: pd.Series,
    y_pred: np.ndarray,
    per_class_df: pd.DataFrame,
) -> None:
    """Append one row to the central comparison CSV.

    Columns: model, f1_macro, and recall for each focus class.
    If the file already has a row for this model, it is replaced.
    """
    os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)

    f1_macro = f1_score(y_test, y_pred, average="macro")
    row = {"model": model_name, "f1_macro": round(f1_macro, 4)}

    for cls in FOCUS_CLASSES:
        col = f"recall_{cls.lower().replace(' ', '_')}"
        row[col] = round(per_class_df.loc[cls, "recall"], 3) if cls in per_class_df.index else None

    # Load existing comparison table (or create new one)
    if os.path.exists(COMPARISON_CSV):
        df = pd.read_csv(COMPARISON_CSV)
        df = df[df["model"] != model_name]
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(COMPARISON_CSV, index=False)
    print(f"  Saved: {COMPARISON_CSV}")

    # Print comparison table so far
    print(f"\n{'=' * 65}")
    print("  MODEL COMPARISON")
    print(f"{'=' * 65}")
    print(df.to_string(index=False))
