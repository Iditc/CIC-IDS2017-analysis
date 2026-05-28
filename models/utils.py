"""
Shared utilities for model evaluation and result management.

Every model script calls:
  - save_model_results()  — saves confusion matrix PNG + per-class CSV
                            to output/models/<model_name>/
  - update_comparison()   — appends one row to output/models/comparison.csv
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

# Classes that are hardest to detect — tracked explicitly in comparison table
FOCUS_CLASSES = [
    "Bot",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Web Attack - Sql Injection",
    "Heartbleed",
    "Infiltration",
]


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
        col = f"recall_{cls.lower().replace(' ', '_').replace('-', '_')}"
        if cls in per_class_df.index:
            row[col] = round(per_class_df.loc[cls, "recall"], 3)
        else:
            row[col] = None

    # Load existing comparison table (or create new one)
    if os.path.exists(COMPARISON_CSV):
        df = pd.read_csv(COMPARISON_CSV)
        df = df[df["model"] != model_name]  # remove old row for this model
    else:
        df = pd.DataFrame()

    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(COMPARISON_CSV, index=False)
    print(f"  Saved: {COMPARISON_CSV}")

    # Print comparison table so far
    print(f"\n{'=' * 65}")
    print("  MODEL COMPARISON")
    print(f"{'=' * 65}")
    print(df.to_string(index=False))
