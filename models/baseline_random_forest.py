"""
Baseline Model — Random Forest (no class balancing).

Trains a Random Forest classifier on the engineered dataset without any
class balancing techniques. Serves as the reference point for all
subsequent models that apply balancing or other improvements.

Metrics reported:
  - F1 Macro
  - Per-class Recall, Precision, F1 (sorted by Recall ascending)
  - Confusion Matrix (normalized) saved as PNG
  - Feature importance (top 20) saved as PNG and CSV

Output:
  - Prints results to terminal.
  - Saves output/models/baseline_engineered/confusion_matrix.png
  - Saves output/models/baseline_engineered/results.csv
  - Saves output/models/baseline_engineered/feature_importance.png
  - Saves output/models/baseline_engineered/feature_importance.csv
  - Updates output/models/comparison.csv
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from models.utils import remap_labels, save_feature_importance, save_model_results, update_comparison

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
INPUT_PARQUET = os.path.join(PROCESSED_DIR, "engineered.parquet")

MODEL_NAME = "baseline_engineered"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load engineered dataset and return features X and labels y."""
    print("Loading engineered dataset...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded: {len(df):,} rows x {df.shape[1]} columns")

    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[feature_cols]
    y = remap_labels(df["Label"])

    print(f"Features : {X.shape[1]}")
    print(f"Classes  : {y.nunique()}")
    return X, y


# ── Training ──────────────────────────────────────────────────────────────────

def train(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train a Random Forest with no class balancing."""
    print(f"\nTraining Random Forest  "
          f"(n_estimators={N_ESTIMATORS}, no class balancing)...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    print("Training complete.")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Predict and print F1 Macro + per-class metrics."""
    y_pred = model.predict(X_test)

    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(f"\n{'=' * 65}")
    print(f"  F1 Macro : {f1_macro:.4f}")
    print(f"{'=' * 65}")

    # Per-class breakdown sorted by Recall ascending (worst classes first)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).T
    per_class = report_df.loc[class_names].copy()
    per_class = per_class.sort_values("recall", ascending=True)

    print(f"\n  {'Class':<45} {'Recall':>8} {'Precision':>10} {'F1':>8} {'Support':>10}")
    print("  " + "-" * 85)
    for label, row in per_class.iterrows():
        print(f"  {label:<45} {row['recall']:>8.3f} "
              f"{row['precision']:>10.3f} {row['f1-score']:>8.3f} "
              f"{int(row['support']):>10,}")

    return per_class, y_pred


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    X, y = load_data()
    class_names = sorted(y.unique())

    # Stratified split — preserves class proportions in both train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\nTrain : {len(X_train):,} rows")
    print(f"Test  : {len(X_test):,} rows")

    model = train(X_train, y_train)
    per_class, y_pred = evaluate(model, X_test, y_test, class_names)

    print(f"\nSaving results...")
    save_model_results(MODEL_NAME, y_test, y_pred, per_class, class_names)
    save_feature_importance(MODEL_NAME, X_train.columns.tolist(), model.feature_importances_)
    update_comparison(MODEL_NAME, y_test, y_pred, per_class)


if __name__ == "__main__":
    main()
