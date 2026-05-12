"""
CIC-IDS2017 — Network Intrusion Detection
Entry point: runs all pipeline steps in order.
"""

from load_data import load_dataset
from eda import run_eda


if __name__ == "__main__":
    df = load_dataset()
    run_eda(df)
