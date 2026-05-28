"""
Feature Engineering — Create new features from the cleaned dataset.

Loads data/processed/clean.parquet and adds the following feature groups:

  Group 1 — Destination Port categories (9 features):
      Binary flags for well-known port ranges and specific sensitive ports.
      Replaces the raw port number (53,805 unique values) with interpretable signals.

  Group 2 — Packet/byte ratios (2 features):
      Ratios between existing volume features.
      Note: only forward-direction counts survived feature selection.
      Down/Up Ratio already exists in the dataset and covers fwd/bwd asymmetry.

  Group 3 — Packet size variability (2 features):
      Range (max - min) of packet sizes per direction.
      Uniform packet sizes suggest automated/scripted traffic (DoS, scans).

  Group 4 — TCP flag ratios (3 features):
      Fraction of packets carrying RST, FIN, or PSH flags.
      Note: SYN Flag Count was removed during feature selection.

  Group 5 — Flow duration bins (3 features):
      Binary flags for very short/long flows + duration per packet.
      Port scans produce very short flows; bots and infiltration produce long ones.

The original clean.parquet is not modified.

Output:
  - Prints feature counts and statistics for all new features.
  - Saves data/processed/engineered.parquet
"""

import os

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
INPUT_PATH = os.path.join(PROCESSED_DIR, "clean.parquet")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "engineered.parquet")


# ── Group 1: Destination Port categories ─────────────────────────────────────

def add_port_features(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flags based on destination port number.

    Port ranges follow IANA standards:
      0–1023    : Well-Known ports (HTTP, HTTPS, SSH, DNS, FTP, ...)
      1024–49151: Registered ports
      49152+    : Dynamic / ephemeral ports

    Specific ports are flagged individually because they are commonly
    targeted in attacks or indicate specific protocols.
    """
    port = df["Destination Port"]

    df["port_is_well_known"] = (port <= 1023).astype(int)
    df["port_is_registered"] = ((port >= 1024) & (port <= 49151)).astype(int)
    df["port_is_dynamic"]    = (port >= 49152).astype(int)

    df["port_is_http"]  = (port == 80).astype(int)
    df["port_is_https"] = (port == 443).astype(int)
    df["port_is_ssh"]   = (port == 22).astype(int)
    df["port_is_dns"]   = (port == 53).astype(int)
    df["port_is_ftp"]   = (port == 21).astype(int)
    df["port_is_rdp"]   = (port == 3389).astype(int)  # Remote Desktop — common attack target

    return df


# ── Group 2: Packet and byte ratios ──────────────────────────────────────────

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios between volume features.

    +1 added to denominators to avoid division by zero.

    Total Backward Packets and Total Length of Bwd Packets were removed
    during feature selection, so only forward-direction ratios are computed.
    Down/Up Ratio (already in the dataset) covers the fwd/bwd asymmetry.
    """
    # Average bytes per forward packet
    df["fwd_bytes_per_packet"] = (
        df["Total Length of Fwd Packets"] / (df["Total Fwd Packets"] + 1)
    )

    # Header overhead relative to forward payload — high ratio = mostly headers, little data
    df["header_to_payload_ratio"] = (
        (df["Fwd Header Length"] + df["Bwd Header Length"] + 1)
        / (df["Total Length of Fwd Packets"] + 1)
    )

    return df


# ── Group 3: Packet size variability ─────────────────────────────────────────

def add_variability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Range of packet sizes in forward and backward directions.

    Automated or scripted traffic (DoS, port scans) tends to send packets
    of uniform size. Legitimate traffic has more variation.
    A small range (max - min ≈ 0) suggests machine-generated traffic.
    """
    df["fwd_packet_size_range"] = (
        df["Fwd Packet Length Max"] - df["Fwd Packet Length Min"]
    )
    df["bwd_packet_size_range"] = (
        df["Bwd Packet Length Max"] - df["Bwd Packet Length Min"]
    )

    return df


# ── Group 4: TCP flag ratios ──────────────────────────────────────────────────

def add_flag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of forward packets carrying specific TCP flags.

    +1 added to denominator to avoid division by zero.

    - High RST ratio: server rejecting connections — typical of port scanning
    - High FIN ratio: repeated connection teardowns — typical of teardown attacks
    - High PSH ratio: data-heavy communication — can indicate exfiltration

    Note: SYN Flag Count was removed during feature selection and is unavailable.
    """
    total_fwd = df["Total Fwd Packets"] + 1

    df["rst_ratio"] = df["RST Flag Count"] / total_fwd
    df["fin_ratio"] = df["FIN Flag Count"] / total_fwd
    df["psh_ratio"] = df["PSH Flag Count"] / total_fwd

    return df


# ── Group 5: Flow duration bins ───────────────────────────────────────────────

def add_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flags for extreme flow durations and duration-per-packet ratio.

    Flow Duration is measured in microseconds:
      1 ms  =     1,000 µs
      1 s   = 1,000,000 µs
      60 s  = 60,000,000 µs

    - Very short flows (< 1ms): typical of port scans.
    - Very long flows (> 60s): typical of botnet C&C or infiltration sessions.
    - duration_per_fwd_packet: low = flood attack, high = slow/stealthy attack.
    """
    df["is_very_short_flow"] = (df["Flow Duration"] < 1_000).astype(int)
    df["is_very_long_flow"]  = (df["Flow Duration"] > 60_000_000).astype(int)

    df["duration_per_fwd_packet"] = (
        df["Flow Duration"] / (df["Total Fwd Packets"] + 1)
    )

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading cleaned dataset...")
    df = pd.read_parquet(INPUT_PATH)
    n_original = df.shape[1]
    print(f"Loaded: {len(df):,} rows x {n_original} columns")

    print("\nAdding new features...")
    df = add_port_features(df)
    df = add_ratio_features(df)
    df = add_variability_features(df)
    df = add_flag_features(df)
    df = add_duration_features(df)

    n_new = df.shape[1] - n_original
    print(f"Added {n_new} new features  |  Total columns: {df.shape[1]}")

    # ── Summary of new features ───────────────────────────────────────────────
    new_cols = df.columns[n_original:].tolist()
    print(f"\nNew features ({n_new}):")
    print(f"  {'Feature':<30} {'Type':<10} {'Info'}")
    print("  " + "-" * 78)
    for col in new_cols:
        if df[col].nunique() <= 2:
            counts = df[col].value_counts().to_dict()
            info = f"0={counts.get(0, 0):,}   1={counts.get(1, 0):,}"
            print(f"  {col:<30} {'binary':<10} {info}")
        else:
            print(f"  {col:<30} {'numeric':<10} "
                  f"mean={df[col].mean():.3f}  "
                  f"std={df[col].std():.3f}  "
                  f"min={df[col].min():.3f}  "
                  f"max={df[col].max():.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()
