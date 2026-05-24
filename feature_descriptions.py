"""
CIC-IDS2017 — Feature Descriptions
Prints a table of all 79 features and saves it to output/feature_descriptions.csv
"""

import os
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

FEATURES = [
    # ── Connection identifiers ───────────────────────────────────────────────
    ("Destination Port",            "Connection",   "Port number of the destination (e.g. 80=HTTP, 443=HTTPS, 22=SSH)"),

    # ── Flow duration ────────────────────────────────────────────────────────
    ("Flow Duration",               "Time",         "Total duration of the flow in microseconds"),

    # ── Packet counts ────────────────────────────────────────────────────────
    ("Total Fwd Packets",           "Volume",       "Total number of packets sent in the forward direction (client → server)"),
    ("Total Backward Packets",      "Volume",       "Total number of packets sent in the backward direction (server → client)"),

    # ── Packet sizes ─────────────────────────────────────────────────────────
    ("Total Length of Fwd Packets", "Volume",       "Total bytes of payload in forward packets"),
    ("Total Length of Bwd Packets", "Volume",       "Total bytes of payload in backward packets"),
    ("Fwd Packet Length Max",       "Packet Size",  "Largest forward packet size (bytes)"),
    ("Fwd Packet Length Min",       "Packet Size",  "Smallest forward packet size (bytes)"),
    ("Fwd Packet Length Mean",      "Packet Size",  "Average forward packet size (bytes)"),
    ("Fwd Packet Length Std",       "Packet Size",  "Standard deviation of forward packet sizes"),
    ("Bwd Packet Length Max",       "Packet Size",  "Largest backward packet size (bytes)"),
    ("Bwd Packet Length Min",       "Packet Size",  "Smallest backward packet size (bytes)"),
    ("Bwd Packet Length Mean",      "Packet Size",  "Average backward packet size (bytes)"),
    ("Bwd Packet Length Std",       "Packet Size",  "Standard deviation of backward packet sizes"),

    # ── Flow rates ───────────────────────────────────────────────────────────
    ("Flow Bytes/s",                "Rate",         "Total bytes per second across the entire flow"),
    ("Flow Packets/s",              "Rate",         "Total packets per second across the entire flow"),

    # ── Inter-Arrival Times (IAT) — full flow ────────────────────────────────
    ("Flow IAT Mean",               "Timing",       "Mean time between any two consecutive packets in the flow (µs)"),
    ("Flow IAT Std",                "Timing",       "Standard deviation of inter-arrival times"),
    ("Flow IAT Max",                "Timing",       "Longest gap between two consecutive packets (µs)"),
    ("Flow IAT Min",                "Timing",       "Shortest gap between two consecutive packets (µs)"),

    # ── IAT — forward direction ──────────────────────────────────────────────
    ("Fwd IAT Total",               "Timing",       "Total time between forward packets (µs)"),
    ("Fwd IAT Mean",                "Timing",       "Mean inter-arrival time of forward packets (µs)"),
    ("Fwd IAT Std",                 "Timing",       "Standard deviation of forward IAT"),
    ("Fwd IAT Max",                 "Timing",       "Longest gap between forward packets (µs)"),
    ("Fwd IAT Min",                 "Timing",       "Shortest gap between forward packets (µs)"),

    # ── IAT — backward direction ─────────────────────────────────────────────
    ("Bwd IAT Total",               "Timing",       "Total time between backward packets (µs)"),
    ("Bwd IAT Mean",                "Timing",       "Mean inter-arrival time of backward packets (µs)"),
    ("Bwd IAT Std",                 "Timing",       "Standard deviation of backward IAT"),
    ("Bwd IAT Max",                 "Timing",       "Longest gap between backward packets (µs)"),
    ("Bwd IAT Min",                 "Timing",       "Shortest gap between backward packets (µs)"),

    # ── TCP flags ────────────────────────────────────────────────────────────
    ("Fwd PSH Flags",               "TCP Flags",    "Number of forward packets with PSH flag (push data immediately to application)"),
    ("Bwd PSH Flags",               "TCP Flags",    "Number of backward packets with PSH flag"),
    ("Fwd URG Flags",               "TCP Flags",    "Number of forward packets with URG flag (urgent data)"),
    ("Bwd URG Flags",               "TCP Flags",    "Number of backward packets with URG flag"),
    ("FIN Flag Count",              "TCP Flags",    "Number of packets with FIN flag (connection teardown)"),
    ("SYN Flag Count",              "TCP Flags",    "Number of packets with SYN flag (connection initiation) — high values may indicate SYN flood"),
    ("RST Flag Count",              "TCP Flags",    "Number of packets with RST flag (abrupt connection reset)"),
    ("PSH Flag Count",              "TCP Flags",    "Total PSH flags across all packets"),
    ("ACK Flag Count",              "TCP Flags",    "Number of packets with ACK flag (acknowledgement)"),
    ("URG Flag Count",              "TCP Flags",    "Total URG flags across all packets"),
    ("CWE Flag Count",              "TCP Flags",    "Number of packets with CWE flag (Congestion Window Reduced Echo)"),
    ("ECE Flag Count",              "TCP Flags",    "Number of packets with ECE flag (Explicit Congestion Notification Echo)"),

    # ── Header lengths ───────────────────────────────────────────────────────
    ("Fwd Header Length",           "Header",       "Total bytes used by headers in forward packets"),
    ("Bwd Header Length",           "Header",       "Total bytes used by headers in backward packets"),

    # ── Packet rates ─────────────────────────────────────────────────────────
    ("Fwd Packets/s",               "Rate",         "Forward packets per second"),
    ("Bwd Packets/s",               "Rate",         "Backward packets per second"),

    # ── Overall packet sizes ─────────────────────────────────────────────────
    ("Min Packet Length",           "Packet Size",  "Smallest packet in the entire flow (bytes)"),
    ("Max Packet Length",           "Packet Size",  "Largest packet in the entire flow (bytes)"),
    ("Packet Length Mean",          "Packet Size",  "Average packet size across the entire flow (bytes)"),
    ("Packet Length Std",           "Packet Size",  "Standard deviation of packet sizes"),
    ("Packet Length Variance",      "Packet Size",  "Variance of packet sizes (Std²)"),

    # ── Ratios ───────────────────────────────────────────────────────────────
    ("Down/Up Ratio",               "Ratio",        "Ratio of download (backward) to upload (forward) traffic"),
    ("Average Packet Size",         "Packet Size",  "Mean packet size across both directions (bytes)"),
    ("Avg Fwd Segment Size",        "Packet Size",  "Mean forward TCP segment size (same as Fwd Packet Length Mean)"),
    ("Avg Bwd Segment Size",        "Packet Size",  "Mean backward TCP segment size"),

    # ── Bulk transfer statistics ─────────────────────────────────────────────
    ("Fwd Avg Bytes/Bulk",          "Bulk",         "Average bytes per bulk transfer in the forward direction"),
    ("Fwd Avg Packets/Bulk",        "Bulk",         "Average packets per bulk transfer in the forward direction"),
    ("Fwd Avg Bulk Rate",           "Bulk",         "Average bulk transfer rate in the forward direction (bytes/s)"),
    ("Bwd Avg Bytes/Bulk",          "Bulk",         "Average bytes per bulk transfer in the backward direction"),
    ("Bwd Avg Packets/Bulk",        "Bulk",         "Average packets per bulk transfer in the backward direction"),
    ("Bwd Avg Bulk Rate",           "Bulk",         "Average bulk transfer rate in the backward direction (bytes/s)"),

    # ── Subflow statistics ───────────────────────────────────────────────────
    ("Subflow Fwd Packets",         "Subflow",      "Mean number of forward packets per subflow (a flow split into time windows)"),
    ("Subflow Fwd Bytes",           "Subflow",      "Mean bytes of forward packets per subflow"),
    ("Subflow Bwd Packets",         "Subflow",      "Mean number of backward packets per subflow"),
    ("Subflow Bwd Bytes",           "Subflow",      "Mean bytes of backward packets per subflow"),

    # ── TCP window ───────────────────────────────────────────────────────────
    ("Init_Win_bytes_forward",      "TCP Window",   "Initial TCP window size advertised in the forward direction (bytes) — reflects receiver buffer"),
    ("Init_Win_bytes_backward",     "TCP Window",   "Initial TCP window size advertised in the backward direction (bytes)"),

    # ── Additional flow features ─────────────────────────────────────────────
    ("act_data_pkt_fwd",            "Volume",       "Number of forward packets that actually carried payload (data packets, not pure ACKs)"),
    ("min_seg_size_forward",        "Packet Size",  "Minimum TCP segment size observed in the forward direction"),

    # ── Active/Idle times ────────────────────────────────────────────────────
    ("Active Mean",                 "Activity",     "Mean time the flow was active before going idle (µs)"),
    ("Active Std",                  "Activity",     "Standard deviation of active periods"),
    ("Active Max",                  "Activity",     "Longest active period (µs)"),
    ("Active Min",                  "Activity",     "Shortest active period (µs)"),
    ("Idle Mean",                   "Activity",     "Mean time the flow was idle between active periods (µs)"),
    ("Idle Std",                    "Activity",     "Standard deviation of idle periods"),
    ("Idle Max",                    "Activity",     "Longest idle period (µs)"),
    ("Idle Min",                    "Activity",     "Shortest idle period (µs)"),

    # ── Target label ─────────────────────────────────────────────────────────
    ("Label",                       "Target",       "Traffic classification: BENIGN or attack type (DDoS, PortScan, Brute Force, etc.)"),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.DataFrame(FEATURES, columns=["Feature", "Category", "Description"])

    # Print to terminal
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.max_rows", 100)
    print(df.to_string(index=False))

    # Save to CSV
    out_path = os.path.join(OUTPUT_DIR, "feature_descriptions.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")

    # Summary by category
    print("\nFeatures per category:")
    print(df["Category"].value_counts().to_string())


if __name__ == "__main__":
    main()
