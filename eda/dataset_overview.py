"""
Dataset Overview — CIC-IDS2017
Prints:
  1. Shape (rows x columns)
  2. Missing values per column with percentages
  3. Class distribution with counts and percentages
  4. Label descriptions (cybersecurity context)
"""

import glob
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# ── Label descriptions from cybersecurity literature ────────────────────────
LABEL_DESCRIPTIONS = {
    "BENIGN": (
        "Normal network traffic. No malicious activity. "
        "Used as the negative class in intrusion detection."
    ),
    "DoS Hulk": (
        "Denial-of-Service attack using the Hulk tool. "
        "Floods a web server with unique HTTP GET requests, bypassing caches "
        "and exhausting server resources."
    ),
    "PortScan": (
        "Reconnaissance attack. The attacker probes a range of ports on a target host "
        "to discover open services and potential entry points (e.g. using Nmap)."
    ),
    "DDoS": (
        "Distributed Denial-of-Service. Multiple compromised machines send traffic "
        "simultaneously to overwhelm the target, making it unavailable to legitimate users."
    ),
    "DoS GoldenEye": (
        "DoS attack using the GoldenEye tool. Keeps HTTP connections open by sending "
        "partial requests, exhausting the server's connection pool."
    ),
    "FTP-Patator": (
        "Brute-force attack against FTP servers using the Patator tool. "
        "Systematically tries username/password combinations to gain unauthorized access."
    ),
    "SSH-Patator": (
        "Brute-force attack against SSH servers using the Patator tool. "
        "Attempts many password guesses to break into remote shell access."
    ),
    "DoS slowloris": (
        "DoS attack using Slowloris. Holds many connections open to a web server "
        "by sending partial HTTP headers very slowly, starving the server of connections."
    ),
    "DoS Slowhttptest": (
        "DoS attack using Slowhttptest. Similar to Slowloris — sends slow HTTP requests "
        "(headers or body) to exhaust server threads and connection limits."
    ),
    "Bot": (
        "Botnet traffic. A compromised host (bot) communicates with a Command & Control (C2) "
        "server to receive instructions for spam, DDoS, credential theft, or data exfiltration."
    ),
    "Web Attack - Brute Force": (
        "Brute-force attack against a web application login form. "
        "Tries many username/password pairs to gain unauthorized access to user accounts."
    ),
    "Web Attack - XSS": (
        "Cross-Site Scripting attack. Injects malicious JavaScript into a web page, "
        "which then executes in the victim's browser to steal cookies or credentials."
    ),
    "Web Attack - Sql Injection": (
        "SQL Injection attack. Malicious SQL code is inserted into input fields to "
        "manipulate the database — dumping data, bypassing authentication, or deleting records."
    ),
    "Infiltration": (
        "Advanced infiltration attack. The attacker is already inside the network and "
        "performs lateral movement, data exfiltration, or installs backdoors. "
        "Very rare and hard to detect."
    ),
    "Heartbleed": (
        "Exploit of the Heartbleed vulnerability (CVE-2014-0160) in OpenSSL. "
        "Allows reading server memory without authentication, leaking private keys, "
        "passwords, and sensitive data."
    ),
}


def load_raw(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].str.strip()
    return df


def print_section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")
    df = load_raw(DATA_DIR)

    # ── 1. Shape ─────────────────────────────────────────────────────────────
    print_section("1. SHAPE")
    rows, cols = df.shape
    print(f"  Rows    : {rows:,}")
    print(f"  Columns : {cols}")

    # ── 2. Missing values ─────────────────────────────────────────────────────
    print_section("2. MISSING VALUES")

    # Replace inf/-inf with NaN for this analysis
    df_check = df.replace([np.inf, -np.inf], np.nan)
    missing = df_check.isnull().sum()
    missing_pct = missing / len(df_check) * 100
    missing_df = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct.round(2),
    })
    missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(
        "Missing Count", ascending=False
    )

    if missing_df.empty:
        print("  No missing values found.")
    else:
        print(missing_df.to_string())

    # Inf values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    print(f"\n  Infinite values (inf/-inf): {inf_count:,} "
          f"({inf_count / (rows * len(numeric_cols)) * 100:.3f}% of numeric cells)")

    # ── 3. Class distribution ─────────────────────────────────────────────────
    print_section("3. CLASS DISTRIBUTION  (15 classes)")

    counts = df["Label"].value_counts()
    pct = counts / len(df) * 100
    class_df = pd.DataFrame({
        "Count": counts,
        "Percent %": pct.round(3),
    })
    print(class_df.to_string())

    benign_pct = pct.get("BENIGN", 0)
    attack_pct = 100 - benign_pct
    print(f"\n  BENIGN  : {benign_pct:.1f}%")
    print(f"  ATTACKS : {attack_pct:.1f}%")
    print(f"  Imbalance ratio (BENIGN / rarest attack): "
          f"{counts.get('BENIGN', 0) / counts.iloc[-1]:,.0f}:1")

    # ── 4. Label descriptions ─────────────────────────────────────────────────
    print_section("4. LABEL DESCRIPTIONS")

    for label, count in counts.items():
        desc = LABEL_DESCRIPTIONS.get(label, "No description available.")
        pct_val = pct[label]
        print(f"\n  [{label}]  ({count:,} samples, {pct_val:.3f}%)")
        # Word-wrap description at 60 chars
        words = desc.split()
        line, lines = "", []
        for word in words:
            if len(line) + len(word) + 1 > 60:
                lines.append(line)
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            lines.append(line)
        for l in lines:
            print(f"    {l}")

    # ── Save summary to CSV ───────────────────────────────────────────────────
    class_df.to_csv(os.path.join(OUTPUT_DIR, "class_distribution.csv"))
    missing_df.to_csv(os.path.join(OUTPUT_DIR, "missing_values.csv"))
    print(f"\n\nSaved: output/class_distribution.csv")
    print(f"Saved: output/missing_values.csv")


if __name__ == "__main__":
    main()
