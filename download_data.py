"""
Downloads CIC-IDS2017 CSV files from the Canadian Institute for Cybersecurity.
Files are saved to ./data/raw/
"""

import os
import zipfile
import requests
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")

# CIC HTTP server — direct ZIP download
CSV_ZIP_URL = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSV.zip"

EXPECTED_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]


def download_file(url: str, dest: str) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dest)) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    existing = [f for f in EXPECTED_FILES if os.path.exists(os.path.join(DATA_DIR, f))]
    if len(existing) == len(EXPECTED_FILES):
        print(f"All {len(EXPECTED_FILES)} CSV files already present in {DATA_DIR}")
        return

    zip_path = os.path.join(DATA_DIR, "CSV.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading from {CSV_ZIP_URL} ...")
        print("(קובץ גדול ~1GB — זה ייקח כמה דקות)")
        try:
            download_file(CSV_ZIP_URL, zip_path)
        except Exception as e:
            print(f"\nהורדה נכשלה: {e}")
            print("\nהורד ידנית מ: https://www.unb.ca/cic/datasets/ids-2017.html")
            print(f"ושמור את קבצי ה-CSV בתיקייה: {DATA_DIR}")
            return

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in tqdm(z.namelist(), desc="Extracting"):
            z.extract(member, DATA_DIR)

    os.remove(zip_path)
    print(f"\nDone. קבצים נשמרו ב: {DATA_DIR}")


if __name__ == "__main__":
    main()
