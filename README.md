# CIC-IDS2017-analysis
This project applies machine learning techniques to network intrusion detection using the CIC-IDS2017 dataset. It covers data preprocessing, feature engineering, model training and evaluation — classifying network traffic as benign or malicious across multiple attack categories including DDoS, PortScan, and Brute Force.

## Files

### `load_data.py`
Loads all CIC-IDS2017 CSV files from `data/raw/`, normalizes column names, and drops rows with missing or infinite values. Returns a single unified DataFrame.

### `eda.py` — Exploratory Data Analysis
Analyzes the dataset and produces initial insights:
- **Label distribution** — count of records per traffic type (BENIGN, DDoS, PortScan, etc.)
- **Basic statistics** — mean, std, min, and max per feature
- **Correlation heatmap** — which features are related to each other
- **Feature distributions** — top 6 features that best distinguish benign traffic from attacks

Plots are saved to `output/eda/`.

### `download_data.py`
Downloads the dataset and saves it to `data/raw/`.

## Running
```bash
python main.py
```
