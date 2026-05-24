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

### `correlation_filter.py` — Redundant Feature Removal
Identifies and removes highly correlated features (threshold: |r| > 0.9).

Algorithm:
- Computes the full correlation matrix for all numeric features
- Iterates over feature pairs — if two features are highly correlated, drops the second one
- Once a feature is marked for dropping, it is skipped in all further comparisons

Output:
- Prints dropped features and their correlation partners to the terminal
- Saves `output/features_to_keep.csv` — the final list of features to use
- Saves `output/dropped_features_heatmap.png` — heatmap showing dropped features vs their correlated partners

### `download_data.py`
Downloads the dataset and saves it to `data/raw/`.

## Running
```bash
python main.py
```

## Features

| Feature | Description |
|---|---|
| Destination Port | Port number of the destination (e.g. 80=HTTP, 443=HTTPS, 22=SSH) |
| Flow Duration | Total duration of the flow in microseconds |
| Total Fwd Packets | Total number of packets sent in the forward direction (client → server) |
| Total Backward Packets | Total number of packets sent in the backward direction (server → client) |
| Total Length of Fwd Packets | Total bytes of payload in forward packets |
| Total Length of Bwd Packets | Total bytes of payload in backward packets |
| Fwd Packet Length Max | Largest forward packet size (bytes) |
| Fwd Packet Length Min | Smallest forward packet size (bytes) |
| Fwd Packet Length Mean | Average forward packet size (bytes) |
| Fwd Packet Length Std | Standard deviation of forward packet sizes |
| Bwd Packet Length Max | Largest backward packet size (bytes) |
| Bwd Packet Length Min | Smallest backward packet size (bytes) |
| Bwd Packet Length Mean | Average backward packet size (bytes) |
| Bwd Packet Length Std | Standard deviation of backward packet sizes |
| Flow Bytes/s | Total bytes per second across the entire flow |
| Flow Packets/s | Total packets per second across the entire flow |
| Flow IAT Mean | Mean time between any two consecutive packets in the flow (µs) |
| Flow IAT Std | Standard deviation of inter-arrival times |
| Flow IAT Max | Longest gap between two consecutive packets (µs) |
| Flow IAT Min | Shortest gap between two consecutive packets (µs) |
| Fwd IAT Total | Total time between forward packets (µs) |
| Fwd IAT Mean | Mean inter-arrival time of forward packets (µs) |
| Fwd IAT Std | Standard deviation of forward IAT |
| Fwd IAT Max | Longest gap between forward packets (µs) |
| Fwd IAT Min | Shortest gap between forward packets (µs) |
| Bwd IAT Total | Total time between backward packets (µs) |
| Bwd IAT Mean | Mean inter-arrival time of backward packets (µs) |
| Bwd IAT Std | Standard deviation of backward IAT |
| Bwd IAT Max | Longest gap between backward packets (µs) |
| Bwd IAT Min | Shortest gap between backward packets (µs) |
| Fwd PSH Flags | Number of forward packets with PSH flag (push data immediately to application) |
| Bwd PSH Flags | Number of backward packets with PSH flag |
| Fwd URG Flags | Number of forward packets with URG flag (urgent data) |
| Bwd URG Flags | Number of backward packets with URG flag |
| Fwd Header Length | Total bytes used by headers in forward packets |
| Bwd Header Length | Total bytes used by headers in backward packets |
| Fwd Packets/s | Forward packets per second |
| Bwd Packets/s | Backward packets per second |
| Min Packet Length | Smallest packet in the entire flow (bytes) |
| Max Packet Length | Largest packet in the entire flow (bytes) |
| Packet Length Mean | Average packet size across the entire flow (bytes) |
| Packet Length Std | Standard deviation of packet sizes |
| Packet Length Variance | Variance of packet sizes (Std²) |
| FIN Flag Count | Number of packets with FIN flag (connection teardown) |
| SYN Flag Count | Number of packets with SYN flag — high values may indicate SYN flood |
| RST Flag Count | Number of packets with RST flag (abrupt connection reset) |
| PSH Flag Count | Total PSH flags across all packets |
| ACK Flag Count | Number of packets with ACK flag (acknowledgement) |
| URG Flag Count | Total URG flags across all packets |
| CWE Flag Count | Number of packets with CWE flag (Congestion Window Reduced Echo) |
| ECE Flag Count | Number of packets with ECE flag (Explicit Congestion Notification Echo) |
| Down/Up Ratio | Ratio of download (backward) to upload (forward) traffic |
| Average Packet Size | Mean packet size across both directions (bytes) |
| Avg Fwd Segment Size | Mean forward TCP segment size |
| Avg Bwd Segment Size | Mean backward TCP segment size |
| Fwd Avg Bytes/Bulk | Average bytes per bulk transfer in the forward direction |
| Fwd Avg Packets/Bulk | Average packets per bulk transfer in the forward direction |
| Fwd Avg Bulk Rate | Average bulk transfer rate in the forward direction (bytes/s) |
| Bwd Avg Bytes/Bulk | Average bytes per bulk transfer in the backward direction |
| Bwd Avg Packets/Bulk | Average packets per bulk transfer in the backward direction |
| Bwd Avg Bulk Rate | Average bulk transfer rate in the backward direction (bytes/s) |
| Subflow Fwd Packets | Mean number of forward packets per subflow |
| Subflow Fwd Bytes | Mean bytes of forward packets per subflow |
| Subflow Bwd Packets | Mean number of backward packets per subflow |
| Subflow Bwd Bytes | Mean bytes of backward packets per subflow |
| Init_Win_bytes_forward | Initial TCP window size in the forward direction (bytes) |
| Init_Win_bytes_backward | Initial TCP window size in the backward direction (bytes) |
| act_data_pkt_fwd | Number of forward packets that carried actual payload |
| min_seg_size_forward | Minimum TCP segment size in the forward direction |
| Active Mean | Mean time the flow was active before going idle (µs) |
| Active Std | Standard deviation of active periods |
| Active Max | Longest active period (µs) |
| Active Min | Shortest active period (µs) |
| Idle Mean | Mean time the flow was idle between active periods (µs) |
| Idle Std | Standard deviation of idle periods |
| Idle Max | Longest idle period (µs) |
| Idle Min | Shortest idle period (µs) |
| Label | Traffic classification: BENIGN or attack type (DDoS, PortScan, Brute Force, etc.) |

## Missing Values

Two features contain missing values (0.1% of rows each):

- **Flow Bytes/s** — total bytes per second in the flow. Missing in 2,867 rows (0.1%). Correlation with label: TBD (pending analysis).
- **Flow Packets/s** — total packets per second in the flow. Missing in 2,867 rows (0.1%). Correlation with label: TBD (pending analysis).

Both features affect the same rows (missing values are co-located). Given the very low missing rate (0.1%), these rows will be dropped during preprocessing. Correlation analysis will determine whether these features carry meaningful signal.

## EDA Plots

#### Label Distribution
![Label Distribution](output/eda/label_distribution.png)

#### Correlation Heatmap
![Correlation Heatmap](output/eda/correlation_heatmap.png)

#### Top 6 Discriminative Features
![Top Features Distribution](output/eda/top_features_distribution.png)

#### Dropped Features Heatmap
![Dropped Features Heatmap](output/dropped_features_heatmap.png)
