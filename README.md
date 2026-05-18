# CIC-IDS2017-analysis
This project applies machine learning techniques to network intrusion detection using the CIC-IDS2017 dataset. It covers data preprocessing, feature engineering, model training and evaluation — classifying network traffic as benign or malicious across multiple attack categories including DDoS, PortScan, and Brute Force.

## קבצים

### `load_data.py`
טוען את כל קבצי ה-CSV של הדאטאסט מתיקיית `data/raw/`, מנרמל שמות עמודות, ומוחק שורות עם ערכים חסרים או אינסופיים. מחזיר DataFrame אחד מאוחד עם כל הנתונים.

### `eda.py` — Exploratory Data Analysis
מנתח את הדאטאסט ומפיק תובנות ראשוניות:
- **התפלגות תוויות** — כמה רשומות יש מכל סוג תעבורה (BENIGN, DDoS, PortScan וכו')
- **סטטיסטיקות בסיסיות** — ממוצע, סטיית תקן, מינימום ומקסימום לכל feature
- **מפת קורלציה** — אילו features קשורים זה לזה
- **התפלגות features** — 6 ה-features שמבדילים הכי טוב בין תעבורה תקינה למתקפות

הגרפים נשמרים בתיקיית `output/eda/`.

### `download_data.py`
מוריד את הדאטאסט מהאינטרנט ושומר אותו ב-`data/raw/`.

## הרצה
```bash
python main.py
```
