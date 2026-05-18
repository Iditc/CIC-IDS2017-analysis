# CIC-IDS2017 Anomaly Detection Project

## Goal
Build an anomaly detection system for network intrusion detection,
relevant to cybersecurity products like Palo Alto Networks.

## Tech Stack
- Python 3.10+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- tensorflow/keras (for Autoencoder, LSTM)

## Project Structure
- src/data/raw/ - Raw dataset files (NOT committed to git)
- src/data/processed/ - Cleaned and processed data (parquet format)
- src/ - Source code and scripts
- results/figures/ - Plots and visualizations
- results/metrics/ - Saved model performance

## Dataset
- Name: CIC-IDS2017
- Source: Canadian Institute for Cybersecurity / Hugging Face
- Type: Network traffic with labeled attacks
- Known issues: infinite values and NaNs in some columns — always handle in preprocessing
- Class distribution: heavily imbalanced (benign >> attacks)

## Algorithms to Explore
- Isolation Forest
- Autoencoder
- One-Class SVM
- LSTM-based detection

## Goals for Learning
- Practice multiple anomaly detection algorithms
- Compare performance (precision, recall, F1)
- Understand tradeoffs between False Positives and False Negatives

## Key Design Decisions

- Primary metric: F1 macro (NOT accuracy — data is imbalanced)
- Evaluate tradeoffs between False Positives and False Negatives explicitly
- Each source file < 300 lines
- Use pathlib for all file paths

## Coding Conventions

- All code comments, docstrings, and documentation must be written in English
- Small, single-purpose functions with docstrings
- Do not reload raw CSVs if processed/ already exists
- Save processed data as parquet for speed

## AI Workflow Strategy (Claude Code)
- Opus: use for high-level decisions — algorithm selection, architecture, 
  debugging complex issues, reviewing results
- Sonnet: use for execution — writing code, refactoring, implementing features
- Rule: start every task with Sonnet. Switch to Opus only when stuck 
  or making a design decision that affects the whole project.
  