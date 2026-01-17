# 💳 Fraud Transaction Detection System

An end-to-end machine learning system to detect fraudulent financial transactions using supervised and unsupervised learning techniques.

---

## 📌 Problem Statement
Banks and financial institutions lose millions due to fraudulent transactions.  
The challenge lies in detecting fraud accurately from highly imbalanced data while maintaining low latency for real-time decision-making.

---

## 🚀 Solution Overview
This project implements a fraud detection pipeline that:
- Handles severe class imbalance
- Classifies transactions in near real time (<120 ms)
- Detects both known and unknown fraud patterns
- Provides a dashboard for monitoring and testing transactions

---

## Features
- Handles highly imbalanced data
- Random Forest & Isolation Forest models
- Real-time fraud classification
- Streamlit dashboard for monitoring

## Tech Stack
Python, Pandas, Scikit-learn, Streamlit, uv

## 🧠 Machine Learning Approach

### Models Used
- **Random Forest** (Supervised classification)
- **Isolation Forest** (Unsupervised anomaly detection)

### Why These Models?
- Random Forest handles non-linear patterns and supports class weighting
- Isolation Forest detects anomalies without labeled data

---

## 📊 Dataset
- **Source:** Credit Card Fraud Detection Dataset (Kaggle)
- **Records:** 284,807 transactions
- **Fraud Ratio:** ~0.17%

Due to GitHub file size limits, the dataset is not included.

### Download Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud



