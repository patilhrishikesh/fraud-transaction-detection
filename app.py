import streamlit as st
import pandas as pd
import joblib

# Set up dashboard page
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Load dataset once and cache it (fast reloads)
@st.cache_data
def load_data():
    return pd.read_csv("data/transactions.csv")

# Load trained Random Forest model once and cache it
@st.cache_resource
def load_model():
    return joblib.load("rf_model.pkl")

# Get data and model ready
df = load_data()
model = load_model()

# Dashboard title
st.title("💳 Fraud Transaction Detection Dashboard")

# --- KPIs ---
# Calculate total, fraud, and legit transactions
total_tx = len(df)
fraud_tx = df["Class"].sum()   # frauds are labeled as 1
legit_tx = total_tx - fraud_tx

# Show KPIs side by side
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", total_tx)
col2.metric("Fraudulent Transactions", fraud_tx)
col3.metric("Legitimate Transactions", legit_tx)

# --- Fraud distribution chart ---
st.subheader("Fraud Distribution")
st.bar_chart(df["Class"].value_counts())

# --- Predict on recent transactions ---
st.subheader("Flagged Transactions (sample)")
sample = df.sample(50, random_state=42)  # take 50 random transactions

# Drop target column (Class) to keep only features
X_sample = sample.drop("Class", axis=1)

# Predict fraud probability for each transaction
sample["Fraud Probability"] = model.predict_proba(X_sample)[:, 1]

# Sort by highest fraud probability
sample = sample.sort_values("Fraud Probability", ascending=False)

# Show top 10 suspicious transactions
st.dataframe(sample.head(10))
