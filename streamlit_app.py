import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="GigAI Intelligence", layout="wide")

# --- 2. LOAD ML MODELS ---
@st.cache_resource 
def load_models():
    # Loading all 5 models from the models/ folder
    scaler = joblib.load('models/gig_scaler.pkl')
    kmeans = joblib.load('models/gig_kmeans_model.pkl')
    dbscan = joblib.load('models/gig_dbscan_model.pkl') # Added DBSCAN
    vec = joblib.load('models/gig_vectorizer.pkl')
    nlp = joblib.load('models/gig_nlp_model.pkl')
    return scaler, kmeans, dbscan, vec, nlp

try:
    scaler, kmeans, dbscan, nlp_vectorizer, nlp_model = load_models()
except Exception as e:
    st.error(f"Missing .pkl files in /models folder! Error: {e}")

# --- 3. SESSION STATE ---
if 'income' not in st.session_state:
    st.session_state.income = 0.0
if 'expenses' not in st.session_state:
    st.session_state.expenses = []

# --- 4. SIDEBAR ---
st.sidebar.title("📊 GigAI Control Panel")

# Set Income
new_income = st.sidebar.number_input("Set Monthly Income ($)", min_value=0.0, value=st.session_state.income)
if st.sidebar.button("Update Income"):
    st.session_state.income = new_income
    st.success("Income Updated!")

st.sidebar.markdown("---")

# Add Expense
st.sidebar.subheader("📝 Log New Expense")
exp_desc = st.sidebar.text_input("Description", placeholder="e.g. Petrol for bike")
exp_amt = st.sidebar.number_input("Amount ($)", min_value=0.0, step=1.0)

if st.sidebar.button("Analyze & Track"):
    if exp_desc and exp_amt > 0:
        # NLP Prediction
        vec_text = nlp_vectorizer.transform([exp_desc.lower()])
        category = nlp_model.predict(vec_text)[0]
        
        st.session_state.expenses.append({
            "Date": datetime.now().strftime("%d %b"),
            "Description": exp_desc,
            "Amount": exp_amt,
            "Category": category
        })
        st.sidebar.balloons()
    else:
        st.sidebar.warning("Please fill details")

# --- 5. MAIN DASHBOARD (The Matrix Row) ---
st.title("Financial Intelligence Dashboard")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4) # Added 4th column for DBSCAN
total_spent = sum(item['Amount'] for item in st.session_state.expenses)

with col1:
    st.metric("Total Earned", f"${st.session_state.income:,.2f}")
with col2:
    st.metric("Total Spent", f"${total_spent:,.2f}")

with col3:
    # K-Means Financial Profile
    if st.session_state.income > 0:
        debt_ratio = total_spent / st.session_state.income
        features = scaler.transform([[st.session_state.income, total_spent, debt_ratio]])
        cluster = kmeans.predict(features)[0]
        
        tiers = {0: "Steady Saver", 1: "Premium Spender", 2: "Budget Conscious", 3: "High-Risk"}
        st.metric("AI Profile", tiers.get(cluster, "Unknown"))
    else:
        st.metric("AI Profile", "Set Income")

with col4:
    # DBSCAN Anomaly Detection
    if st.session_state.income > 0:
        # Note: DBSCAN fit_predict returns -1 for outliers
        is_anomaly = dbscan.fit_predict(features)[0]
        status = "⚠️ Anomaly" if is_anomaly == -1 else "✅ Normal"
        st.metric("Pattern Status", status)
    else:
        st.metric("Pattern Status", "N/A")

st.markdown("---")

# Visuals
c1, c2 = st.columns([1, 1])
if st.session_state.expenses:
    df = pd.DataFrame(st.session_state.expenses)
    with c1:
        fig = px.pie(df, values='Amount', names='Category', hole=0.5, title="Spending by Category")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Recent Ledger")
        st.dataframe(df.iloc[::-1], use_container_width=True, hide_index=True)