import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px

# --- 1. PAGE CONFIG & MODERN THEME ---
st.set_page_config(page_title="GigAI | Financial Intelligence", layout="wide", initial_sidebar_state="expanded")

# Inject Custom CSS for a professional FinTech aesthetic
st.markdown("""
    <style>
    /* Glassmorphism effect for metrics */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        padding: 20px !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    /* Modern Gradient Sidebar */
    [data-testid="stSidebar"] {
        background-image: linear-gradient(#1e3c72, #2a5298) !important;
        color: white !important;
    }
    /* Title styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    </style>
    """, unsafe_allow_html=True) # FIXED: Using correct parameter for HTML injection

# --- 2. LOAD ML MODELS (4-Model Configuration) ---
@st.cache_resource 
def load_models():
    # Only loading the 4 files present in your GitHub /models folder
    scaler = joblib.load('models/gig_scaler.pkl')
    kmeans = joblib.load('models/gig_kmeans_model.pkl')
    vec = joblib.load('models/gig_vectorizer.pkl')
    nlp = joblib.load('models/gig_nlp_model.pkl')
    return scaler, kmeans, vec, nlp

try:
    # Unpack exactly 4 variables to prevent TypeError
    scaler, kmeans, nlp_vectorizer, nlp_model = load_models()
except Exception as e:
    st.error(f"⚠️ Error loading models from /models: {e}")

# --- 3. SESSION STATE ---
if 'income' not in st.session_state: 
    st.session_state.income = 0.0
if 'expenses' not in st.session_state: 
    st.session_state.expenses = []

# --- 4. SIDEBAR (User Controls) ---
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Income Input
    new_income = st.number_input("Set Monthly Income ($)", min_value=0.0, value=st.session_state.income)
    if st.button("Update Portfolio", use_container_width=True):
        st.session_state.income = new_income
        st.toast("Financial profile updated!")

    st.markdown("---")
    st.subheader("📝 Log New Expense")
    exp_desc = st.text_input("Description", placeholder="e.g. Fuel for Bike")
    exp_amt = st.number_input("Amount ($)", min_value=0.0, step=1.0)

    if st.button("Analyze with AI", use_container_width=True, type="primary"):
        if exp_desc and exp_amt > 0:
            # NLP Model Inference
            vec_text = nlp_vectorizer.transform([exp_desc.lower()])
            category = nlp_model.predict(vec_text)[0]
            
            st.session_state.expenses.append({
                "Date": datetime.now().strftime("%d %b"),
                "Description": exp_desc,
                "Amount": exp_amt,
                "Category": category
            })
            st.balloons()
        else:
            st.warning("Please provide both description and amount.")

# --- 5. MAIN DASHBOARD UI ---
st.markdown('<p class="main-title">GigAI Intelligence</p>', unsafe_allow_html=True)
st.markdown("#### Real-time Clustering & Expenditure Analytics")
st.markdown("---")

total_spent = sum(item['Amount'] for item in st.session_state.expenses)
balance = st.session_state.income - total_spent

# Metrics Row: 4 metrics for a balanced look
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Income", f"${st.session_state.income:,.2f}")
with col2:
    st.metric("Total Expenses", f"${total_spent:,.2f}")
with col3:
    # Wallet Balance with dynamic percentage
    rem_pct = (balance / st.session_state.income * 100) if st.session_state.income > 0 else 0
    st.metric("Wallet Balance", f"${balance:,.2f}", delta=f"{rem_pct:.1f}% Rem.")

with col4:
    # K-Means Clustering Inference
    if st.session_state.income > 0:
        debt_ratio = total_spent / st.session_state.income
        features = scaler.transform([[st.session_state.income, total_spent, debt_ratio]])
        cluster = kmeans.predict(features)[0]
        
        # Color-coded AI Status
        tiers = {
            0: ("Steady Saver", "🟢"), 
            1: ("Premium Spender", "🟡"), 
            2: ("Budget Conscious", "🟠"), 
            3: ("High-Risk", "🔴")
        }
        label, icon = tiers.get(cluster, ("Unknown", "⚪"))
        st.metric("AI Profile", f"{icon} {label}")
    else:
        st.metric("AI Profile", "⚪ Set Income")

st.markdown("---")

# Visuals Section: Left for Pie Chart, Right for Ledger Table
c1, c2 = st.columns([6, 4])

if st.session_state.expenses:
    df = pd.DataFrame(st.session_state.expenses)
    with c1:
        st.subheader("📊 Category Distribution")
        fig = px.pie(df, values='Amount', names='Category', hole=0.5, 
                     color_discrete_sequence=px.colors.sequential.Ice)
        fig.update_layout(template="plotly_dark", margin=dict(t=20, b=20, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.subheader("📜 Recent Activity")
        st.dataframe(df.iloc[::-1], use_container_width=True, hide_index=True)
else:
    st.info("👋 Welcome! Use the sidebar to log an expense and generate your first AI profile.")
