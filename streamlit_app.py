import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(page_title="GigAI | Intelligence", layout="wide")

# Custom CSS for a high-end "Platinum" look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(0, 210, 255, 0.2);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .login-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 40px;
        border-radius: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(45deg, #00d2ff, #3a7bd5);
        border: none;
        color: white;
        font-weight: bold;
    }
    .main-title {
        font-size: 4rem;
        font-weight: 900;
        background: -webkit-linear-gradient(#00d2ff, #9d50bb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD ML MODELS (4-Model Configuration) ---
@st.cache_resource 
def load_models():
    scaler = joblib.load('models/gig_scaler.pkl')
    kmeans = joblib.load('models/gig_kmeans_model.pkl')
    vec = joblib.load('models/gig_vectorizer.pkl')
    nlp = joblib.load('models/gig_nlp_model.pkl')
    return scaler, kmeans, vec, nlp

try:
    scaler, kmeans, nlp_vectorizer, nlp_model = load_models()
except Exception as e:
    st.error(f"⚠️ Model error: {e}")

# --- 3. SESSION STATE FOR AUTH & DATA ---
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'income' not in st.session_state: st.session_state.income = 0.0
if 'expenses' not in st.session_state: st.session_state.expenses = []

# --- 4. LOGIN PAGE ---
if not st.session_state.logged_in:
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<p class="main-title">GigAI</p>', unsafe_allow_html=True)
        st.subheader("Welcome to the Future of Gig Finance")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            user = st.text_input("Username", key="login_user")
            pw = st.text_input("Password", type="password", key="login_pw")
            if st.button("Access Dashboard"):
                if user and pw: # Simple auth for local/Streamlit demo
                    st.session_state.logged_in = True
                    st.rerun()
        
        with tab2:
            st.text_input("Choose Username")
            st.text_input("Choose Password", type="password")
            st.button("Create Account")
        st.markdown('</div>', unsafe_allow_html=True)

# --- 5. MAIN DASHBOARD ---
else:
    with st.sidebar:
        st.markdown("### 👋 Hello, User!")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
        
        st.markdown("---")
        st.header("⚙️ Controls")
        new_income = st.number_input("Set Monthly Income ($)", min_value=0.0, value=st.session_state.income)
        if st.button("Sync Income"):
            st.session_state.income = new_income
            st.toast("Profile Synced!")

        st.markdown("---")
        st.subheader("💸 Log Expense")
        desc = st.text_input("Description", placeholder="e.g. Uber Ride")
        amt = st.number_input("Amount ($)", min_value=0.0)
        
        if st.button("AI Analyze", type="primary"):
            if desc and amt > 0:
                vec = nlp_vectorizer.transform([desc.lower()])
                cat = nlp_model.predict(vec)[0]
                st.session_state.expenses.append({"Date": datetime.now().strftime("%d %b"), "Description": desc, "Amount": amt, "Category": cat})
                st.balloons()

    # Dashboard UI
    st.markdown('<p class="main-title">GigAI Intelligence</p>', unsafe_allow_html=True)
    st.markdown("---")

    total_spent = sum(item['Amount'] for item in st.session_state.expenses)
    balance = st.session_state.income - total_spent

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Income", f"${st.session_state.income:,.2f}")
    with m2: st.metric("Expenses", f"${total_spent:,.2f}")
    with m3: st.metric("Balance", f"${balance:,.2f}")
    with m4:
        if st.session_state.income > 0:
            dr = total_spent / st.session_state.income
            feats = scaler.transform([[st.session_state.income, total_spent, dr]])
            cluster = kmeans.predict(feats)[0]
            tiers = {0: ("Steady Saver", "🟢"), 1: ("Premium Spender", "🟡"), 2: ("Budget Conscious", "🟠"), 3: ("High-Risk", "🔴")}
            label, icon = tiers.get(cluster, ("Unknown", "⚪"))
            st.metric("AI Profile", f"{icon} {label}")
        else:
            st.metric("AI Profile", "⚪ Pending")

    st.markdown("---")
    
    # Visuals
    v1, v2 = st.columns([6, 4])
    if st.session_state.expenses:
        df = pd.DataFrame(st.session_state.expenses)
        with v1:
            fig = px.pie(df, values='Amount', names='Category', hole=0.7, color_discrete_sequence=px.colors.sequential.Electric)
            fig.update_layout(template="plotly_dark", margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
        with v2:
            st.dataframe(df.iloc[::-1], use_container_width=True, hide_index=True)
