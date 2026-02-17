import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px

# --- 1. PAGE CONFIG & NEON THEME ---
st.set_page_config(page_title="GigAI | Financial Intelligence", layout="wide")

st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: #0E1117 !important;
    }
    
    /* Neon Metric Cards with Glassmorphism */
    [data-testid="stMetric"] {
        background: rgba(17, 25, 40, 0.7) !important;
        backdrop-filter: blur(12px) !important;
        border-radius: 20px !important;
        border: 1px solid rgba(0, 210, 255, 0.4) !important;
        padding: 25px !important;
        box-shadow: 0 10px 30px rgba(0, 210, 255, 0.15) !important;
    }

    /* Target Metric Labels and Values for Neon Colors */
    [data-testid="stMetricLabel"] p {
        color: #00d2ff !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
    }
    
    [data-testid="stMetricValue"] div {
        color: #ffffff !important;
        font-family: 'Monaco', monospace !important;
        text-shadow: 0 0 10px rgba(0, 210, 255, 0.5) !important;
    }

    /* Custom Login Card */
    .login-container {
        background: rgba(255, 255, 255, 0.03);
        padding: 50px;
        border-radius: 30px;
        border: 1px solid rgba(0, 210, 255, 0.2);
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
    }

    /* Vibrant Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #9d50bb 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 900 !important;
        border-radius: 12px !important;
        transition: 0.4s all ease !important;
    }
    
    .stButton>button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 0 20px rgba(0, 210, 255, 0.6) !important;
    }

    /* Gradient Title */
    .neon-title {
        font-size: 4.5rem !important;
        font-weight: 900 !important;
        background: linear-gradient(to right, #00d2ff, #9d50bb, #ff007c) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 0px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD ML MODELS (4-Model Configuration) ---
@st.cache_resource 
def load_models():
    # Only loading the 4 .pkl files you have in your /models folder
    scaler = joblib.load('models/gig_scaler.pkl')
    kmeans = joblib.load('models/gig_kmeans_model.pkl')
    vec = joblib.load('models/gig_vectorizer.pkl')
    nlp = joblib.load('models/gig_nlp_model.pkl')
    return scaler, kmeans, vec, nlp

try:
    scaler, kmeans, nlp_vectorizer, nlp_model = load_models()
except Exception as e:
    st.error(f"⚠️ Model Loading Error: {e}")

# --- 3. SESSION STATE FOR AUTH & DATA ---
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'income' not in st.session_state: st.session_state.income = 0.0
if 'expenses' not in st.session_state: st.session_state.expenses = []

# --- 4. LOGIN PAGE UI ---
if not st.session_state.logged_in:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<h1 class="neon-title">GigAI</h1>', unsafe_allow_html=True)
        st.write("### Financial Intelligence for the Gig Economy")
        
        tab_login, tab_reg = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab_login:
            user_input = st.text_input("Username", placeholder="ayansayyad")
            pass_input = st.text_input("Password", type="password")
            if st.button("Unlocking Intelligence", use_container_width=True):
                # Basic authentication toggle
                st.session_state.logged_in = True
                st.rerun()
        
        with tab_reg:
            st.text_input("Full Name")
            st.text_input("New Username")
            st.text_input("New Password", type="password")
            st.button("Create GigAI Account", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- 5. MAIN DASHBOARD UI ---
else:
    with st.sidebar:
        st.markdown("### 👤 User Profile")
        st.info(f"Connected as: Admin")
        if st.button("Secure Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()
        
        st.markdown("---")
        st.header("⚙️ Wallet Config")
        new_inc = st.number_input("Monthly Revenue ($)", min_value=0.0, value=st.session_state.income)
        if st.button("Sync with AI", use_container_width=True):
            st.session_state.income = new_inc
            st.toast("Intelligence Synced!")

        st.markdown("---")
        st.subheader("💸 Log Transaction")
        desc = st.text_input("Description", placeholder="e.g. Server Hosting")
        amt = st.number_input("Amount ($)", min_value=0.0)
        
        if st.button("Run AI Analysis", type="primary", use_container_width=True):
            if desc and amt > 0:
                # NLP Inference
                vec = nlp_vectorizer.transform([desc.lower()])
                cat = nlp_model.predict(vec)[0]
                st.session_state.expenses.append({
                    "Date": datetime.now().strftime("%d %b"),
                    "Description": desc,
                    "Amount": amt,
                    "Category": cat
                })
                st.balloons()

    # Top Section
    st.markdown('<h1 class="neon-title">GigAI Intelligence</h1>', unsafe_allow_html=True)
    st.markdown("#### Advanced K-Means Clustering & NLP Classification Dashboard")
    st.markdown("---")

    total_spent = sum(item['Amount'] for item in st.session_state.expenses)
    balance = st.session_state.income - total_spent

    # Vibrant Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Income", f"${st.session_state.income:,.2f}")
    with m2: st.metric("Expenses", f"${total_spent:,.2f}")
    with m3: st.metric("Wallet", f"${balance:,.2f}")
    with m4:
        if st.session_state.income > 0:
            # K-Means Profile Logic
            dr = total_spent / st.session_state.income
            feats = scaler.transform([[st.session_state.income, total_spent, dr]])
            cluster = kmeans.predict(feats)[0]
            
            tiers = {
                0: ("Steady Saver", "🟢"), 
                1: ("Premium Spender", "🟡"), 
                2: ("Budget Conscious", "🟠"), 
                3: ("High-Risk", "🔴")
            }
            label, icon = tiers.get(cluster, ("Unknown", "⚪"))
            st.metric("AI Profile", f"{icon} {label}")
        else:
            st.metric("AI Profile", "⚪ NO DATA")

    st.markdown("---")
    
    # Visualization Row
    v1, v2 = st.columns([6, 4])
    if st.session_state.expenses:
        df = pd.DataFrame(st.session_state.expenses)
        with v1:
            st.subheader("📈 Real-time Sector Analysis")
            fig = px.pie(df, values='Amount', names='Category', hole=0.7, 
                         color_discrete_sequence=px.colors.sequential.Electric)
            fig.update_layout(template="plotly_dark", margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
        with v2:
            st.subheader("📜 Transaction Ledger")
            st.dataframe(df.iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ No data detected. Please log an expense in the sidebar to activate AI clustering.")
