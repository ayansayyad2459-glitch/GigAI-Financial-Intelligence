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
    .stApp { background: #0E1117 !important; }
    
    /* Neon Metric Cards with Glassmorphism */
    [data-testid="stMetric"] {
        background: rgba(17, 25, 40, 0.7) !important;
        backdrop-filter: blur(12px) !important;
        border-radius: 20px !important;
        border: 1px solid rgba(0, 210, 255, 0.4) !important;
        padding: 25px !important;
        box-shadow: 0 10px 30px rgba(0, 210, 255, 0.15) !important;
    }

    [data-testid="stMetricLabel"] p {
        color: #00d2ff !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
    }
    
    [data-testid="stMetricValue"] div {
        color: #ffffff !important;
        font-family: 'Monaco', monospace !important;
        text-shadow: 0 0 10px rgba(0, 210, 255, 0.5) !important;
    }

    .login-container {
        background: rgba(255, 255, 255, 0.03);
        padding: 50px;
        border-radius: 30px;
        border: 1px solid rgba(0, 210, 255, 0.2);
        text-align: center;
    }

    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #9d50bb 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 900 !important;
        border-radius: 12px !important;
    }

    .neon-title {
        font-size: 4.5rem !important;
        font-weight: 900 !important;
        background: linear-gradient(to right, #00d2ff, #9d50bb, #ff007c) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD ML MODELS ---
@st.cache_resource 
def load_models():
    # Loading 4 models as per current GitHub structure
    scaler = joblib.load('models/gig_scaler.pkl')
    kmeans = joblib.load('models/gig_kmeans_model.pkl')
    vec = joblib.load('models/gig_vectorizer.pkl')
    nlp = joblib.load('models/gig_nlp_model.pkl')
    return scaler, kmeans, vec, nlp

try:
    scaler, kmeans, nlp_vectorizer, nlp_model = load_models()
except Exception as e:
    st.error(f"⚠️ Model Error: {e}")

# --- 3. SESSION STATE ---
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'username' not in st.session_state: st.session_state.username = ""
if 'income' not in st.session_state: st.session_state.income = 0.0
if 'expenses' not in st.session_state: st.session_state.expenses = []
if 'user_db' not in st.session_state: st.session_state.user_db = {"Ayan": "password123"} 

# --- 4. AUTHENTICATION PAGE ---
if not st.session_state.logged_in:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<h1 class="neon-title">GigAI</h1>', unsafe_allow_html=True)
        st.write("### Secure Financial Intelligence")
        
        tab_login, tab_reg = st.tabs(["🔐 Login", "📝 Register"])
        with tab_login:
            u_input = st.text_input("Username", placeholder="e.g. Ayan")
            p_input = st.text_input("Password", type="password")
            if st.button("Access Portfolio", use_container_width=True):
                if u_input in st.session_state.user_db and st.session_state.user_db[u_input] == p_input:
                    st.session_state.logged_in = True
                    st.session_state.username = u_input
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        with tab_reg:
            reg_full_name = st.text_input("Full Name", placeholder="Ayan Sayyad")
            reg_user = st.text_input("Choose Username")
            reg_pass = st.text_input("Choose Password", type="password")
            if st.button("Create Account", use_container_width=True):
                if reg_user and reg_pass:
                    st.session_state.user_db[reg_user] = reg_pass
                    st.success(f"Account created for {reg_full_name}! Please login.")
                    st.balloons()
        st.markdown('</div>', unsafe_allow_html=True)

# --- 5. MAIN DASHBOARD ---
else:
    with st.sidebar:
        # Displays the logged-in username dynamically
        st.markdown(f"### 👋 Welcome, {st.session_state.username}!")
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()
        
        st.markdown("---")
        st.header("⚙️ Wallet")
        new_inc = st.number_input("Monthly Income ($)", min_value=0.0, value=st.session_state.income)
        if st.button("Sync Data", use_container_width=True):
            st.session_state.income = new_inc
            st.toast("Intelligence Updated!")

        st.markdown("---")
        st.subheader("💸 Log Expense")
        desc = st.text_input("Description", placeholder="e.g. AWS Credits")
        amt = st.number_input("Amount ($)", min_value=0.0)
        
        if st.button("Analyze with AI", type="primary", use_container_width=True):
            if desc and amt > 0:
                vec = nlp_vectorizer.transform([desc.lower()])
                cat = nlp_model.predict(vec)[0]
                st.session_state.expenses.append({
                    "Date": datetime.now().strftime("%d %b"), 
                    "Description": desc, 
                    "Amount": float(amt), 
                    "Category": cat
                })
                st.balloons()

    st.markdown('<h1 class="neon-title">GigAI Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")

    total_spent = sum(item['Amount'] for item in st.session_state.expenses)
    balance = st.session_state.income - total_spent

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Income", f"${st.session_state.income:,.2f}")
    with m2: st.metric("Expenses", f"${total_spent:,.2f}")
    with m3: st.metric("Wallet", f"${balance:,.2f}")
    with m4:
        if st.session_state.income > 0:
            dr = total_spent / st.session_state.income
            feats = scaler.transform([[st.session_state.income, total_spent, dr]])
            cluster = kmeans.predict(feats)[0]
            tiers = {0: ("Steady Saver", "🟢"), 1: ("Premium Spender", "🟡"), 2: ("Budget Conscious", "🟠"), 3: ("High-Risk", "🔴")}
            label, icon = tiers.get(cluster, ("Unknown", "⚪"))
            st.metric("AI Profile", f"{icon} {label}")
        else:
            st.metric("AI Profile", "⚪ NO DATA")

    st.markdown("---")
    
    # Visualization Row
    v1, v2 = st.columns([6, 4])
    if st.session_state.expenses:
        df = pd.DataFrame(st.session_state.expenses)
        df['Amount'] = df['Amount'].astype(float)
        
        with v1:
            st.subheader("📈 Sector Analysis")
            # Donut chart optimized to fill the space
            fig = px.pie(
                df, values='Amount', names='Category', hole=0.5, 
                color_discrete_sequence=px.colors.sequential.Electric
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                template="plotly_dark", 
                margin=dict(t=20, b=20, l=0, r=0),
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        with v2:
            st.subheader("📜 Recent Activity")
            st.dataframe(df.iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("👋 Add expenses to generate your AI sector analysis.")
