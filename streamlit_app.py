import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(page_title="GigAI | Financial Intelligence", layout="wide", initial_sidebar_state="expanded")

# Inject Custom CSS for a modern "FinTech" look
st.markdown("""
    <style>
    .stMetric {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }
    .stMetric:hover {
        transform: translateY(-5px);
        background-color: rgba(255, 255, 255, 0.1);
    }
    [data-testid="stSidebar"] {
        background-image: linear-gradient(#2e3192, #1bffff);
        color: white;
    }
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#1bffff, #2e3192);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    """, unsafe_allow_exists=True)

# --- 2. LOAD ML MODELS ---
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
    st.error(f"⚠️ Error loading models: {e}")

# --- 3. SESSION STATE ---
if 'income' not in st.session_state: st.session_state.income = 0.0
if 'expenses' not in st.session_state: st.session_state.expenses = []

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.title("Settings")
    
    new_income = st.number_input("💵 Set Monthly Income ($)", min_value=0.0, value=st.session_state.income)
    if st.button("Update Income", use_container_width=True):
        st.session_state.income = new_income
        st.toast("Income Updated!")

    st.markdown("---")
    st.subheader("📝 Quick Log")
    exp_desc = st.text_input("Description", placeholder="e.g. Petrol for bike")
    exp_amt = st.number_input("Amount ($)", min_value=0.0, step=1.0)

    if st.button("Analyze & Track", use_container_width=True, type="primary"):
        if exp_desc and exp_amt > 0:
            vec_text = nlp_vectorizer.transform([exp_desc.lower()])
            category = nlp_model.predict(vec_text)[0]
            st.session_state.expenses.append({
                "Date": datetime.now().strftime("%d %b"),
                "Description": exp_desc,
                "Amount": exp_amt,
                "Category": category
            })
            st.balloons()

# --- 5. MAIN DASHBOARD ---
st.markdown('<p class="main-title">GigAI Intelligence</p>', unsafe_allow_html=True)
st.markdown("#### Real-time Financial Clustering & NLP Tracking")
st.markdown("---")

total_spent = sum(item['Amount'] for item in st.session_state.expenses)
balance = st.session_state.income - total_spent

# Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Income", f"${st.session_state.income:,.2f}")
with col2:
    st.metric("Total Expenses", f"${total_spent:,.2f}", delta=f"-${total_spent:,.2f}", delta_color="inverse")
with col3:
    st.metric("Wallet Balance", f"${balance:,.2f}", delta=f"{(balance/st.session_state.income*100 if st.session_state.income > 0 else 0):.1f}% Rem.")

with col4:
    if st.session_state.income > 0:
        debt_ratio = total_spent / st.session_state.income
        features = scaler.transform([[st.session_state.income, total_spent, debt_ratio]])
        cluster = kmeans.predict(features)[0]
        
        # Color-coded AI labels
        tiers = {
            0: ("Steady Saver", "🟢"), 
            1: ("Premium Spender", "🟡"), 
            2: ("Budget Conscious", "🟠"), 
            3: ("High-Risk", "🔴")
        }
        label, icon = tiers.get(cluster, ("Unknown", "⚪"))
        st.metric("AI Status", f"{icon} {label}")
    else:
        st.metric("AI Status", "⚪ Set Income")

st.markdown("---")

# Visuals Section
c1, c2 = st.columns([6, 4])

if st.session_state.expenses:
    df = pd.DataFrame(st.session_state.expenses)
    with c1:
        st.subheader("📈 Spending Breakdown")
        fig = px.pie(df, values='Amount', names='Category', hole=0.6, 
                     color_discrete_sequence=px.colors.sequential.Tealgrn)
        fig.update_layout(template="plotly_dark", margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.subheader("📄 Transaction Ledger")
        st.dataframe(df.iloc[::-1], use_container_width=True, hide_index=True)
else:
    st.info("💡 Start by logging an expense in the sidebar to see AI insights!")
