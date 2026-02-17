# 🚀 GigAI: Financial Intelligence for Gig Workers

**GigAI** is an end-to-end Machine Learning application designed to help gig economy workers (like delivery partners, freelancers, and riders) track their expenses, categorize spending using NLP, and receive behavioral financial insights via K-Means clustering.

---

## 🧠 Machine Learning Features
- **NLP Categorization:** Uses a **Naïve Bayes** model to automatically categorize expenses based on text descriptions (e.g., "Fuel for CB350RS" ➔ Transport).
- **Behavioral Clustering:** Implements **K-Means Clustering** to segment users into financial profiles (Steady Savers, High-Risk, etc.) based on Income-to-Expense ratios.
- **Model Serialization:** Models are trained in Jupyter Notebooks and serialized using **Joblib** for high-performance production inference.

## 🛠️ Tech Stack
- **Languages:** Python (Core Logic), JavaScript (Frontend)
- **ML Libraries:** Scikit-Learn, NumPy, Pandas
- **APIs & Backend:** Flask, JWT (JSON Web Tokens) for Secure Auth
- **Frontend:** Streamlit (Deployment) & Tailwind CSS (Custom Dashboard)
- **Visuals:** Chart.js & Plotly

---

## 📁 Project Structure
- `streamlit_app.py`: The live, interactive dashboard deployed for users.
- `app.py`: A RESTful Flask API version featuring JWT Authentication.
- `training.ipynb`: The research notebook where data is generated and models are trained.
- `models/`: Contains serialized `.pkl` files (Scaler, K-Means, NLP, Vectorizer).

---

## 🚀 How to Run

### Option 1: Live Streamlit App (Recommended)
1. Install requirements: `pip install -r requirements.txt`
2. Run the app: `streamlit run streamlit_app.py`

### Option 2: Flask API & HTML Dashboard
1. Run the backend: `python app.py`
2. Open `index.html` in any modern browser.

---

## 📈 Future Roadmap
- [ ] **Anomaly Detection:** Implement Isolation Forest to flag unusual spending.
- [ ] **Forecasting:** Use Linear Regression to predict next month's burn rate.
- [ ] **Persistence:** Migrate from In-Memory dictionaries to MongoDB/PostgreSQL.


## DBSCAN Outlier Detection: 
Implemented DBSCAN to identify irregular spending "noise" that doesn't fit standard profiles, helping users spot one-time financial shocks or anomalies that K-Means might misclassify.
---
**Developed by Ayan** *Second-year IT Student | Machine Learning Enthusiast*