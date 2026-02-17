from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)

app.config['JWT_SECRET_KEY'] = 'gig-ai-internship-2026'
jwt = JWTManager(app)

# --- 1. LOAD THE ML MODELS ---
try:
    scaler = joblib.load('models/gig_scaler.pkl')
    kmeans_model = joblib.load('models/gig_kmeans_model.pkl')
    # Removed DBSCAN load line here
    nlp_vectorizer = joblib.load('models/gig_vectorizer.pkl')
    nlp_model = joblib.load('models/gig_nlp_model.pkl')
    print("✅ ML Models Loaded Successfully (DBSCAN omitted)")
except Exception as e:
    print(f"❌ ERROR: Model loading failed: {e}")

users_db = {}

tier_data = {
    0: {'name': 'Steady Savers', 'advice': 'Low burn rate. Good for long-term wealth.'},
    1: {'name': 'Premium Spenders', 'advice': 'High outflow. Watch your luxury expenses.'},
    2: {'name': 'Budget Conscious', 'advice': 'Narrow margins. Focus on increasing gig hours.'},
    3: {'name': 'High-Risk', 'advice': 'Debt trap alert! Cut all non-essential spending.'}
}

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    u, p = data.get('username'), data.get('password')
    if not u or not p:
        return jsonify({"message": "Username and Password required"}), 400
    if u in users_db:
        return jsonify({"message": "User already exists"}), 400
    users_db[u] = {'password': generate_password_hash(p), 'income': 0, 'expenses': []}
    return jsonify({"message": "Registered successfully"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    u, p = data.get('username'), data.get('password')
    user = users_db.get(u)
    if user and check_password_hash(user['password'], p):
        access_token = create_access_token(identity=u)
        return jsonify(access_token=access_token), 200
    return jsonify({"message": "Invalid credentials"}), 401

@app.route('/set_income', methods=['POST'])
@jwt_required()
def set_income():
    user = get_jwt_identity()
    if user not in users_db:
        return jsonify({"message": "Session expired."}), 401
    users_db[user]['income'] = float(request.get_json().get('amount'))
    return jsonify({"message": "Income updated"})

@app.route('/add_expense', methods=['POST'])
@jwt_required()
def add_expense():
    user = get_jwt_identity()
    if user not in users_db:
        return jsonify({"message": "Session expired."}), 401

    data = request.get_json()
    desc = data.get('description', '')
    vec = nlp_vectorizer.transform([desc.lower()])
    category = nlp_model.predict(vec)[0]
    
    new_tx = {
        "date": datetime.now().strftime("%d %b"),
        "desc": desc,
        "amt": float(data.get('amount', 0)),
        "cat": category
    }
    users_db[user]['expenses'].append(new_tx)
    return jsonify(new_tx)

@app.route('/get_dashboard', methods=['GET'])
@jwt_required()
def get_dashboard():
    user = get_jwt_identity()
    if user not in users_db:
        return jsonify({"message": "User not found"}), 404
        
    u_data = users_db[user]
    total_inc = u_data['income']
    total_exp = sum(tx['amt'] for tx in u_data['expenses'])
    
    ai_status = {"profile": "N/A", "advice": "N/A"}
    
    if total_inc > 0:
        debt_ratio = total_exp / total_inc
        features = scaler.transform([[total_inc, total_exp, debt_ratio]])
        cluster = kmeans_model.predict(features)[0]
        
        # Removed DBSCAN logic here
        ai_status = {
            "profile": tier_data[cluster]['name'],
            "advice": tier_data[cluster]['advice']
        }

    return jsonify({
        "income": total_inc,
        "expenses": total_exp,
        "txs": list(reversed(u_data['expenses'])),
        "ai": ai_status
    })

if __name__ == '__main__':
    app.run(debug=True, port=8000, use_reloader=False)
