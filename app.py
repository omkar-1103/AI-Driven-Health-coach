import sqlite3
import bcrypt
import threading
import time
import random
import uuid
import os
import numpy as np
import joblib
from datetime import datetime, date
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─────────────────────────────────────────────
#  LOAD ML MODELS & RAG
# ─────────────────────────────────────────────
import json as _json, pickle, warnings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))

_hr_model = joblib.load(os.path.join(BASE_DIR, 'health_risk_model.pkl'))
_hr_le    = joblib.load(os.path.join(BASE_DIR, 'health_risk_label_encoder.pkl'))

# Feature column order used during training (base + activity one-hot)
with open(os.path.join(BASE_DIR, 'health_risk_feature_columns.json')) as _f:
    _hr_feature_cols = _json.load(_f)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    with open(os.path.join(BASE_DIR, 'diabetes_model.pkl'), 'rb') as f:
        _diab_model = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'diabetes_scaler.pkl'), 'rb') as f:
        _diab_scaler = pickle.load(f)

# ─────────────────────────────────────────────
#  SETUP RAG & LLM
# ─────────────────────────────────────────────
try:
    _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    _vector_db = FAISS.load_local(
        os.path.join(BASE_DIR, "faiss_index"), 
        _embeddings, 
        allow_dangerous_deserialization=True
    )
    _retriever = _vector_db.as_retriever(search_kwargs={"k": 2})

    _llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    _prompt = ChatPromptTemplate.from_template("""
    You are an AI Health Assistant.
    
    Here is the user's current live health data:
    {vitals}
    
    Here is medical context retrieved from health guidelines:
    {context}
    
    Question:
    {question}
    
    Instructions:
    1. Answer the user's question clearly and compassionately.
    2. If the user asks about their own health, vitals, or predictions, use their 'live health data' to give a personalized assessment.
    3. If the user asks general medical questions, answer using ONLY the provided 'medical context'.
    4. Do not invent medical advice outside of these sources.
    5. GUARDRAIL: If the user asks a question that is malicious, harmful, illegal, or completely unrelated to health/medicine, you MUST refuse to answer. Respond EXACTLY with: "I am a health assistant. I cannot answer that question."
    """)

    def _format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)[:1000]

    def _retrieve_docs(inputs):
        docs = _retriever.invoke(inputs["question"])
        return _format_docs(docs)

    _rag_chain = (
        RunnablePassthrough.assign(context=_retrieve_docs)
        | _prompt
        | _llm
    )
    print("[OK] RAG Chain initialized successfully.")
except Exception as e:
    print(f"[WARNING] Failed to initialize RAG Chain: {e}")
    _rag_chain = None

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

DB_PATH = os.path.join(BASE_DIR, "health.db")

# ─────────────────────────────────────────────
#  DATABASE INIT
# ─────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT    UNIQUE NOT NULL,
                password    TEXT    NOT NULL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token       TEXT    UNIQUE NOT NULL,
                login_time  TEXT    NOT NULL DEFAULT (datetime('now')),
                logout_time TEXT,
                date        TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS wearable_data (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                timestamp      TEXT    NOT NULL DEFAULT (datetime('now')),
                heart_rate     REAL    NOT NULL,
                temperature    REAL    NOT NULL,
                blood_oxygen   REAL    NOT NULL,
                bp_systolic    INTEGER NOT NULL,
                bp_diastolic   INTEGER NOT NULL,
                step_count     INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_wearable_user_ts
                ON wearable_data (user_id, timestamp DESC);

            CREATE TABLE IF NOT EXISTS health_risk_predictions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                timestamp    TEXT    NOT NULL DEFAULT (datetime('now')),
                heart_rate   REAL    NOT NULL,
                temperature  REAL    NOT NULL,
                blood_oxygen REAL    NOT NULL,
                bp_systolic  INTEGER NOT NULL,
                bp_diastolic INTEGER NOT NULL,
                step_count   INTEGER NOT NULL,
                prediction   TEXT    NOT NULL,
                confidence   REAL    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS diabetes_predictions (
                id                        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id                   INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                timestamp                 TEXT    NOT NULL DEFAULT (datetime('now')),
                pregnancies               REAL    NOT NULL,
                glucose                   REAL    NOT NULL,
                blood_pressure            REAL    NOT NULL,
                skin_thickness            REAL    NOT NULL,
                insulin                   REAL    NOT NULL,
                bmi                       REAL    NOT NULL,
                diabetes_pedigree         REAL    NOT NULL,
                age                       INTEGER NOT NULL,
                prediction                TEXT    NOT NULL,
                confidence                REAL    NOT NULL
            );
        """)
        conn.commit()


# ─────────────────────────────────────────────
#  DATA GENERATION
# ─────────────────────────────────────────────

# Holds the "current" sensor state per user: {user_id: {metric: value}}
user_sensor_state = {}
sensor_lock = threading.Lock()

DEFAULT_STATE = {
    "heart_rate":    72.0,
    "temperature":   36.6,
    "blood_oxygen":  98.0,
    "bp_systolic":   120.0,
    "bp_diastolic":  80.0,
    "step_count":    0,
    "last_step_date": str(date.today()),
}

LIMITS = {
    "heart_rate":   (60.0,  120.0),
    "temperature":  (36.0,  38.2),
    "blood_oxygen": (95.0,  100.0),
    "bp_systolic":  (110.0, 140.0),
    "bp_diastolic": (71.0,  90.0),
}

DELTAS = {
    "heart_rate":   3.0,
    "temperature":  0.1,
    "blood_oxygen": 0.5,
    "bp_systolic":  2.0,
    "bp_diastolic": 1.5,
}


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def random_walk(prev, delta, lo, hi):
    change = random.uniform(-delta, delta)
    return round(clamp(prev + change, lo, hi), 2)


def get_or_init_state(user_id, conn):
    """Load the last wearable row for a user as seed, or use defaults."""
    with sensor_lock:
        if user_id not in user_sensor_state:
            row = conn.execute(
                """SELECT heart_rate, temperature, blood_oxygen,
                          bp_systolic, bp_diastolic, step_count, timestamp
                   FROM wearable_data
                   WHERE user_id = ?
                   ORDER BY id DESC LIMIT 1""",
                (user_id,)
            ).fetchone()
            if row:
                last_date = row["timestamp"][:10]
                today = str(date.today())
                state = {
                    "heart_rate":    row["heart_rate"],
                    "temperature":   row["temperature"],
                    "blood_oxygen":  row["blood_oxygen"],
                    "bp_systolic":   row["bp_systolic"],
                    "bp_diastolic":  row["bp_diastolic"],
                    "step_count":    row["step_count"] if last_date == today else 0,
                    "last_step_date": today,
                }
            else:
                state = dict(DEFAULT_STATE)
                state["last_step_date"] = str(date.today())
            user_sensor_state[user_id] = state
        return user_sensor_state[user_id]


def generate_data_for_user(user_id, conn):
    state = get_or_init_state(user_id, conn)
    today = str(date.today())

    # Reset step count at the start of a new day
    if state["last_step_date"] != today:
        state["step_count"] = 0
        state["last_step_date"] = today

    # Random-walk each metric
    for metric in ["heart_rate", "temperature", "blood_oxygen", "bp_systolic", "bp_diastolic"]:
        lo, hi = LIMITS[metric]
        state[metric] = random_walk(state[metric], DELTAS[metric], lo, hi)

    # Increment steps while session is active (called only for active sessions)
    state["step_count"] += random.randint(5, 10)

    conn.execute(
        """INSERT INTO wearable_data
               (user_id, heart_rate, temperature, blood_oxygen,
                bp_systolic, bp_diastolic, step_count)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            round(state["heart_rate"], 1),
            round(state["temperature"], 2),
            round(state["blood_oxygen"], 1),
            int(state["bp_systolic"]),
            int(state["bp_diastolic"]),
            state["step_count"],
        ),
    )


def data_generation_loop():
    """Background thread: every 5 seconds generate data for all active sessions."""
    while True:
        time.sleep(5)
        try:
            with get_db() as conn:
                active = conn.execute(
                    """SELECT DISTINCT user_id FROM sessions
                       WHERE logout_time IS NULL"""
                ).fetchall()
                for row in active:
                    generate_data_for_user(row["user_id"], conn)
                conn.commit()
        except Exception as exc:
            print(f"[DataGen] Error: {exc}")


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def get_user_from_token(token):
    if not token:
        return None
    with get_db() as conn:
        row = conn.execute(
            """SELECT u.id, u.username FROM users u
               JOIN sessions s ON s.user_id = u.id
               WHERE s.token = ? AND s.logout_time IS NULL""",
            (token,),
        ).fetchone()
    return row


# ─────────────────────────────────────────────
#  API ROUTES
# ─────────────────────────────────────────────

@app.route("/chat")
def chat_page():
    return send_from_directory("frontend", "chat.html")

@app.route("/predictions")
def predictions_page():
    return send_from_directory("frontend", "predictions.html")


@app.route("/")
def index():
    return send_from_directory("frontend", "login.html")


@app.route("/dashboard")
def dashboard_page():
    return send_from_directory("frontend", "dashboard.html")


@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, hashed),
            )
            conn.commit()
        return jsonify({"message": "Account created successfully!"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already exists."}), 409


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    with get_db() as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()

        if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
            return jsonify({"error": "Invalid username or password."}), 401

        token = str(uuid.uuid4())
        today = str(date.today())

        # Invalidate any previous active sessions for same user
        conn.execute(
            "UPDATE sessions SET logout_time = datetime('now') WHERE user_id = ? AND logout_time IS NULL",
            (user["id"],),
        )
        conn.execute(
            "INSERT INTO sessions (user_id, token, date) VALUES (?, ?, ?)",
            (user["id"], token, today),
        )
        conn.commit()

    return jsonify({"token": token, "username": username, "user_id": user["id"]}), 200


@app.route("/api/logout", methods=["POST"])
def logout():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    with get_db() as conn:
        conn.execute(
            "UPDATE sessions SET logout_time = datetime('now') WHERE token = ?",
            (token,),
        )
        conn.commit()
    return jsonify({"message": "Logged out."}), 200


@app.route("/api/dashboard", methods=["GET"])
def dashboard_data():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = get_user_from_token(token)
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    with get_db() as conn:
        # Latest reading
        latest = conn.execute(
            """SELECT * FROM wearable_data
               WHERE user_id = ?
               ORDER BY id DESC LIMIT 1""",
            (user["id"],),
        ).fetchone()

        # Last 30 readings for charts
        history = conn.execute(
            """SELECT timestamp, heart_rate, temperature, blood_oxygen,
                      bp_systolic, bp_diastolic, step_count
               FROM wearable_data
               WHERE user_id = ?
               ORDER BY id DESC LIMIT 30""",
            (user["id"],),
        ).fetchall()

    history_list = [dict(r) for r in reversed(history)]

    return jsonify({
        "username": user["username"],
        "latest": dict(latest) if latest else None,
        "history": history_list,
    }), 200


@app.route("/api/history", methods=["GET"])
def history_data():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = get_user_from_token(token)
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))
    offset = (page - 1) * per_page

    with get_db() as conn:
        rows = conn.execute(
            """SELECT timestamp, heart_rate, temperature, blood_oxygen,
                      bp_systolic, bp_diastolic, step_count
               FROM wearable_data
               WHERE user_id = ?
               ORDER BY id DESC LIMIT ? OFFSET ?""",
            (user["id"], per_page, offset),
        ).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) as cnt FROM wearable_data WHERE user_id = ?",
            (user["id"],),
        ).fetchone()["cnt"]

    return jsonify({
        "data": [dict(r) for r in rows],
        "total": total,
        "page": page,
        "per_page": per_page,
    }), 200


@app.route("/api/predict/health-risk", methods=["POST"])
def predict_health_risk():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = get_user_from_token(token)
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    activity = data.get("activity_status", "Walking")

    # Fetch latest wearable reading for this user
    with get_db() as conn:
        row = conn.execute(
            """SELECT heart_rate, temperature, blood_oxygen,
                      bp_systolic, bp_diastolic, step_count
               FROM wearable_data WHERE user_id = ?
               ORDER BY id DESC LIMIT 1""",
            (user["id"],)
        ).fetchone()

    if not row:
        return jsonify({"error": "No wearable data found. Please wait for data to be generated."}), 404

    # Build full feature vector matching training column order
    # Activity_Status is unknown from wearable alone — default to Walking (most common)
    base = {
        'Heart_Rate':       row['heart_rate'],
        'Body_Temperature': row['temperature'],
        'Blood_Oxygen':     row['blood_oxygen'],
        'Systolic_BP':      row['bp_systolic'],
        'Diastolic_BP':     row['bp_diastolic'],
        'Step_Count':       row['step_count'],
        'Activity_Status_Cycling': 1 if activity == 'Cycling' else 0,
        'Activity_Status_Resting': 1 if activity == 'Resting' else 0,
        'Activity_Status_Running': 1 if activity == 'Running' else 0,
        'Activity_Status_Walking': 1 if activity == 'Walking' else 0,
    }
    feature_vec = np.array([[base[col] for col in _hr_feature_cols]])

    pred_enc   = _hr_model.predict(feature_vec)[0]
    pred_prob  = _hr_model.predict_proba(feature_vec)[0]
    pred_label = _hr_le.inverse_transform([pred_enc])[0]
    confidence = round(float(pred_prob.max()) * 100, 1)

    with get_db() as conn:
        conn.execute(
            """INSERT INTO health_risk_predictions
               (user_id, heart_rate, temperature, blood_oxygen,
                bp_systolic, bp_diastolic, step_count, prediction, confidence)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (user["id"], row["heart_rate"], row["temperature"], row["blood_oxygen"],
             row["bp_systolic"], row["bp_diastolic"], row["step_count"],
             pred_label, confidence)
        )
        conn.commit()

    return jsonify({
        "prediction": pred_label,
        "confidence": confidence,
        "inputs": {
            "heart_rate":   row["heart_rate"],
            "temperature":  row["temperature"],
            "blood_oxygen": row["blood_oxygen"],
            "bp_systolic":  row["bp_systolic"],
            "bp_diastolic": row["bp_diastolic"],
            "step_count":   row["step_count"],
        }
    }), 200


@app.route("/api/predict/diabetes", methods=["POST"])
def predict_diabetes():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = get_user_from_token(token)
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    try:
        vals = [
            float(data["pregnancies"]),
            float(data["glucose"]),
            float(data["blood_pressure"]),
            float(data["skin_thickness"]),
            float(data["insulin"]),
            float(data["bmi"]),
            float(data["diabetes_pedigree"]),
            float(data["age"]),
        ]
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        features = np.array([vals])
        scaled   = _diab_scaler.transform(features)
        pred     = _diab_model.predict(scaled)[0]
        prob     = _diab_model.predict_proba(scaled)[0]

    pred_label = "Diabetic" if int(pred) == 1 else "Non-Diabetic"
    confidence = round(float(prob.max()) * 100, 1)

    with get_db() as conn:
        conn.execute(
            """INSERT INTO diabetes_predictions
               (user_id, pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, diabetes_pedigree, age, prediction, confidence)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (user["id"], *vals, pred_label, confidence)
        )
        conn.commit()

    return jsonify({
        "prediction": pred_label,
        "confidence": confidence,
    }), 200


@app.route("/api/predict/history", methods=["GET"])
def predict_history():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = get_user_from_token(token)
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    with get_db() as conn:
        hr_rows = conn.execute(
            """SELECT timestamp, heart_rate, temperature, blood_oxygen,
                      bp_systolic, bp_diastolic, step_count, prediction, confidence
               FROM health_risk_predictions WHERE user_id = ?
               ORDER BY id DESC LIMIT 10""",
            (user["id"],)
        ).fetchall()
        diab_rows = conn.execute(
            """SELECT timestamp, pregnancies, glucose, blood_pressure,
                      skin_thickness, insulin, bmi, diabetes_pedigree,
                      age, prediction, confidence
               FROM diabetes_predictions WHERE user_id = ?
               ORDER BY id DESC LIMIT 10""",
            (user["id"],)
        ).fetchall()

    return jsonify({
        "health_risk": [dict(r) for r in hr_rows],
        "diabetes":    [dict(r) for r in diab_rows],
    }), 200


@app.route("/api/chat", methods=["POST"])
def api_chat():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = get_user_from_token(token)
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Message is required."}), 400

    if not _rag_chain:
        return jsonify({"error": "AI Chat is currently unavailable due to server configuration."}), 500

    try:
        vitals_text = "No recent health data available for this user."
        with get_db() as conn:
            w_row = conn.execute(
                "SELECT * FROM wearable_data WHERE user_id = ? ORDER BY id DESC LIMIT 1", 
                (user["id"],)
            ).fetchone()
            
            d_row = conn.execute(
                "SELECT prediction, confidence FROM diabetes_predictions WHERE user_id = ? ORDER BY id DESC LIMIT 1", 
                (user["id"],)
            ).fetchone()
            
            if w_row:
                vitals_text = (
                    f"Heart Rate: {w_row['heart_rate']} bpm\n"
                    f"Blood Oxygen: {w_row['blood_oxygen']}%\n"
                    f"Blood Pressure: {w_row['bp_systolic']}/{w_row['bp_diastolic']} mmHg\n"
                    f"Body Temperature: {w_row['temperature']} °C\n"
                    f"Steps Today: {w_row['step_count']}\n"
                )
            if d_row:
                vitals_text += f"\nLatest Diabetes Risk Assessment: {d_row['prediction']} ({d_row['confidence']}% confidence)"

        response = _rag_chain.invoke({
            "question": message,
            "vitals": vitals_text
        })
        return jsonify({"reply": response.content}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to generate response: {e}"}), 500


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    gen_thread = threading.Thread(target=data_generation_loop, daemon=True)
    gen_thread.start()
    print("[OK] Health Dashboard running at http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)
