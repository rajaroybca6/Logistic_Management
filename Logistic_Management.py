import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import requests
import time
import smtplib
from email.mime.text import MIMEText
import random # Added for simulation

# ============================================================
# ✅ SECURE API KEYS & SECRETS (Loaded from Streamlit Cloud)
# ============================================================
# These are loaded from the "Secrets" tab on your Streamlit Dashboard.
OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")
TOMTOM_API_KEY = st.secrets.get("TOMTOM_API_KEY", "")
CUSTOMS_STATUS_API_URL = st.secrets.get("CUSTOMS_STATUS_API_URL", "")
CUSTOMS_STATUS_API_KEY = st.secrets.get("CUSTOMS_STATUS_API_KEY", "")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Logistic Guardian Dashboard",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 🎨 UI/UX DESIGN SYSTEM (PREMIUM CSS)
# ============================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
      background:
        radial-gradient(1200px 600px at 10% 0%, rgba(46, 134, 222, 0.14), transparent 60%),
        radial-gradient(900px 500px at 90% 10%, rgba(155, 89, 182, 0.14), transparent 55%),
        linear-gradient(180deg, #f8fafc 0%, #f3f6fb 100%);
    }

    /* HEADER */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 10px;
    }

    /* CARDS */
    .css-card {
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.06);
        border: 1px solid rgba(255,255,255,0.6);
        margin-bottom: 1rem;
    }

    /* METRICS */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
        text-align: center;
    }

    /* RISK ALERTS */
    .risk-high {
        background: linear-gradient(135deg, #ff9966 0%, #ff5e62 100%);
        color: white; padding: 1rem; border-radius: 12px; text-align: center; font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 94, 98, 0.4);
    }
    .risk-low {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        color: white; padding: 1rem; border-radius: 12px; text-align: center; font-weight: bold;
        box-shadow: 0 4px 15px rgba(86, 171, 47, 0.4);
    }

    .stButton button {
        border-radius: 12px;
        font-weight: 600;
        border: 1px solid #e0e0e0;
        background: white;
        transition: all 0.2s;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-color: #2a5298;
        color: #2a5298;
    }
    </style>
""", unsafe_allow_html=True)


# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    model_path = "models/logistic_guardian_v3_2.pkl"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

pipeline = load_model()
geolocator = Nominatim(user_agent="logistic_guardian_app")


# --- HELPER FUNCTIONS ---
def get_coordinates(address):
    try:
        location = geolocator.geocode(address)
        if location:
            return (location.latitude, location.longitude)
        return None
    except:
        return None

def create_gauge_visual(probability):
    color = "#56ab2f" if probability < 0.3 else "#f1c40f" if probability < 0.7 else "#ff5e62"
    percentage = probability * 100
    html = f"""
    <div style="text-align: center; margin: 20px 0; background: white; padding: 20px; border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
        <div style="font-size: 3.5rem; font-weight: 800; color: {color};">{percentage:.1f}%</div>
        <div style="margin: 15px auto; width: 80%; height: 25px; background: #f0f0f0; border-radius: 20px; overflow: hidden; box-shadow: inset 0 2px 5px rgba(0,0,0,0.1);">
            <div style="width: {percentage}%; height: 100%; background: {color}; transition: width 0.5s ease-in-out;"></div>
        </div>
        <div style="font-size: 1.1rem; color: #888; font-weight: 600; margin-top: 10px; text-transform: uppercase; letter-spacing: 1px;">Delay Risk Probability</div>
    </div>
    """
    return html

def get_risk_factors(input_data):
    risks = []
    if input_data['distanza_km'].iloc[0] > 1000: risks.append("🔴 Long distance (>1000km)")
    if input_data['valore_merce_eur'].iloc[0] > 50000: risks.append("🔴 High value cargo (>€50k)")
    if input_data['rischio_meteo'].iloc[0] >= 4: risks.append("🔴 Severe weather risk")
    if input_data['rischio_doganale'].iloc[0] >= 4: risks.append("🔴 High customs complexity")
    if input_data['numero_transiti'].iloc[0] > 5: risks.append("🔴 Multiple transit points")
    if input_data['fragile'].iloc[0] == 1: risks.append("⚠️ Fragile goods")
    if input_data['tracking_gps'].iloc[0] == 0: risks.append("⚠️ No GPS tracking")
    if not risks: risks.append("✅ No significant risk factors detected")
    return risks

def haversine(lat1, lon1, lat2, lon2):
    r = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ============================================================
# ✅ REAL-TIME API INTEGRATION
# ============================================================
@st.cache_data(ttl=300)
def fetch_live_weather(lat: float, lon: float):
    if not OPENWEATHER_API_KEY:
        return {"_error": "Missing OPENWEATHER_API_KEY in Secrets"}
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        return {
            "temp_c": data.get("main", {}).get("temp"),
            "wind_mps": data.get("wind", {}).get("speed"),
            "weather": (data.get("weather") or [{}])[0].get("main"),
            "weather_desc": (data.get("weather") or [{}])[0].get("description"),
            "rain_1h": (data.get("rain") or {}).get("1h", 0.0),
            "snow_1h": (data.get("snow") or {}).get("1h", 0.0),
            "humidity": data.get("main", {}).get("humidity"),
            "visibility_m": data.get("visibility"),
        }
    except Exception as e:
        return {"_error": str(e)}

@st.cache_data(ttl=300)
def fetch_live_traffic(lat: float, lon: float):
    if not TOMTOM_API_KEY:
        return {"_error": "Missing TOMTOM_API_KEY in Secrets"}
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {"key": TOMTOM_API_KEY, "point": f"{lat},{lon}"}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        fsd = data.get("flowSegmentData", {})
        return {
            "current_speed": fsd.get("currentSpeed"),
            "free_flow_speed": fsd.get("freeFlowSpeed"),
            "road_closure": fsd.get("roadClosure"),
        }
    except Exception as e:
        return {"_error": str(e)}

@st.cache_data(ttl=300)
def fetch_customs_status(pickup: str, delivery: str, mode: str):
    if not CUSTOMS_STATUS_API_URL:
        sim_status = random.choice(["NORMAL", "NORMAL", "ELEVATED", "SEVERE"])
        sim_delay = 0
        if sim_status == "ELEVATED": sim_delay = random.randint(4, 12)
        if sim_status == "SEVERE": sim_delay = random.randint(12, 48)
        return {"status": sim_status, "summary": "Simulated Customs Data", "delay_hours_est": sim_delay}

    headers = {"Authorization": f"Bearer {CUSTOMS_STATUS_API_KEY}"} if CUSTOMS_STATUS_API_KEY else {}
    payload = {"pickup": pickup, "delivery": delivery, "mode": mode}
    try:
        r = requests.post(CUSTOMS_STATUS_API_URL, json=payload, headers=headers, timeout=12)
        data = r.json()
        return {"status": data.get("status"), "summary": data.get("summary"), "delay_hours_est": data.get("delay_hours_est")}
    except Exception as e:
        return {"_error": str(e)}

def compute_operational_overlay(weather: dict, traffic: dict, customs: dict):
    score = 0.0
    if weather and not weather.get("_error"):
        if (weather.get("rain_1h") or 0.0) >= 3: score += 0.15
        if (weather.get("snow_1h") or 0.0) >= 1: score += 0.20
    if traffic and not traffic.get("_error"):
        if traffic.get("road_closure"): score += 0.35
    if customs and not customs.get("_error"):
        stt = (customs.get("status") or "").upper()
        if stt == "SEVERE": score += 0.30
    return max(0.0, min(1.0, score))


# ============================================================
# ✅ SECURE AUTOMATED ALERTS (SMTP)
# ============================================================
def send_email_smtp(to_email: str, subject: str, body: str) -> dict:
    # Safe loading from Secrets
    host = st.secrets.get("SMTP_HOST", "smtp.gmail.com")
    port = int(st.secrets.get("SMTP_PORT", "587"))
    user = st.secrets.get("SMTP_USER", "")
    password = st.secrets.get("SMTP_PASS", "")
    from_email = st.secrets.get("ALERT_EMAIL_FROM", user)

    if not user or not password:
        return {"ok": False, "error": "Missing SMTP Credentials in Streamlit Secrets"}

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    try:
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls()
            server.login(user, password)
            recipients = [e.strip() for e in to_email.split(",") if e.strip()]
            server.sendmail(from_email, recipients, msg.as_string())
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def trigger_email_alert(context: str, prob: float, threshold: float, to_email: str, extra: dict = None):
    if not to_email or prob < threshold: return
    now = time.time()
    last_key = f"_last_alert_{context}"
    if (now - st.session_state.get(last_key, 0)) < 900: return
    
    st.session_state[last_key] = now
    body = f"🚨 Logistic Guardian Alert\nContext: {context}\nDelay Risk: {prob:.1%}\nTime: {datetime.now()}"
    res = send_email_smtp(to_email, "🚨 High Delay Risk Alert", body)
    if res.get("ok"): st.toast("✅ Email alert sent.", icon="📧")

# --- SIDEBAR & MAIN UI LOGIC ---
st.sidebar.markdown("<div class='sidebar-header'>🚚 Logistic Guardian</div>", unsafe_allow_html=True)

with st.sidebar.expander("🔔 Automated Alerts Settings", expanded=False):
    st.toggle("Enable Email Alerts", value=False, key="alert_email_on")
    st.text_input("Send alerts to (email)", value="", key="alert_email_to")
    st.slider("Alert threshold", 0.0, 1.0, 0.70, 0.05, key="alert_threshold")

def manual_input():
    with st.sidebar.form("manual_entry_form"):
        distanza = st.number_input("Distance (km)", min_value=0, value=500)
        valore = st.number_input("Cargo Value (€)", min_value=0, value=10000)
        peso = st.number_input("Weight (kg)", min_value=0, value=500)
        transiti = st.slider("Number of Transits", 0, 10, 2)
        modalita = st.selectbox("Transport Mode", ["Road", "Sea", "Railway", "Airplane"])
        meteo = st.slider("Weather Risk (1-5)", 1, 5, 2)
        doganale = st.slider("Customs Risk (1-5)", 1, 5, 1)
        fragile = st.radio("Fragile?", [0, 1], horizontal=True)
        gps = st.radio("GPS?", [0, 1], horizontal=True)
        submitted = st.form_submit_button("Update Data")
    return pd.DataFrame([{"distanza_km": distanza, "valore_merce_eur": valore, "peso_kg": peso, "numero_transiti": transiti, "rischio_meteo": meteo, "rischio_doganale": doganale, "modalità_trasporto": modalita, "fragile": fragile, "tracking_gps": gps}])

st.markdown('<div class="main-header">🚚 Logistic Guardian <span style="font-size:1.5rem; color:#888;">AI Analytics</span></div>', unsafe_allow_html=True)

if pipeline is None:
    st.error("❌ Model file not found in 'models/logistic_guardian_v3_2.pkl'")
else:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Prediction", "📂 Batch", "📈 Analytics", "📍 Live Tracking", "🗺️ Route", "🤖 AI Assistant"])

    with tab1:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1.5], gap="large")
        input_df = manual_input()
        st.session_state["last_manual_df"] = input_df.copy()
        with col1:
            st.subheader("📋 Shipment Config")
            st.dataframe(input_df.T, use_container_width=True)
            if st.button("🔍 Analyze Risk", type="primary", use_container_width=True):
                prob = pipeline.predict_proba(input_df)[0][1]
                pred = pipeline.predict(input_df)[0]
                st.session_state['last_result'] = (prob, pred)
                if st.session_state.get("alert_email_on"):
                    trigger_email_alert("Manual Entry", prob, st.session_state.alert_threshold, st.session_state.alert_email_to)
        with col2:
            if 'last_result' in st.session_state:
                prob, pred = st.session_state['last_result']
                div_class = 'risk-high' if pred == 1 else 'risk-low'
                st.markdown(f"<div class='{div_class}'>{'⚠️ DELAY' if pred == 1 else '✅ ON TIME'}</div>", unsafe_allow_html=True)
                st.markdown(create_gauge_visual(prob), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # [Remaining Tabs 2-5 follow your original logic using the pipeline]
    
    with tab6:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.subheader("🤖 AI Simulation & What-If Planner")
        active_df = st.session_state.get("last_manual_df")
        if active_df is not None:
            st.info(f"Context loaded: Last Manual Shipment")
            if "sim_df" not in st.session_state: st.session_state.sim_df = active_df.copy()
            sim = st.session_state.sim_df
            
            c1, c2, c3 = st.columns(3)
            if c1.button("✅ GPS ON"): sim.at[0, "tracking_gps"] = 1
            if c2.button("➕ Transit +1"): sim.at[0, "numero_transiti"] += 1
            if c3.button("🔄 Reset"): st.session_state.sim_df = active_df.copy()
            
            new_prob = pipeline.predict_proba(sim)[0][1]
            st.metric("Simulated Risk", f"{new_prob:.1%}", delta=f"{new_prob - pipeline.predict_proba(active_df)[0][1]:.1%}", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("✅ System Created By Raja Roy | 2025_2026")