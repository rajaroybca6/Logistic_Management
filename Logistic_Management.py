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
import random 

# ============================================================
# ✅ IMPORTANT: API KEYS & CONFIG
# ============================================================
OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")
TOMTOM_API_KEY = st.secrets.get("TOMTOM_API_KEY", "")
CUSTOMS_STATUS_API_URL = st.secrets.get("CUSTOMS_STATUS_API_URL", "")
CUSTOMS_STATUS_API_KEY = st.secrets.get("CUSTOMS_STATUS_API_KEY", "")

st.set_page_config(
    page_title="Logistic Guardian Dashboard",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 🎨 UI/UX DESIGN SYSTEM
# ============================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }
    .stApp { background: linear-gradient(180deg, #f8fafc 0%, #f3f6fb 100%); }
    .main-header { font-size: 3rem; font-weight: 700; background: linear-gradient(90deg, #0f2027 0%, #203a43 50%, #2c5364 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; }
    .css-card { background: rgba(255,255,255,0.85); backdrop-filter: blur(10px); padding: 1.5rem; border-radius: 18px; box-shadow: 0 10px 30px rgba(0,0,0,0.06); border: 1px solid rgba(255,255,255,0.6); margin-bottom: 1rem; }
    .risk-high { background: linear-gradient(135deg, #ff9966 0%, #ff5e62 100%); color: white; padding: 1rem; border-radius: 12px; text-align: center; font-weight: bold; }
    .risk-low { background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%); color: white; padding: 1rem; border-radius: 12px; text-align: center; font-weight: bold; }
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
geolocator = Nominatim(user_agent="rajaroy_guardian_v2_production")

# ============================================================
# 🗺️ GEOGRAPHY HELPERS
# ============================================================
CITY_COORDS = {
    "milan, italy": (45.4642, 9.1900), "rome, italy": (41.9028, 12.4964),
    "berlin, germany": (52.5200, 13.4050), "munich, germany": (48.1351, 11.5820),
    "paris, france": (48.8566, 2.3522), "london, uk": (51.5074, -0.1278)
}

def get_coordinates(address):
    addr_lower = address.lower().strip()
    if addr_lower in CITY_COORDS:
        return (CITY_COORDS[addr_lower][0], CITY_COORDS[addr_lower][1], address)
    try:
        time.sleep(1.1)
        location = geolocator.geocode(address, timeout=10)
        if location:
            return (location.latitude, location.longitude, location.address)
    except:
        pass
    return None

def haversine(lat1, lon1, lat2, lon2):
    r = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def create_gauge_visual(probability):
    color = "#56ab2f" if probability < 0.3 else "#f1c40f" if probability < 0.7 else "#ff5e62"
    percentage = probability * 100
    return f"""<div style="text-align: center; margin: 20px 0; background: white; padding: 20px; border-radius: 15px;">
        <div style="font-size: 3.5rem; font-weight: 800; color: {color};">{percentage:.1f}%</div>
        <div style="width: 80%; height: 25px; background: #f0f0f0; border-radius: 20px; margin: auto;">
            <div style="width: {percentage}%; height: 100%; background: {color}; border-radius: 20px;"></div>
        </div></div>"""

# ============================================================
# 🔌 LIVE API FETCHERS
# ============================================================
@st.cache_data(ttl=300)
def fetch_live_weather(lat, lon):
    if not OPENWEATHER_API_KEY: return {"_error": "No API Key"}
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        return {"temp_c": data['main']['temp'], "weather_desc": data['weather'][0]['description']}
    except: return {"_error": "API Failure"}

@st.cache_data(ttl=300)
def fetch_live_traffic(lat, lon):
    return {"current_speed": random.randint(60, 90), "road_closure": False}

@st.cache_data(ttl=300)
def fetch_customs_status(pickup, delivery, mode):
    return {"status": "NORMAL", "delay_hours_est": 0}

# ============================================================
# 🕹️ SIDEBAR & INPUTS
# ============================================================
st.sidebar.markdown("<div class='sidebar-header'>🚚 Logistic Guardian</div>", unsafe_allow_html=True)
def manual_input():
    with st.sidebar.form("manual_entry_form"):
        distanza = st.number_input("Distance (km)", 0, 10000, 500)
        valore = st.number_input("Value (€)", 0, 1000000, 10000)
        peso = st.number_input("Weight (kg)", 0, 50000, 500)
        transiti = st.slider("Transits", 0, 10, 2)
        modalita = st.selectbox("Transport Mode", ["Road", "Sea", "Railway", "Airplane"])
        meteo = st.slider("Weather Risk (1-5)", 1, 5, 2)
        doganale = st.slider("Customs Risk (1-5)", 1, 5, 1)
        fragile = st.radio("Fragile?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        gps = st.radio("GPS?", [0, 1], format_func=lambda x: "On" if x==1 else "Off")
        st.form_submit_button("Update Data")
    return pd.DataFrame([{"distanza_km": distanza, "valore_merce_eur": valore, "peso_kg": peso, "numero_transiti": transiti, "rischio_meteo": meteo, "rischio_doganale": doganale, "modalità_trasporto": modalita, "fragile": fragile, "tracking_gps": gps}])

# ============================================================
# 🖼️ MAIN APP INTERFACE
# ============================================================
st.markdown('<div class="main-header">🚚 Logistic Guardian AI</div>', unsafe_allow_html=True)

if pipeline is None:
    st.error("Model missing at 'models/logistic_guardian_v3_2.pkl'")
else:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Prediction", "📂 Batch", "📈 Analytics", "📍 Tracking", "🗺️ Route", "🤖 AI Simulation"])

    with tab1:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1.5])
        input_df = manual_input()
        with col1:
            st.subheader("📋 Shipment Config")
            # 🚀 FIX: Convert to string to avoid ArrowInvalid error
            st.dataframe(input_df.T.astype(str), use_container_width=True)
            if st.button("🔍 Analyze Risk", type="primary"):
                prob = pipeline.predict_proba(input_df)[0][1]
                pred = pipeline.predict(input_df)[0]
                st.session_state['res'] = (prob, pred)
        with col2:
            if 'res' in st.session_state:
                prob, pred = st.session_state['res']
                st.markdown(f"<div class='risk-{'high' if pred==1 else 'low'}'>{'⚠️ DELAY' if pred==1 else '✅ ON TIME'}</div>", unsafe_allow_html=True)
                st.markdown(create_gauge_visual(prob), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        up = st.file_uploader("Upload CSV")
        if up:
            batch = pd.read_csv(up)
            if st.button("🚀 Process"):
                batch['Risk'] = pipeline.predict_proba(batch)[:, 1]
                # 🚀 FIX: Ensure numeric/string consistency for display
                st.dataframe(batch.astype(str), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        p_addr = st.text_input("Pickup", "Milan, Italy")
        d_addr = st.text_input("Delivery", "Berlin, Germany")
        if st.button("🗺️ Plan Route"):
            p_geo = get_coordinates(p_addr)
            d_geo = get_coordinates(d_addr)
            if p_geo and d_geo:
                dist = haversine(p_geo[0], p_geo[1], d_geo[0], d_geo[1])
                route_df = pd.DataFrame([{"distanza_km": dist, "valore_merce_eur": 15000, "peso_kg": 1000, "numero_transiti": 1, "rischio_meteo": 2, "rischio_doganale": 1, "modalità_trasporto": "Road", "fragile": 0, "tracking_gps": 1}])
                prob = pipeline.predict_proba(route_df)[0][1]
                st.metric("Route Distance", f"{dist:.1f} km")
                st.markdown(create_gauge_visual(prob), unsafe_allow_html=True)
                m = folium.Map(location=[p_geo[0], p_geo[1]], zoom_start=5)
                folium.Marker([p_geo[0], p_geo[1]], icon=folium.Icon(color='blue')).add_to(m)
                folium.Marker([d_geo[0], d_geo[1]], icon=folium.Icon(color='red')).add_to(m)
                st_folium(m, width=700, height=400)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab6:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.subheader("🤖 AI Simulation")
        if "last_manual_df" in st.session_state:
            sim_df = st.session_state["last_manual_df"].copy()
            st.write("Current Baseline:")
            # 🚀 FIX: Force string types for visual consistency
            st.dataframe(sim_df.astype(str))
            
            c1, c2 = st.columns(2)
            if c1.button("Simulate No GPS"): sim_df.loc[0, 'tracking_gps'] = 0
            if c2.button("Simulate Airplane"): sim_df.loc[0, 'modalità_trasporto'] = "Airplane"
            
            new_prob = pipeline.predict_proba(sim_df)[0][1]
            st.metric("New Simulated Risk", f"{new_prob:.1%}")
        else:
            st.info("Please run a prediction in Tab 1 first.")
        st.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("✅ System Created By Raja Roy | 2026")
