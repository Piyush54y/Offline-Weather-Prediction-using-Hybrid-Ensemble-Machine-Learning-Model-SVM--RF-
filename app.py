import streamlit as st
import requests
from model import predict_weather

st.set_page_config(page_title="Weather AI", layout="centered")

st.markdown("""
<style>
.big-title {
    font-size: 40px;
    font-weight: bold;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🌦️ Offline + Online Weather AI System")
st.caption("Hybrid Model (SVM + Random Forest)")

# ===============================
# API CONFIG
# ===============================
API_KEY = "efd7a881ace6419480e100155251006"
CITY = "Patna"

def get_weather():
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={CITY}"
        data = requests.get(url).json()

        temp = data["current"]["temp_c"]
        humidity = data["current"]["humidity"]
        pressure = data["current"]["pressure_mb"]
        wind = data["current"]["wind_kph"]

        return temp, humidity, pressure, wind, True
    except:
        return 25, 60, 1013, 10, False  # fallback


# ===============================
# AUTO FETCH
# ===============================
with st.spinner("🔄 Fetching live weather..."):
    temp, humidity, pressure, wind, online = get_weather()

st.success("✅ Data Loaded")

# Show current values
st.markdown(f"""
📍 **City:** {CITY}  
🌡️ Temperature: **{temp}°C**  
💧 Humidity: **{humidity}%**  
🌪️ Wind: **{wind} km/h**  
""")

# ===============================
# PREDICTION
# ===============================
pred, prob = predict_weather([temp, humidity, pressure, wind])

st.markdown("---")

if pred == 1:
    st.markdown(f"""
    <div class="result-box" style="background: linear-gradient(90deg, #1e5631, #2ecc71); color:white;">
    🌧️ <b>Rain Expected</b><br>
    Confidence: {prob:.2f}
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="result-box" style="background: linear-gradient(90deg, #1f3c88, #3498db); color:white;">
    ☀️ <b>No Rain</b><br>
    Confidence: {prob:.2f}
    </div>
    """, unsafe_allow_html=True)

# ===============================
# STATUS
# ===============================
if online:
    st.info("🌐 Using LIVE weather data (Online Mode)")
else:
    st.warning("📴 Using Offline fallback prediction")
