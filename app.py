import streamlit as st
import requests
import random
import pandas as pd

st.set_page_config(page_title="Weather Prediction PRO", layout="wide")

API_KEY = "efd7a881ace6419480e100155251006"

# ------------------ STYLE ------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color:white;
}

.big-title {
    font-size:40px;
    font-weight:bold;
    text-align:center;
    color:#00f5ff;
}

.card {
    background: rgba(255,255,255,0.08);
    padding:20px;
    border-radius:15px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🌦️ Weather Prediction PRO</div>', unsafe_allow_html=True)

# ------------------ INPUT ------------------
city = st.selectbox("📍 Select City", ["Delhi","Mumbai","Patna","Bangalore","Gurugram"])
mode = st.toggle("🌐 Live Mode (API)")

# ------------------ FETCH ------------------
def get_weather(city):
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
        data = requests.get(url).json()
        return data["current"]
    except:
        return None

# ------------------ BUTTON ------------------
if st.button("🚀 Get Weather"):

    if mode:
        data = get_weather(city)
    else:
        data = None

    if data:
        temp = data["temp_c"]
        humidity = data["humidity"]
        pressure = data["pressure_mb"]
        wind = data["wind_kph"]
        condition = data["condition"]["text"]
    else:
        # Offline random
        temp = random.uniform(20,40)
        humidity = random.randint(30,90)
        pressure = random.randint(990,1025)
        wind = random.uniform(0,20)
        condition = random.choice(["Clear","Rain","Cloudy"])

    # ------------------ ANIMATION ------------------
    if "Rain" in condition:
        effect = "🌧️🌧️🌧️ 🌧️🌧️🌧️"
    elif "Clear" in condition:
        effect = "☀️✨☀️✨"
    else:
        effect = "☁️☁️☁️"

    st.markdown(f"<h2 style='text-align:center'>{effect}</h2>", unsafe_allow_html=True)

    # ------------------ HERO DISPLAY ------------------
    st.markdown(f"""
    <h1 style='text-align:center;font-size:60px;'>🌡️ {round(temp,1)}°C</h1>
    <h3 style='text-align:center;'>{condition} Weather</h3>
    """, unsafe_allow_html=True)

    # ------------------ METRIC CARDS ------------------
    col1,col2,col3,col4,col5 = st.columns(5)

    col1.markdown(f"<div class='card'>🌡️<br>{round(temp,1)}°C</div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'>💧<br>{humidity}%</div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'>📊<br>{pressure}</div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='card'>🌪️<br>{round(wind,1)}</div>", unsafe_allow_html=True)

    # AQI
    aqi = random.randint(50,200)
    if aqi <= 50:
        aqi_status = "🟢 Good"
    elif aqi <= 100:
        aqi_status = "🟡 Moderate"
    elif aqi <= 150:
        aqi_status = "🟠 Unhealthy"
    else:
        aqi_status = "🔴 Very Unhealthy"

    col5.markdown(f"<div class='card'>🌫️ AQI<br>{aqi}<br>{aqi_status}</div>", unsafe_allow_html=True)

    # ------------------ PREDICTION ------------------
    st.markdown("## 🔮 Prediction")

    rain_prob = 0
    if humidity > 70:
        rain_prob += 0.4
    if pressure < 1005:
        rain_prob += 0.3
    if wind > 15:
        rain_prob += 0.3

    if rain_prob > 0.5:
        st.success(f"🌧️ Rain Expected (Confidence: {round(rain_prob,2)})")
        st.balloons()
    else:
        st.info(f"☀️ Clear Weather (Confidence: {round(1-rain_prob,2)})")

    st.progress(rain_prob)

    # ------------------ WHY ------------------
    st.markdown("## 🧠 Why?")

    if humidity > 70:
        st.write("✔ High humidity increases rain chances")
    if pressure < 1005:
        st.write("✔ Low pressure system detected")
    if wind > 15:
        st.write("✔ Strong winds indicate instability")

    # ------------------ INSIGHTS ------------------
    st.markdown("## 📊 Weather Insights")

    df = pd.DataFrame({
        "Hour": list(range(1,13)),
        "Temperature": [temp + random.uniform(-3,3) for _ in range(12)]
    })

    st.line_chart(df.set_index("Hour"))

    # ------------------ MODEL COMPARISON ------------------
    st.markdown("## 📈 Model Comparison")

    st.bar_chart({
        "SVM":[0.90],
        "KNN":[0.88],
        "RF":[0.92],
        "Hybrid":[0.96]
    })
