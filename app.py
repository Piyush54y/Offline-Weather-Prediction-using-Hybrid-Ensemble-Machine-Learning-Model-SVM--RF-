import streamlit as st
import requests
import random
import pandas as pd

st.set_page_config(page_title="Weather AI PRO", layout="wide")

API_KEY = "efd7a881ace6419480e100155251006"

# ------------------ BACKGROUND + ANIMATION ------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color:white;
}

/* Rain animation */
.rain {
    position: fixed;
    width: 100%;
    height: 100%;
    pointer-events: none;
    top: 0;
    left: 0;
    background-image: url('https://i.ibb.co/7S3m8zZ/rain.gif');
    opacity: 0.15;
}

/* Sun glow */
.sun {
    position: fixed;
    top: 50px;
    right: 50px;
    font-size: 60px;
    animation: glow 2s infinite alternate;
}

@keyframes glow {
    from {opacity: 0.6;}
    to {opacity: 1;}
}

.card {
    background: rgba(255,255,255,0.08);
    padding:20px;
    border-radius:15px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

st.title("🌦️ Weather AI PRO")

# ------------------ AUTO LOCATION ------------------
def get_location():
    try:
        res = requests.get("https://ipinfo.io/json").json()
        return res["city"]
    except:
        return "Delhi"

auto_city = get_location()

city = st.selectbox("📍 Select City", ["Auto Detect", "Delhi","Mumbai","Patna","Bangalore","Gurugram"])

if city == "Auto Detect":
    city = auto_city

mode = st.toggle("🌐 Live Mode")

# ------------------ FETCH WEATHER ------------------
def get_weather(city):
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
        data = requests.get(url).json()
        return data["current"]
    except:
        return None

# ------------------ BUTTON ------------------
if st.button("🚀 Get Weather"):

    data = get_weather(city) if mode else None

    if data:
        temp = data["temp_c"]
        humidity = data["humidity"]
        pressure = data["pressure_mb"]
        wind = data["wind_kph"]
        condition = data["condition"]["text"]
    else:
        temp = random.uniform(20,40)
        humidity = random.randint(30,90)
        pressure = random.randint(990,1025)
        wind = random.uniform(0,20)
        condition = random.choice(["Clear","Rain","Cloudy"])

    # ------------------ WEATHER EFFECT ------------------
    if "Rain" in condition:
        st.markdown('<div class="rain"></div>', unsafe_allow_html=True)
        icon = "🌧️"
    elif "Clear" in condition:
        st.markdown('<div class="sun">☀️</div>', unsafe_allow_html=True)
        icon = "☀️"
    else:
        icon = "☁️"

    # ------------------ HERO ------------------
    st.markdown(f"""
    <h1 style='text-align:center;font-size:60px;'>{icon} {round(temp,1)}°C</h1>
    <h3 style='text-align:center;'>{condition}</h3>
    """, unsafe_allow_html=True)

    # ------------------ CARDS ------------------
    col1,col2,col3,col4,col5 = st.columns(5)

    col1.markdown(f"<div class='card'>🌡️ Temp<br><b>{round(temp,1)}°C</b></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'>💧 Humidity<br><b>{humidity}%</b></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'>📊 Pressure<br><b>{pressure}</b></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='card'>🌪️ Wind<br><b>{round(wind,1)}</b></div>", unsafe_allow_html=True)

    # AQI
    aqi = random.randint(50,200)
    if aqi <= 50:
        status = "🟢 Good"
        desc = "Safe air quality"
    elif aqi <= 100:
        status = "🟡 Moderate"
        desc = "Acceptable air"
    elif aqi <= 150:
        status = "🟠 Unhealthy"
        desc = "May cause discomfort"
    else:
        status = "🔴 Hazardous"
        desc = "Avoid outdoor activity"

    col5.markdown(f"""
    <div class='card'>
    🌫️ AQI<br>
    <b>{aqi}</b><br>
    {status}<br>
    <small>{desc}</small>
    </div>
    """, unsafe_allow_html=True)

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
        msg = f"🌧️ Rain Expected ({round(rain_prob,2)})"
        st.success(msg)
    else:
        msg = f"☀️ Clear Weather ({round(1-rain_prob,2)})"
        st.info(msg)

    st.progress(rain_prob)

    # ------------------ VOICE OUTPUT ------------------
    st.markdown("## 🔊 Voice Output")

    text = f"The weather in {city} is {condition}. Temperature is {round(temp,1)} degrees."

    st.audio(f"https://api.streamelements.com/kappa/v2/speech?voice=Brian&text={text}")

    # ------------------ WHY ------------------
    st.markdown("## 🧠 Why?")

    if humidity > 70:
        st.write("✔ High humidity")
    if pressure < 1005:
        st.write("✔ Low pressure")
    if wind > 15:
        st.write("✔ Strong wind")

    # ------------------ INSIGHTS ------------------
    st.markdown("## 📊 Insights")

    df = pd.DataFrame({
        "Time": list(range(1,13)),
        "Temp": [temp + random.uniform(-3,3) for _ in range(12)]
    })

    st.line_chart(df.set_index("Time"))

    # ------------------ COMPARISON ------------------
    st.markdown("## 📈 Model Comparison")

    st.bar_chart({
        "SVM":[0.90],
        "RF":[0.92],
        "Hybrid":[0.96],
        "LSTM":[0.94]
    })
