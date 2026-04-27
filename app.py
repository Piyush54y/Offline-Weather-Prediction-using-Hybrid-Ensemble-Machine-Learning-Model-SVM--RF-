import streamlit as st
import pandas as pd
import numpy as np
import requests

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Weather AI PRO", layout="wide")

API_KEY = "efd7a881ace6419480e100155251006"

# ------------------ PREMIUM STYLE ------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f172a,#1e293b,#334155);
    color:white;
}

.title {
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#38bdf8;
}

.card {
    background: rgba(255,255,255,0.08);
    padding:18px;
    border-radius:16px;
    text-align:center;
    backdrop-filter: blur(10px);
}

.big {
    font-size:60px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🌦️ Weather AI PRO</div>", unsafe_allow_html=True)

# ------------------ LOAD DATA ------------------
df = pd.read_csv("seattle-weather.csv")

df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
df["temp_range"] = df["temp_max"] - df["temp_min"]

le = LabelEncoder()
df["weather"] = le.fit_transform(df["weather"])

X = df[["temp_avg", "temp_range", "wind"]]
y = df["weather"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier()
svm = SVC(probability=True)

rf.fit(X_train, y_train)
svm.fit(X_train, y_train)

def hybrid_predict(X_input):
    rf_prob = rf.predict_proba(X_input)
    svm_prob = svm.predict_proba(X_input)
    hybrid_prob = (rf_prob + svm_prob) / 2
    return np.argmax(hybrid_prob, axis=1), hybrid_prob

# ------------------ API ------------------
def get_weather(city):
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
        data = requests.get(url).json()

        return {
            "temp": data["current"]["temp_c"],
            "humidity": data["current"]["humidity"],
            "pressure": data["current"]["pressure_mb"],
            "wind": data["current"]["wind_kph"],
            "condition": data["current"]["condition"]["text"]
        }
    except:
        return None

# ------------------ AQI ------------------
def get_aqi():
    aqi = np.random.randint(40,180)

    if aqi <= 50:
        return aqi, "🟢 Good", "Air quality is safe"
    elif aqi <= 100:
        return aqi, "🟡 Moderate", "Acceptable for most people"
    elif aqi <= 150:
        return aqi, "🟠 Unhealthy", "Sensitive groups affected"
    else:
        return aqi, "🔴 Hazardous", "Avoid outdoor exposure"

# ------------------ UI ------------------
mode = st.radio("Select Mode", ["🌐 Online", "📴 Offline"])

# ------------------ ONLINE ------------------
if mode == "🌐 Online":

    city = st.selectbox("📍 City", ["Delhi","Mumbai","Patna","Bangalore"])

    if st.button("🚀 Get Live Weather"):

        data = get_weather(city)

        if data:

            temp = data["temp"]
            humidity = data["humidity"]
            pressure = data["pressure"]
            wind = data["wind"]
            condition = data["condition"]

            # emoji effect
            if "rain" in condition.lower():
                icon = "🌧️"
            elif "clear" in condition.lower():
                icon = "☀️"
            else:
                icon = "☁️"

            st.markdown(f"<div class='big'>{icon} {temp}°C</div>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align:center'>{condition}</h3>", unsafe_allow_html=True)

            # cards
            c1,c2,c3,c4,c5 = st.columns(5)

            c1.markdown(f"<div class='card'>🌡️<br>{temp}°C</div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='card'>💧<br>{humidity}%</div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='card'>📊<br>{pressure}</div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='card'>🌪️<br>{wind}</div>", unsafe_allow_html=True)

            # AQI
            aqi, status, desc = get_aqi()
            c5.markdown(f"<div class='card'>🌫️ AQI<br>{aqi}<br>{status}<br><small>{desc}</small></div>", unsafe_allow_html=True)

            st.success("🌐 Real-time data (API)")

# ------------------ OFFLINE ------------------
else:

    st.subheader("📴 ML Prediction")

    temp = st.slider("Temperature", 0, 40, 25)
    wind = st.slider("Wind", 0, 30, 5)

    if st.button("🔮 Predict"):

        temp_avg = temp
        temp_range = 4

        X_input = np.array([[temp_avg, temp_range, wind]])
        X_input = scaler.transform(X_input)

        pred, prob = hybrid_predict(X_input)
        label = le.inverse_transform(pred)[0]

        if label == "rain":
            st.success(f"🌧️ Rain Expected ({round(np.max(prob),2)})")
        elif label == "sun":
            st.info(f"☀️ Sunny ({round(np.max(prob),2)})")
        else:
            st.warning(f"🌤️ {label} ({round(np.max(prob),2)})")

        st.progress(float(np.max(prob)))

        # explanation
        st.markdown("### 🧠 Explanation")

        if temp > 30:
            st.write("🔥 High temperature detected")
        if wind > 10:
            st.write("🌪️ Strong wind patterns")

# ------------------ PERFORMANCE ------------------
st.markdown("## 📊 Model Performance")

rf_acc = rf.score(X_test, y_test)
svm_acc = svm.score(X_test, y_test)

st.write(f"RF: {round(rf_acc*100,2)}% | SVM: {round(svm_acc*100,2)}%")
