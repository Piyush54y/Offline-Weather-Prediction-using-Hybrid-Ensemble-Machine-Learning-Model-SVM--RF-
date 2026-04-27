import streamlit as st
import pandas as pd
import numpy as np
import requests

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Weather AI PRO MAX", layout="wide")

API_KEY = "efd7a881ace6419480e100155251006"

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

# ------------------ MODELS ------------------
rf = RandomForestClassifier(n_estimators=200)
svm = SVC(probability=True)

rf.fit(X_train, y_train)
svm.fit(X_train, y_train)

# ------------------ HYBRID ML ------------------
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
            "condition": data["current"]["condition"]["text"].lower()
        }
    except:
        return None

# ------------------ RAIN RULE ------------------
def api_rain_logic(humidity, pressure, wind):
    score = 0
    if humidity > 70: score += 1
    if pressure < 1005: score += 1
    if wind > 15: score += 1

    if score >= 2:
        return "rain"
    elif score == 1:
        return "cloudy"
    else:
        return "sun"

# ------------------ UI ------------------
st.title("🌦️ Weather AI PRO MAX (ML + API Hybrid)")

city = st.selectbox("📍 Select City", ["Delhi","Mumbai","Patna","Bangalore"])

if st.button("🚀 Predict Weather"):

    data = get_weather(city)

    if not data:
        st.error("API Failed")
        st.stop()

    temp = data["temp"]
    humidity = data["humidity"]
    pressure = data["pressure"]
    wind = data["wind"]
    api_condition = data["condition"]

    # ------------------ SHOW API ------------------
    st.subheader("🌐 Real-Time Weather")

    st.write(f"🌡️ Temp: {temp}°C")
    st.write(f"💧 Humidity: {humidity}%")
    st.write(f"📊 Pressure: {pressure}")
    st.write(f"🌪️ Wind: {wind}")
    st.write(f"🌤️ Condition: {api_condition}")

    # ------------------ ML PREDICTION ------------------
    temp_avg = temp
    temp_range = 4

    X_input = np.array([[temp_avg, temp_range, wind]])
    X_input = scaler.transform(X_input)

    ml_pred, prob = hybrid_predict(X_input)
    ml_label = le.inverse_transform(ml_pred)[0]

    st.subheader("🧠 ML Prediction")
    st.write(f"{ml_label} (Confidence: {round(np.max(prob),2)})")

    # ------------------ API LOGIC ------------------
    api_label = api_rain_logic(humidity, pressure, wind)

    st.subheader("🔍 API Logic Prediction")
    st.write(api_label)

    # ------------------ FINAL HYBRID DECISION ------------------
    st.subheader("🔥 Final Hybrid Prediction")

    # combine logic
    if ml_label == api_label:
        final = ml_label
    else:
        # priority to API if strong signals
        if humidity > 75:
            final = "rain"
        else:
            final = ml_label

    # emoji output
    if "rain" in final:
        st.success("🌧️ Rain Expected")
    elif "sun" in final:
        st.info("☀️ Clear Weather")
    else:
        st.warning("🌥️ Cloudy Weather")

    # ------------------ EXPLANATION ------------------
    st.markdown("### 🧠 Explanation")

    st.write(f"• ML predicted: {ml_label}")
    st.write(f"• API logic predicted: {api_label}")

    if ml_label == api_label:
        st.write("✔ Both agree → High confidence")
    else:
        st.write("⚠ Conflict → API prioritized due to real-time data")
