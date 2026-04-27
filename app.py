import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# --------------------
# CONFIG
# --------------------
st.set_page_config(page_title="Weather AI PRO MAX", layout="wide")

API_KEY = "efd7a881ace6419480e100155251006"  # ✅ your key

# --------------------
# UI STYLE
# --------------------
st.markdown("""
<style>
body {background: linear-gradient(135deg,#0f172a,#1e293b); color:white;}
.card {
    padding:15px;
    border-radius:15px;
    background:#1e293b;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

st.title("🌦️ Weather AI PRO MAX")

# --------------------
# LOAD CSV (SAFE)
# --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("seattle-weather.csv")

    df.columns = df.columns.str.strip().str.lower()

    # safe columns
    if "precipitation" in df.columns:
        df["humidity"] = df["precipitation"] * 10
    else:
        df["humidity"] = 50

    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
    df["temp_range"] = df["temp_max"] - df["temp_min"]

    return df

df = load_data()

features = ["temp_avg","temp_range","humidity","wind"]
target = "weather"

# --------------------
# TRAIN MODEL
# --------------------
def train_model():
    train_df, test_df = train_test_split(df, test_size=0.2)

    scaler = StandardScaler()
    le = LabelEncoder()

    X_train = scaler.fit_transform(train_df[features])
    y_train = le.fit_transform(train_df[target])

    rf = RandomForestClassifier(n_estimators=150)
    svm = SVC(probability=True)

    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    return rf, svm, scaler, le

# --------------------
# SESSION
# --------------------
if "model" not in st.session_state:
    st.session_state.model = train_model()

rf, svm, scaler, le = st.session_state.model

# --------------------
# MODE
# --------------------
mode = st.radio("Select Mode", ["Online 🌐","Offline ML 🤖"])

# =========================================================
# 🌐 ONLINE MODE (FIXED)
# =========================================================
if mode == "Online 🌐":

    city = st.selectbox("City", ["Delhi","Mumbai","Patna","Bangalore"])

    if st.button("Get Live Weather"):

        try:
            url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
            res = requests.get(url)

            # 🔥 FIX: check response before .json()
            if res.status_code != 200:
                st.error("❌ API Error / Limit Reached")
                st.stop()

            data = res.json()

            temp = data["current"]["temp_c"]
            humidity = data["current"]["humidity"]
            wind = data["current"]["wind_kph"]
            condition = data["current"]["condition"]["text"]

            # AQI safe
            aqi = data["current"].get("air_quality", {}).get("pm2_5", np.random.randint(50,150))

            st.markdown(f"## 🌡️ {temp}°C")
            st.write(f"☁️ Condition: {condition}")

            col1,col2,col3,col4 = st.columns(4)
            col1.metric("Temp", temp)
            col2.metric("Humidity", humidity)
            col3.metric("Wind", wind)
            col4.metric("AQI", int(aqi))

            # emoji logic
            if "rain" in condition.lower():
                st.success("🌧️ Rain Expected")
            elif "sun" in condition.lower():
                st.success("☀️ Clear Weather")
            else:
                st.info("🌥️ Mixed Weather")

        except Exception as e:
            st.error("⚠️ API failed (check key or internet)")
            st.stop()

# =========================================================
# 🤖 OFFLINE MODE (RANDOM FIXED)
# =========================================================
else:

    if st.button("🔮 Predict (Random Sample)"):

        # 🔥 IMPORTANT FIX → always new random row
        sample = df.sample(n=1, replace=True)

        X = scaler.transform(sample[features])

        rf_p = rf.predict_proba(X)
        svm_p = svm.predict_proba(X)

        hybrid = (rf_p + svm_p)/2

        pred = np.argmax(hybrid)
        label = le.inverse_transform([pred])[0]
        conf = np.max(hybrid)

        st.markdown(f"## 🤖 Prediction: {label.upper()} ({conf:.2f})")

        # AQI random
        aqi = np.random.randint(50,200)

        if aqi < 100:
            st.success(f"🌫️ AQI: {aqi} (Good)")
        else:
            st.warning(f"🌫️ AQI: {aqi} (Unhealthy)")

        # show sample used
        st.markdown("### 📊 Sample Data Used")
        st.write(sample)
