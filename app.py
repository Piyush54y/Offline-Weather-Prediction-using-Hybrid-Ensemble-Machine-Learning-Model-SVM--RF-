import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix

st.set_page_config(layout="wide")

API_KEY = "efd7a881ace6419480e100155251006"

# ------------------ UI ------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f172a,#1e293b,#334155);
    color:white;
}
.title {
    text-align:center;
    font-size:38px;
    color:#38bdf8;
}
.card {
    background: rgba(255,255,255,0.08);
    padding:15px;
    border-radius:15px;
    text-align:center;
    backdrop-filter: blur(10px);
    box-shadow:0 0 20px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🌦️ Weather AI PRO MAX</div>", unsafe_allow_html=True)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("seattle-weather.csv")
    return train_test_split(df, test_size=0.2, random_state=42)

train_df, test_df = load_data()

# ------------------ FEATURE ENGINEERING ------------------
for df in [train_df, test_df]:
    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["month"] = pd.to_datetime(df["date"]).dt.month

le = LabelEncoder()
train_df["weather"] = le.fit_transform(train_df["weather"])
test_df["weather"] = le.transform(test_df["weather"])

features = ["temp_avg","temp_range","wind","precipitation","month"]

X_train = train_df[features]
y_train = train_df["weather"]

X_test = test_df[features]
y_test = test_df["weather"]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ TRAIN ------------------
@st.cache_resource
def train_model():
    start = time.time()

    rf = RandomForestClassifier(n_estimators=500, max_depth=15)
    svm = SVC(probability=True, C=20, gamma=0.05)

    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    return rf, svm, time.time() - start

rf, svm, train_time = train_model()

# ------------------ ML PREDICT ------------------
def ml_predict(temp, wind):
    X_input = scaler.transform([[temp, 5, wind, 0, 6]])

    prob = (rf.predict_proba(X_input) + svm.predict_proba(X_input)) / 2
    pred = np.argmax(prob, axis=1)

    return le.inverse_transform(pred)[0], np.max(prob)

# ------------------ API MODE ------------------
def get_weather_api(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=yes"
    res = requests.get(url).json()

    temp = res["current"]["temp_c"]
    humidity = res["current"]["humidity"]
    condition = res["current"]["condition"]["text"].lower()

    if "rain" in condition:
        pred = "rain"
    elif "cloud" in condition:
        pred = "cloudy"
    else:
        pred = "sun"

    try:
        aqi = res["current"]["air_quality"]["pm2_5"]
    except:
        aqi = 80

    return temp, humidity, pred, aqi

# ------------------ MODE ------------------
mode = st.radio("Select Mode", ["🌐 Online (API)", "💻 Offline (ML)"])
city = st.selectbox("City", ["Delhi","Mumbai","Patna","Bangalore"])

# ------------------ ONLINE ------------------
if mode == "🌐 Online (API)":
    if st.button("🌍 Get Live Weather"):

        temp, humidity, pred, aqi = get_weather_api(city)

        st.subheader("🌐 Real-Time Weather")

        col1, col2 = st.columns(2)
        col1.metric("🌡 Temp", f"{temp}°C")
        col2.metric("💧 Humidity", f"{humidity}%")

        if pred == "rain":
            st.success("🌧️ Rain Expected")
        elif pred == "sun":
            st.success("☀️ Clear Weather")
        else:
            st.success("☁️ Cloudy")

        # AQI
        if aqi <= 50:
            status = "🟢 Good"
            msg = "Air is clean"
        elif aqi <= 100:
            status = "🟡 Moderate"
            msg = "Acceptable air"
        else:
            status = "🔴 Unhealthy"
            msg = "Avoid outdoor activity"

        st.markdown("### 🌫 AQI (Live)")
        st.info(f"AQI: {round(aqi)} | {status}")
        st.write(msg)

# ------------------ OFFLINE ------------------
if mode == "💻 Offline (ML)":
    temp = st.slider("🌡 Temperature", 0, 50, 25)
    wind = st.slider("🌪 Wind", 0, 50, 10)

    if st.button("🤖 Predict (ML)"):

        pred, conf = ml_predict(temp, wind)

        if pred == "rain":
            st.success("🌧️ Rain Expected")
        elif pred == "sun":
            st.success("☀️ Clear Weather")
        else:
            st.success("☁️ Cloudy")

        st.write(f"Confidence: {round(conf,2)}")

        # Simulated AQI
        aqi = np.random.randint(40,150)

        if aqi <= 50:
            status = "🟢 Good"
            msg = "Air quality is safe"
        elif aqi <= 100:
            status = "🟡 Moderate"
            msg = "Sensitive people be careful"
        else:
            status = "🔴 Unhealthy"
            msg = "Avoid outdoor exposure"

        st.markdown("### 🌫 AQI (Estimated)")
        st.info(f"AQI: {aqi} | {status}")
        st.write(msg)

# ------------------ PERFORMANCE ------------------
st.subheader("📊 Model Performance")

rf_acc = rf.score(X_test, y_test)
svm_acc = svm.score(X_test, y_test)

st.write(f"RF Accuracy: {rf_acc:.2f}")
st.write(f"SVM Accuracy: {svm_acc:.2f}")
st.write(f"⏱ Training Time: {train_time:.2f}s")

# ------------------ CONFUSION MATRIX ------------------
st.subheader("📉 Confusion Matrix")
cm = confusion_matrix(y_test, rf.predict(X_test))
st.dataframe(cm)
