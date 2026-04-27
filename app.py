import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(layout="wide")

API_KEY = "efd7a881ace6419480e100155251006"

# ------------------ UI ------------------
st.markdown("""
<style>
body {background: linear-gradient(135deg,#0f172a,#1e293b);}
.title {font-size:38px;font-weight:bold;color:#38bdf8;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌦️ Weather AI PRO MAX</div>', unsafe_allow_html=True)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    if not os.path.exists("weather.csv"):
        st.error("❌ Upload weather.csv in project folder")
        st.stop()

    df = pd.read_csv("weather.csv")
    return train_test_split(df, test_size=0.2, random_state=42)

train_df, test_df = load_data()

# ------------------ COLUMN AUTO DETECT ------------------
def detect(df):
    cols = df.columns

    temp = next((c for c in cols if "temp" in c.lower()), None)
    humidity = next((c for c in cols if "humidity" in c.lower()), None)
    wind = next((c for c in cols if "wind" in c.lower()), None)
    rain = next((c for c in cols if "precip" in c.lower()), None)
    target = next((c for c in cols if "weather" in c.lower()), None)

    return temp, humidity, wind, rain, target

temp_col, hum_col, wind_col, rain_col, target_col = detect(train_df)

# ------------------ FEATURES ------------------
features = [c for c in [temp_col, hum_col, wind_col, rain_col] if c]

X_train = train_df[features]
y_train = train_df[target_col]

X_test = test_df[features]
y_test = test_df[target_col]

# ------------------ ENCODE ------------------
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# ------------------ SCALE ------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ TRAIN ------------------
@st.cache_resource
def train():
    start = time.time()

    rf = RandomForestClassifier(n_estimators=200)
    svm = SVC(probability=True)

    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    return rf, svm, time.time() - start

rf, svm, t_time = train()

# ------------------ HYBRID ------------------
def hybrid(X):
    rf_p = rf.predict_proba(X)
    svm_p = svm.predict_proba(X)
    prob = (rf_p + svm_p) / 2
    pred = np.argmax(prob, axis=1)
    return pred, prob

# ------------------ ACCURACY ------------------
rf_acc = rf.score(X_test, y_test)
svm_acc = svm.score(X_test, y_test)
hy_pred, _ = hybrid(X_test)
hy_acc = accuracy_score(y_test, hy_pred)

# ------------------ API + AQI ------------------
def api(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=yes"
    data = requests.get(url).json()

    temp = data["current"]["temp_c"]
    humidity = data["current"]["humidity"]
    cond = data["current"]["condition"]["text"]
    aqi = data["current"]["air_quality"]["pm2_5"]

    return temp, humidity, cond, aqi

# ------------------ MODE ------------------
mode = st.radio("Mode", ["🌐 Online", "💻 Offline"])

# =========================================================
# 🌐 ONLINE
# =========================================================
if mode == "🌐 Online":

    city = st.selectbox("City", ["Delhi","Mumbai","Patna","Bangalore"])

    if st.button("🌍 Get Weather"):

        temp, hum, cond, aqi = api(city)

        c1, c2, c3 = st.columns(3)
        c1.metric("🌡 Temp", f"{temp}°C")
        c2.metric("💧 Humidity", f"{hum}%")
        c3.metric("🌫 AQI", round(aqi,1))

        if aqi < 50:
            st.success("🟢 Good Air")
        elif aqi < 100:
            st.warning("🟡 Moderate")
        else:
            st.error("🔴 Unhealthy")

        st.success(f"☁️ {cond}")

# =========================================================
# 💻 OFFLINE
# =========================================================
if mode == "💻 Offline":

    if st.button("⚡ Predict"):

        sample = test_df.sample(1)
        X_input = scaler.transform(sample[features])

        pred, prob = hybrid(X_input)
        weather = le.inverse_transform(pred)[0]

        st.markdown("## 🔮 Prediction")

        if "rain" in weather.lower():
            st.success("🌧️ Rain")
        elif "sun" in weather.lower():
            st.success("☀️ Clear")
        else:
            st.success("☁️ Cloudy")

# =========================================================
# 📊 PERFORMANCE
# =========================================================
st.markdown("## 📊 Model Performance")

c1, c2, c3 = st.columns(3)
c1.metric("RF", f"{rf_acc:.2f}")
c2.metric("SVM", f"{svm_acc:.2f}")
c3.metric("Hybrid", f"{hy_acc:.2f}")

st.write(f"⏱ Time: {t_time:.2f}s")

# =========================================================
# 📉 HEATMAP
# =========================================================
st.markdown("## 📉 Confusion Matrix Heatmap")

cm = confusion_matrix(y_test, hy_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', ax=ax)
st.pyplot(fig)
