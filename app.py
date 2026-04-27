import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="Weather AI PRO MAX", layout="wide")

API_KEY = "YOUR_API_KEY_HERE"  # 🔴 replace

# -----------------------
# UI STYLE (PREMIUM)
# -----------------------
st.markdown("""
<style>
body {background: linear-gradient(135deg,#0f172a,#1e293b); color:white;}
.card {
    padding:20px;
    border-radius:15px;
    background:#1e293b;
    box-shadow:0 0 15px rgba(0,255,255,0.2);
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

st.title("🌦️ Weather AI PRO MAX")

# -----------------------
# LOAD DATA (SAFE)
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("seattle-weather.csv")

    # 🔥 FIX ALL COLUMN ISSUES
    df.columns = df.columns.str.strip().str.lower()

    required = ["temp_max","temp_min","wind","weather"]
    for col in required:
        if col not in df.columns:
            st.error(f"❌ Missing column: {col}")
            st.stop()

    # safe humidity (if not present)
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

# -----------------------
# TRAIN MODEL
# -----------------------
def train():
    train_df, test_df = train_test_split(df, test_size=0.2)

    scaler = StandardScaler()
    le = LabelEncoder()

    X_train = scaler.fit_transform(train_df[features])
    X_test = scaler.transform(test_df[features])

    y_train = le.fit_transform(train_df[target])
    y_test = le.transform(test_df[target])

    rf = RandomForestClassifier(n_estimators=200)
    svm = SVC(probability=True)

    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    rf_p = rf.predict_proba(X_test)
    svm_p = svm.predict_proba(X_test)

    hybrid_p = (rf_p + svm_p)/2
    pred = np.argmax(hybrid_p, axis=1)

    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)

    return rf, svm, scaler, le, acc, cm

# -----------------------
# SESSION
# -----------------------
if "model" not in st.session_state:
    st.session_state.model = train()

rf, svm, scaler, le, acc, cm = st.session_state.model

# -----------------------
# MODE SELECT
# -----------------------
mode = st.radio("Select Mode", ["Online 🌐", "Offline ML 🤖"])

# -----------------------
# ONLINE MODE
# -----------------------
if mode == "Online 🌐":

    city = st.selectbox("City", ["Delhi","Mumbai","Patna","Bangalore"])

    if st.button("Get Live Weather"):

        url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
        data = requests.get(url).json()

        temp = data["current"]["temp_c"]
        humidity = data["current"]["humidity"]
        wind = data["current"]["wind_kph"]
        condition = data["current"]["condition"]["text"]

        aqi = data["current"].get("air_quality", {}).get("pm2_5", 50)

        st.markdown(f"## 🌡️ {temp}°C - {condition}")

        col1,col2,col3,col4 = st.columns(4)
        col1.metric("Temp", temp)
        col2.metric("Humidity", humidity)
        col3.metric("Wind", wind)
        col4.metric("AQI", int(aqi))

        # 🌧️ animation
        if "rain" in condition.lower():
            st.markdown("🌧️ Rain Expected")
        elif "sun" in condition.lower():
            st.markdown("☀️ Sunny Weather")

# -----------------------
# OFFLINE ML MODE
# -----------------------
else:

    sample = df.sample(1)

    X = scaler.transform(sample[features])

    rf_p = rf.predict_proba(X)
    svm_p = svm.predict_proba(X)

    hybrid_p = (rf_p + svm_p)/2

    pred = np.argmax(hybrid_p)
    label = le.inverse_transform([pred])[0]
    conf = np.max(hybrid_p)

    st.markdown(f"## 🤖 Prediction: {label.upper()} ({conf:.2f})")

    # -----------------------
    # METRICS
    # -----------------------
    st.markdown("### 📊 Model Accuracy")
    st.success(f"Hybrid Accuracy: {acc:.2f}")

    # -----------------------
    # CONFUSION MATRIX
    # -----------------------
    st.markdown("### 📊 Confusion Matrix")

    fig, ax = plt.subplots()
    ax.imshow(cm)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,cm[i,j],ha="center",va="center",color="white")

    st.pyplot(fig)

    # -----------------------
    # AQI (SIMULATED)
    # -----------------------
    st.markdown("### 🌫️ Air Quality")

    aqi = np.random.randint(50,200)

    if aqi < 100:
        st.success(f"AQI: {aqi} (Good)")
    else:
        st.warning(f"AQI: {aqi} (Moderate/Unhealthy)")
