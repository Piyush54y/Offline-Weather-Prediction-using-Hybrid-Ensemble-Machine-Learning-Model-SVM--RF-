import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix

st.set_page_config(layout="wide")

API_KEY = "efd7a881ace6419480e100155251006"

# ------------------ UI STYLE ------------------
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f172a,#1e293b,#334155); color:white;}
.card {background: rgba(255,255,255,0.08); padding:15px; border-radius:15px;
       text-align:center; backdrop-filter: blur(10px);}
.sun {font-size:70px; animation: glow 2s infinite alternate;}
@keyframes glow {from {text-shadow:0 0 10px yellow;} to {text-shadow:0 0 30px orange;}}
</style>
""", unsafe_allow_html=True)

st.title("🌦️ Weather AI PRO MAX")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("seattle-weather.csv")
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    return train, test

train_df, test_df = load_data()

# ------------------ FEATURE ENGINEERING ------------------
for df in [train_df, test_df]:
    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
    df["temp_range"] = df["temp_max"] - df["temp_min"]

# ------------------ ENCODING ------------------
le = LabelEncoder()
train_df["weather"] = le.fit_transform(train_df["weather"])
test_df["weather"] = le.transform(test_df["weather"])

# ------------------ FEATURES ------------------
X_train = train_df[["temp_avg","temp_range","wind"]]
y_train = train_df["weather"]

X_test = test_df[["temp_avg","temp_range","wind"]]
y_test = test_df["weather"]

# ------------------ SCALING ------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ TRAIN ------------------
@st.cache_resource
def train_models():
    rf = RandomForestClassifier(n_estimators=300, max_depth=12)
    svm = SVC(probability=True, C=10, gamma=0.1)

    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    return rf, svm

rf, svm = train_models()

# ------------------ ML PREDICT ------------------
def ml_predict(temp, wind):
    temp_range = 5
    X_input = scaler.transform([[temp, temp_range, wind]])

    rf_p = rf.predict_proba(X_input)
    svm_p = svm.predict_proba(X_input)

    prob = (rf_p + svm_p)/2
    pred = np.argmax(prob, axis=1)

    return le.inverse_transform(pred)[0]

# ------------------ API ------------------
def get_weather(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
    return requests.get(url).json()["current"]

# ------------------ MODE SELECT ------------------
mode = st.radio("Select Mode", ["🌐 Online (API)", "💻 Offline (ML)"])

city = st.selectbox("📍 City", ["Delhi","Mumbai","Patna","Bangalore"])

# ------------------ ONLINE MODE ------------------
if mode == "🌐 Online (API)":

    if st.button("🚀 Get Live Weather"):

        data = get_weather(city)

        temp = data["temp_c"]
        humidity = data["humidity"]
        pressure = data["pressure_mb"]
        wind = data["wind_kph"]
        condition = data["condition"]["text"]

        # ICON
        if "rain" in condition.lower():
            icon = "🌧️"
        elif "sun" in condition.lower() or "clear" in condition.lower():
            icon = "<div class='sun'>☀️</div>"
        else:
            icon = "☁️"

        st.markdown(f"<h1 style='text-align:center'>{icon} {temp}°C</h1>", unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f"<div class='card'>🌡️ {temp}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='card'>💧 {humidity}</div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='card'>📊 {pressure}</div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='card'>🌪️ {wind}</div>", unsafe_allow_html=True)

        st.success(f"🌐 API Prediction: {condition.upper()}")

# ------------------ OFFLINE MODE ------------------
if mode == "💻 Offline (ML)":

    st.subheader("Enter Weather Values")

    temp = st.slider("Temperature (°C)", 0, 50, 25)
    wind = st.slider("Wind Speed", 0, 50, 10)

    if st.button("🤖 Predict (Offline ML)"):

        prediction = ml_predict(temp, wind)

        if prediction == "rain":
            st.success("🌧️ Rain Expected")
        elif prediction == "sun":
            st.success("☀️ Clear Weather")
        else:
            st.success("☁️ Cloudy")

# ------------------ MODEL PERFORMANCE ------------------
st.subheader("📊 Model Performance")

rf_acc = rf.score(X_test, y_test)
svm_acc = svm.score(X_test, y_test)

st.write(f"RF Accuracy: {rf_acc:.2f}")
st.write(f"SVM Accuracy: {svm_acc:.2f}")

# ------------------ CONFUSION MATRIX ------------------
st.subheader("📉 Confusion Matrix")

cm = confusion_matrix(y_test, rf.predict(X_test))
st.dataframe(cm)
