import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(layout="wide")

API_KEY = "efd7a881ace6419480e100155251006"

st.title("🌦️ Weather AI PRO MAX")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    if not os.path.exists("weather.csv"):
        st.error("Upload weather.csv")
        st.stop()
    df = pd.read_csv("weather.csv")
    return train_test_split(df, test_size=0.2, random_state=42)

train_df, test_df = load_data()

# ------------------ FEATURES ------------------
features = [
    "MinTemp","MaxTemp","Rainfall",
    "Humidity9am","Humidity3pm",
    "Pressure9am","Pressure3pm",
    "Temp9am","Temp3pm"
]

target = "RainTomorrow"

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

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

# ------------------ API ------------------
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

# ================= ONLINE =================
if mode == "🌐 Online":
    city = st.selectbox("City", ["Delhi","Mumbai","Patna","Bangalore"])

    if st.button("🌍 Get Weather"):
        temp, hum, cond, aqi = api(city)

        st.metric("🌡 Temp", f"{temp}°C")
        st.metric("💧 Humidity", f"{hum}%")
        st.metric("🌫 AQI", round(aqi,1))

        st.success(cond)

# ================= OFFLINE =================
if mode == "💻 Offline":

    if st.button("⚡ Predict"):
        sample = test_df.sample(1)
        X_input = scaler.transform(sample[features])

        pred, prob = hybrid(X_input)
        weather = le.inverse_transform(pred)[0]

        st.markdown("## 🔮 Prediction")

        if weather == "Yes":
            st.success("🌧️ Rain Tomorrow")
        else:
            st.success("☀️ No Rain Tomorrow")

# ------------------ PERFORMANCE ------------------
st.markdown("## 📊 Performance")

st.write(f"RF: {rf_acc:.2f}")
st.write(f"SVM: {svm_acc:.2f}")
st.write(f"Hybrid: {hy_acc:.2f}")
st.write(f"Time: {t_time:.2f}s")

# ------------------ HEATMAP ------------------
st.markdown("## 📉 Confusion Matrix")

cm = confusion_matrix(y_test, hy_pred)

fig, ax = plt.subplots()
ax.imshow(cm, cmap='coolwarm')

for i in range(len(cm)):
    for j in range(len(cm[0])):
        ax.text(j, i, cm[i][j], ha='center', va='center', color='white')

st.pyplot(fig)
