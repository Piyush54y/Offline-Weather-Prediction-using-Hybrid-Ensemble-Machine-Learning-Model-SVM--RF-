import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(layout="wide")

API_KEY = "efd7a881ace6419480e100155251006"

# ------------------ PREMIUM UI ------------------
st.markdown("""
<style>
body {background: linear-gradient(135deg,#0f172a,#1e293b);}
.big-title {font-size:40px;font-weight:700;color:#38bdf8;}
.card {
    background:#1e293b;
    padding:20px;
    border-radius:15px;
    box-shadow:0 0 15px rgba(0,0,0,0.5);
}
.metric {
    text-align:center;
    font-size:20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🌦️ Weather Prediction PRO</div>', unsafe_allow_html=True)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("weather.csv")
    return train_test_split(df, test_size=0.2, random_state=42)

train_df, test_df = load_data()

# ------------------ AUTO COLUMN DETECTION ------------------
def detect_columns(df):
    cols = df.columns.str.lower()

    temp = None
    if "temp_max" in cols:
        temp = "temp_max"
    elif "temperature" in cols:
        temp = "temperature"
    elif "temp" in cols:
        temp = "temp"

    humidity = next((c for c in df.columns if "humidity" in c.lower()), None)
    wind = next((c for c in df.columns if "wind" in c.lower()), None)
    rain = next((c for c in df.columns if "precip" in c.lower()), None)
    target = next((c for c in df.columns if "weather" in c.lower()), None)

    return temp, humidity, wind, rain, target

temp_col, hum_col, wind_col, rain_col, target_col = detect_columns(train_df)

# ------------------ VALIDATION ------------------
if None in [temp_col, wind_col, target_col]:
    st.error("❌ Your CSV format not supported. Send columns.")
    st.stop()

# ------------------ FEATURES ------------------
features = [temp_col, wind_col]

if hum_col:
    features.append(hum_col)
if rain_col:
    features.append(rain_col)

X_train = train_df[features]
y_train = train_df[target_col]

X_test = test_df[features]
y_test = test_df[target_col]

# ------------------ ENCODING ------------------
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# ------------------ SCALING ------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ TRAIN ------------------
@st.cache_resource
def train_models():
    start = time.time()

    rf = RandomForestClassifier(n_estimators=200)
    svm = SVC(probability=True)

    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    return rf, svm, time.time() - start

rf, svm, train_time = train_models()

# ------------------ HYBRID ------------------
def hybrid_predict(X):
    rf_prob = rf.predict_proba(X)
    svm_prob = svm.predict_proba(X)
    prob = (rf_prob + svm_prob) / 2
    pred = np.argmax(prob, axis=1)
    return pred, prob

# ------------------ ACCURACY ------------------
rf_acc = rf.score(X_test, y_test)
svm_acc = svm.score(X_test, y_test)
hybrid_pred, _ = hybrid_predict(X_test)
hybrid_acc = accuracy_score(y_test, hybrid_pred)

# ------------------ API ------------------
def get_weather_api(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
    data = requests.get(url).json()

    temp = data["current"]["temp_c"]
    humidity = data["current"]["humidity"]
    condition = data["current"]["condition"]["text"]

    return temp, humidity, condition

# ------------------ MODE ------------------
mode = st.radio("Mode", ["🌐 Online", "💻 Offline"])

# =========================================================
# 🌐 ONLINE MODE
# =========================================================
if mode == "🌐 Online":
    city = st.selectbox("City", ["Delhi","Mumbai","Patna","Bangalore"])

    if st.button("🌍 Get Live Weather"):
        temp, humidity, condition = get_weather_api(city)

        st.markdown("### 🌐 Live Weather")

        c1, c2 = st.columns(2)
        c1.metric("🌡 Temp", f"{temp}°C")
        c2.metric("💧 Humidity", f"{humidity}%")

        st.success(f"☁️ {condition}")

# =========================================================
# 💻 OFFLINE MODE
# =========================================================
if mode == "💻 Offline":

    if st.button("⚡ Predict (Auto Dataset)"):
        sample = test_df.sample(1)

        X_input = scaler.transform(sample[features])

        pred, prob = hybrid_predict(X_input)
        weather = le.inverse_transform(pred)[0]
        confidence = np.max(prob)

        st.markdown("### 🔮 Prediction")

        if "rain" in weather.lower():
            st.success("🌧️ Rain Expected")
        elif "sun" in weather.lower():
            st.success("☀️ Clear Weather")
        else:
            st.success("☁️ Cloudy")

        st.write(f"Confidence: {round(confidence,2)}")

# ------------------ PERFORMANCE ------------------
st.markdown("### 📊 Model Performance")

c1, c2, c3 = st.columns(3)
c1.metric("🌳 RF", f"{rf_acc:.2f}")
c2.metric("🧠 SVM", f"{svm_acc:.2f}")
c3.metric("🔥 Hybrid", f"{hybrid_acc:.2f}")

st.write(f"⏱ Training Time: {train_time:.2f}s")

# ------------------ CONFUSION MATRIX ------------------
st.markdown("### 📉 Confusion Matrix")

cm = confusion_matrix(y_test, hybrid_pred)
st.dataframe(cm)
