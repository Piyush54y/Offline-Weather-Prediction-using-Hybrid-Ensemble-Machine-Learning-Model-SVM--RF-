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

st.title("🌦️ Weather Prediction System (Hybrid + API)")

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

# ------------------ SCALING ------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ TRAIN ------------------
@st.cache_resource
def train_models():
    start = time.time()

    rf = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42)
    svm = SVC(probability=True, C=20, gamma=0.05)

    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    return rf, svm, time.time() - start

rf, svm, train_time = train_models()

# ------------------ HYBRID MODEL ------------------
def hybrid_predict(X):
    rf_prob = rf.predict_proba(X)
    svm_prob = svm.predict_proba(X)

    # Paper-based fusion
    hybrid_prob = (rf_prob + svm_prob) / 2
    pred = np.argmax(hybrid_prob, axis=1)

    return pred, hybrid_prob

# ------------------ ACCURACY ------------------
rf_acc = rf.score(X_test, y_test)
svm_acc = svm.score(X_test, y_test)

hybrid_pred, _ = hybrid_predict(X_test)
hybrid_acc = accuracy_score(y_test, hybrid_pred)

# ------------------ API ------------------
def get_weather_api(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=yes"
    data = requests.get(url).json()

    temp = data["current"]["temp_c"]
    humidity = data["current"]["humidity"]
    condition = data["current"]["condition"]["text"].lower()

    # Simple interpretation
    if "rain" in condition:
        pred = "rain"
    elif "cloud" in condition:
        pred = "cloudy"
    else:
        pred = "sun"

    return temp, humidity, condition, pred

# ------------------ MODE ------------------
mode = st.radio("Select Mode", ["🌐 Online (API)", "💻 Offline (Hybrid ML)"])
city = st.selectbox("City", ["Delhi","Mumbai","Patna","Bangalore"])

# =========================================================
# 🌐 ONLINE MODE (API ONLY)
# =========================================================
if mode == "🌐 Online (API)":

    if st.button("🌍 Get Live Weather"):

        temp, humidity, condition, pred = get_weather_api(city)

        st.subheader("🌐 Real-Time Weather (API)")

        c1, c2 = st.columns(2)
        c1.metric("🌡 Temperature", f"{temp}°C")
        c2.metric("💧 Humidity", f"{humidity}%")

        st.write(f"Condition: {condition}")

        if pred == "rain":
            st.success("🌧️ Rain Detected (API)")
        elif pred == "sun":
            st.success("☀️ Clear Weather (API)")
        else:
            st.success("☁️ Cloudy Weather (API)")

# =========================================================
# 💻 OFFLINE MODE (HYBRID ML)
# =========================================================
if mode == "💻 Offline (Hybrid ML)":

    st.subheader("💻 Offline Auto Prediction (Hybrid Model)")

    if st.button("⚡ Generate Prediction"):

        sample = test_df.sample(1)

        temp = float(sample["temp_avg"].values[0])
        wind = float(sample["wind"].values[0])
        precipitation = float(sample["precipitation"].values[0])
        month = int(sample["month"].values[0])

        X_input = scaler.transform([[temp, 5, wind, precipitation, month]])

        pred, prob = hybrid_predict(X_input)
        weather = le.inverse_transform(pred)[0]
        confidence = np.max(prob)

        # DISPLAY
        st.markdown("### 🌍 Dataset Sample Used")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🌡 Temp", round(temp,1))
        c2.metric("💧 Rain", precipitation)
        c3.metric("🌪 Wind", wind)
        c4.metric("📅 Month", month)

        st.markdown("## 🔮 Hybrid Prediction")

        if weather == "rain":
            st.success("🌧️ Rain Expected (Hybrid)")
        elif weather == "sun":
            st.success("☀️ Clear Weather (Hybrid)")
        else:
            st.success("☁️ Cloudy (Hybrid)")

        st.write(f"Confidence: {round(confidence,2)}")

        st.info("🔥 Hybrid model combines RF + SVM probabilities")

# =========================================================
# 📊 PERFORMANCE
# =========================================================
st.subheader("📊 Model Performance")

st.write(f"Random Forest Accuracy: {rf_acc:.2f}")
st.write(f"SVM Accuracy: {svm_acc:.2f}")
st.write(f"Hybrid Accuracy: {hybrid_acc:.2f}")
st.write(f"⏱ Training Time: {train_time:.2f}s")

# ------------------ CONFUSION MATRIX ------------------
st.subheader("📉 Confusion Matrix (Hybrid)")

cm = confusion_matrix(y_test, hybrid_pred)
st.dataframe(cm)
