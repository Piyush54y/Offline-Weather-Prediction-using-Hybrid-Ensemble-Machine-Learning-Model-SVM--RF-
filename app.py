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

st.title("🌦️ Weather Prediction System")

# ------------------ LOAD YOUR CSV ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("weather.csv")   # 🔥 YOUR FILE
    return train_test_split(df, test_size=0.2, random_state=42)

train_df, test_df = load_data()

# ------------------ FEATURE ENGINEERING ------------------
# (adjust if your CSV columns differ)

for df in [train_df, test_df]:
    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["month"] = pd.to_datetime(df["date"]).dt.month

# ------------------ ENCODING ------------------
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

    rf = RandomForestClassifier(n_estimators=300, max_depth=12)
    svm = SVC(probability=True, C=10, gamma=0.1)

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
mode = st.radio("Select Mode", ["🌐 Online (API)", "💻 Offline (ML)"])
city = st.selectbox("City", ["Delhi","Mumbai","Patna","Bangalore"])

# =========================================================
# 🌐 ONLINE MODE
# =========================================================
if mode == "🌐 Online (API)":

    if st.button("🌍 Get Live Weather"):

        temp, humidity, condition = get_weather_api(city)

        st.subheader("🌐 Real-Time Weather")

        c1, c2 = st.columns(2)
        c1.metric("🌡 Temp", f"{temp}°C")
        c2.metric("💧 Humidity", f"{humidity}%")

        st.success(f"Condition: {condition}")

# =========================================================
# 💻 OFFLINE MODE
# =========================================================
if mode == "💻 Offline (ML)":

    st.subheader("💻 Offline Auto Prediction")

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

        # Display dataset sample
        st.markdown("### 🌍 Dataset Sample")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🌡 Temp", round(temp,1))
        c2.metric("💧 Rain", precipitation)
        c3.metric("🌪 Wind", wind)
        c4.metric("📅 Month", month)

        # Prediction
        st.markdown("## 🔮 Prediction")

        if weather == "rain":
            st.success("🌧️ Rain Expected")
        elif weather == "sun":
            st.success("☀️ Clear Weather")
        else:
            st.success("☁️ Cloudy")

        st.write(f"Confidence: {round(confidence,2)}")

# =========================================================
# 📊 PERFORMANCE
# =========================================================
st.subheader("📊 Model Performance")

st.write(f"RF Accuracy: {rf_acc:.2f}")
st.write(f"SVM Accuracy: {svm_acc:.2f}")
st.write(f"Hybrid Accuracy: {hybrid_acc:.2f}")
st.write(f"⏱ Training Time: {train_time:.2f}s")

# ------------------ CONFUSION MATRIX ------------------
st.subheader("📉 Confusion Matrix")

cm = confusion_matrix(y_test, hybrid_pred)
st.dataframe(cm)
