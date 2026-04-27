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

st.title("🌦️ Weather AI PRO MAX")

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

# ------------------ ENCODING ------------------
le = LabelEncoder()
train_df["weather"] = le.fit_transform(train_df["weather"])
test_df["weather"] = le.transform(test_df["weather"])

# ------------------ FEATURES ------------------
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

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=4,
        random_state=42
    )

    svm = SVC(
        kernel="rbf",
        C=20,
        gamma=0.05,
        probability=True
    )

    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    return rf, svm, time.time() - start

rf, svm, train_time = train_models()

# ------------------ ML PREDICT ------------------
def ml_predict(temp, wind):
    temp_range = 5
    precipitation = 0
    month = 6

    X_input = scaler.transform([[temp, temp_range, wind, precipitation, month]])

    rf_p = rf.predict_proba(X_input)
    svm_p = svm.predict_proba(X_input)

    prob = (rf_p + svm_p)/2
    pred = np.argmax(prob, axis=1)

    return le.inverse_transform(pred)[0]

# ------------------ API ------------------
def get_weather(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
    return requests.get(url).json()["current"]

# ------------------ MODE ------------------
mode = st.radio("Select Mode", ["🌐 Online (API)", "💻 Offline (ML)"])

city = st.selectbox("City", ["Delhi","Mumbai","Patna","Bangalore"])

# ------------------ ONLINE ------------------
if mode == "🌐 Online (API)":
    if st.button("Get Weather"):
        data = get_weather(city)

        st.success(f"🌐 {data['condition']['text']}")
        st.write(f"🌡 Temp: {data['temp_c']}°C")
        st.write(f"💧 Humidity: {data['humidity']}")

# ------------------ OFFLINE ------------------
if mode == "💻 Offline (ML)":
    temp = st.slider("Temperature", 0, 50, 25)
    wind = st.slider("Wind", 0, 50, 10)

    if st.button("Predict"):
        pred = ml_predict(temp, wind)

        if pred == "rain":
            st.success("🌧️ Rain Expected")
        elif pred == "sun":
            st.success("☀️ Clear Weather")
        else:
            st.success("☁️ Cloudy")

# ------------------ PERFORMANCE ------------------
st.subheader("📊 Model Performance (Fixed)")

rf_acc = rf.score(X_test, y_test)
svm_acc = svm.score(X_test, y_test)

st.write(f"RF Accuracy: {rf_acc:.2f}")
st.write(f"SVM Accuracy: {svm_acc:.2f}")
st.write(f"⏱ Training Time: {train_time:.2f}s")

# ------------------ CONFUSION MATRIX ------------------
st.subheader("📉 Confusion Matrix")

cm = confusion_matrix(y_test, rf.predict(X_test))
st.dataframe(cm)
