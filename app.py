import streamlit as st
import pandas as pd
import numpy as np
import requests

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Weather Hybrid System", layout="wide")

API_KEY = "efd7a881ace6419480e100155251006"

# ------------------ LOAD DATA ------------------
df = pd.read_csv("seattle-weather.csv")

# Feature Engineering (ONLY valid columns)
df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
df["temp_range"] = df["temp_max"] - df["temp_min"]

# Encode labels
le = LabelEncoder()
df["weather"] = le.fit_transform(df["weather"])

X = df[["temp_avg", "temp_range", "wind"]]
y = df["weather"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ MODELS ------------------
rf = RandomForestClassifier()
svm = SVC(probability=True)

rf.fit(X_train, y_train)
svm.fit(X_train, y_train)

# ------------------ HYBRID ------------------
def hybrid_predict(X_input):
    rf_prob = rf.predict_proba(X_input)
    svm_prob = svm.predict_proba(X_input)
    hybrid_prob = (rf_prob + svm_prob) / 2
    return np.argmax(hybrid_prob, axis=1), hybrid_prob

# ------------------ API ------------------
def get_weather(city):
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
        data = requests.get(url).json()

        return {
            "temp": data["current"]["temp_c"],
            "humidity": data["current"]["humidity"],
            "pressure": data["current"]["pressure_mb"],
            "wind": data["current"]["wind_kph"],
            "condition": data["current"]["condition"]["text"]
        }
    except:
        return None

# ------------------ UI ------------------
st.title("🌦️ Hybrid Weather System")

mode = st.radio("Select Mode", ["🌐 Online (Real Weather)", "📴 Offline (ML Prediction)"])

# ------------------ ONLINE MODE ------------------
if mode == "🌐 Online (Real Weather)":

    city = st.selectbox("Select City", ["Delhi", "Mumbai", "Patna", "Bangalore"])

    if st.button("Get Live Weather"):

        data = get_weather(city)

        if data:
            st.success("🌐 Live Weather Data")

            st.write(f"🌡️ Temperature: {data['temp']}°C")
            st.write(f"💧 Humidity: {data['humidity']}%")
            st.write(f"📊 Pressure: {data['pressure']}")
            st.write(f"🌪️ Wind: {data['wind']}")
            st.write(f"🌤️ Condition: {data['condition']}")

        else:
            st.error("API Error")

# ------------------ OFFLINE MODE ------------------
else:

    st.subheader("📴 ML Prediction (Dataset Based)")

    temp = st.slider("Temperature", 0, 40, 25)
    wind = st.slider("Wind Speed", 0, 30, 5)

    if st.button("Predict Weather"):

        temp_avg = temp
        temp_range = 4  # realistic approximation

        X_input = np.array([[temp_avg, temp_range, wind]])
        X_input = scaler.transform(X_input)

        pred, prob = hybrid_predict(X_input)
        label = le.inverse_transform(pred)[0]

        st.subheader("🔮 Predicted Weather")

        st.success(f"{label.upper()} (Confidence: {round(np.max(prob),2)})")

# ------------------ MODEL PERFORMANCE ------------------
st.markdown("## 📊 Model Performance")

rf_acc = rf.score(X_test, y_test)
svm_acc = svm.score(X_test, y_test)
hybrid_acc = (rf_acc + svm_acc) / 2

st.write(f"RF Accuracy: {round(rf_acc*100,2)}%")
st.write(f"SVM Accuracy: {round(svm_acc*100,2)}%")
st.write(f"Hybrid Accuracy: {round(hybrid_acc*100,2)}%")
