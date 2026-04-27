import streamlit as st
from model import predict_weather

st.set_page_config(page_title="Weather Prediction", layout="centered")

st.title("🌦️ Offline Weather Prediction System")
st.write("Hybrid Model (SVM + Random Forest)")

# Inputs
temp = st.slider("Temperature (°C)", 0, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 60)
pressure = st.slider("Pressure (hPa)", 900, 1100, 1013)
wind = st.slider("Wind Speed (km/h)", 0, 50, 10)

if st.button("Predict"):
    pred, prob = predict_weather([temp, humidity, pressure, wind])

    if pred == 1:
        st.success(f"🌧️ Rain Expected (Confidence: {prob:.2f})")
    else:
        st.info(f"☀️ No Rain (Confidence: {prob:.2f})")
