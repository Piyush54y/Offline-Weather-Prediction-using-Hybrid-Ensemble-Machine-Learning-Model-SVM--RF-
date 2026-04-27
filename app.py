import streamlit as st
import requests
from model import predict_weather

st.set_page_config(page_title="Weather AI", layout="centered")

API_KEY = "efd7a881ace6419480e100155251006"

def get_weather(city):
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
        data = requests.get(url).json()

        temp = data["current"]["temp_c"]
        humidity = data["current"]["humidity"]
        pressure = data["current"]["pressure_mb"]
        wind = data["current"]["wind_kph"]

        return temp, humidity, pressure, wind, True
    except:
        return 25, 60, 1013, 10, False


# UI
st.title("🌦️ Offline + Online Weather Prediction")

city = st.selectbox("Select City", ["Patna", "Delhi", "Mumbai", "Bangalore"])

if st.button("Predict"):
    temp, humidity, pressure, wind, online = get_weather(city)

    pred, prob = predict_weather([temp, humidity, pressure, wind])

    st.write(f"🌡️ Temp: {temp}°C | 💧 {humidity}% | 🌪️ {wind}")

    if pred == 1:
        st.success(f"🌧️ Rain Expected (Confidence: {prob:.2f})")
    else:
        st.info(f"☀️ No Rain (Confidence: {prob:.2f})")

    if online:
        st.caption("🌐 Online Mode")
    else:
        st.caption("📴 Offline Mode")
