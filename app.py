from flask import Flask, render_template, request, jsonify
import requests
from model import predict_weather

app = Flask(__name__)

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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    city = request.json.get("city")

    temp, humidity, pressure, wind, online = get_weather(city)

    pred, prob = predict_weather([temp, humidity, pressure, wind])

    return jsonify({
        "temp": temp,
        "humidity": humidity,
        "pressure": pressure,
        "wind": wind,
        "prediction": "Rain" if pred == 1 else "No Rain",
        "confidence": round(prob, 2),
        "mode": "Online" if online else "Offline"
    })

if __name__ == "__main__":
    app.run(debug=True)
