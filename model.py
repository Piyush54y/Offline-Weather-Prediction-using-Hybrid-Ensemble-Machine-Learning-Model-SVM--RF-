import numpy as np

def predict_weather(input_data):
    temp, humidity, pressure, wind = input_data

    # Simple realistic logic
    rain_score = 0

    if humidity > 70:
        rain_score += 1
    if pressure < 1005:
        rain_score += 1
    if wind > 15:
        rain_score += 1

    prob = rain_score / 3

    if prob > 0.5:
        return 1, prob
    else:
        return 0, 1 - prob
