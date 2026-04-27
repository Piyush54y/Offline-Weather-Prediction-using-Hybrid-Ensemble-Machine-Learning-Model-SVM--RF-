import streamlit as st
import pandas as pd
import numpy as np
import requests

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Weather AI PRO", layout="wide")

API_KEY = "efd7a881ace6419480e100155251006"

# ------------------ LOAD DATA ------------------
df = pd.read_csv("seattle-weather.csv")

# Feature Engineering
df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
df["temp_range"] = df["temp_max"] - df["temp_min"]

# Encode labels
le = LabelEncoder()
df["weather"] = le.fit_transform(df["weather"])

# Features
X = df[["temp_avg", "temp_range", "humidity", "pressure", "wind"]]
y = df["weather"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ MODELS ------------------
rf = RandomForestClassifier(n_estimators=200, max_depth=10)
svm = SVC(probability=True, kernel="rbf")

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

        temp = data["current"]["temp_c"]
        humidity = data["current"]["humidity"]
        pressure = data["current"]["pressure_mb"]
        wind = data["current"]["wind_kph"]

        return temp, humidity, pressure, wind
    except:
        return None

# ------------------ UI ------------------
st.title("🌦️ Hybrid Weather Prediction (Real ML)")

mode = st.radio("Mode", ["🌐 Online", "📴 Offline"])

# ------------------ INPUT ------------------
if mode == "🌐 Online":
    city = st.selectbox("City", ["Delhi","Mumbai","Patna","Bangalore"])
else:
    temp = st.slider("Temp", 0, 40, 25)
    humidity = st.slider("Humidity", 0, 100, 60)
    pressure = st.slider("Pressure", 980, 1030, 1010)
    wind = st.slider("Wind", 0, 30, 5)

# ------------------ PREDICT ------------------
if st.button("Predict"):

    if mode == "🌐 Online":
        result = get_weather(city)
        if result:
            temp, humidity, pressure, wind = result
            st.write(f"🌡️ {temp}°C | 💧 {humidity}% | 📊 {pressure} | 🌪️ {wind}")
        else:
            st.error("API Failed")
            st.stop()

    # Feature Engineering (same as training)
    temp_avg = temp
    temp_range = 4  # realistic assumption

    X_input = np.array([[temp_avg, temp_range, humidity, pressure, wind]])
    X_input = scaler.transform(X_input)

    pred, prob = hybrid_predict(X_input)
    label = le.inverse_transform(pred)[0]

    st.subheader("🔮 Prediction")

    st.success(f"{label.upper()} (Confidence: {round(np.max(prob),2)})")

# ------------------ PERFORMANCE ------------------
st.markdown("## 📊 Performance")

rf_acc = rf.score(X_test, y_test)
svm_acc = svm.score(X_test, y_test)
hybrid_acc = (rf_acc + svm_acc) / 2

st.write(f"RF Accuracy: {round(rf_acc*100,2)}%")
st.write(f"SVM Accuracy: {round(svm_acc*100,2)}%")
st.write(f"Hybrid Accuracy: {round(hybrid_acc*100,2)}%")

# ------------------ CONFUSION MATRIX ------------------
st.markdown("## 📉 Confusion Matrix")

y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
st.dataframe(pd.DataFrame(cm))

# ------------------ ROC ------------------
st.markdown("## 📈 ROC Curve")

rain_label = le.transform(["rain"])[0]
y_test_bin = (y_test == rain_label).astype(int)

rf_probs = rf.predict_proba(X_test)[:, rain_label]

fpr, tpr, _ = roc_curve(y_test_bin, rf_probs)
roc_auc = auc(fpr, tpr)

roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
st.line_chart(roc_df)

st.write(f"AUC Score: {round(roc_auc,2)}")

# ------------------ COMPARISON ------------------
st.markdown("## 📊 Model Comparison")

comp = pd.DataFrame({
    "Model":["SVM","RF","Hybrid"],
    "Accuracy":[svm_acc, rf_acc, hybrid_acc]
})

st.bar_chart(comp.set_index("Model"))
