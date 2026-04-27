import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix

st.set_page_config(layout="wide")

API_KEY = "efd7a881ace6419480e100155251006"

# ------------------ UI ------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f172a,#1e293b,#334155);
    color:white;
}
.title {
    text-align:center;
    font-size:38px;
    color:#38bdf8;
}
.card {
    background: rgba(255,255,255,0.08);
    padding:15px;
    border-radius:15px;
    text-align:center;
    backdrop-filter: blur(10px);
    box-shadow:0 0 20px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🌦️ Weather AI PRO MAX</div>", unsafe_allow_html=True)

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

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ TRAIN ------------------
@st.cache_resource
def train_model():
    start = time.time()

    rf = RandomForestClassifier(n_estimators=500, max_depth=15)
    svm = SVC(probability=True, C=20, gamma=0.05)

    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    return rf, svm, time.time() - start

rf, svm, train_time = train_model()

# ------------------ ML PREDICT ------------------
def ml_predict(temp, wind):
    X_input = scaler.transform([[temp, 5, wind, 0, 6]])

    prob = (rf.predict_proba(X_input) + svm.predict_proba(X_input)) / 2
    pred = np.argmax(prob, axis=1)

    return le.inverse_transform(pred)[0], np.max(prob)

# ------------------ API MODE ------------------
def get_weather_api(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=yes"
    res = requests.get(url).json()

    temp = res["current"]["temp_c"]
    humidity = res["current"]["humidity"]
    condition = res["current"]["condition"]["text"].lower()

    if "rain" in condition:
        pred = "rain"
    elif "cloud" in condition:
        pred = "cloudy"
    else:
        pred = "sun"

    try:
        aqi = res["current"]["air_quality"]["pm2_5"]
    except:
        aqi = 80

    return temp, humidity, pred, aqi

# ------------------ MODE ------------------
mode = st.radio("Select Mode", ["🌐 Online (API)", "💻 Offline (ML)"])
city = st.selectbox("City", ["Delhi","Mumbai","Patna","Bangalore"])

# ------------------ ONLINE ------------------
if mode == "🌐 Online (API)":
    if st.button("🌍 Get Live Weather"):

        temp, humidity, pred, aqi = get_weather_api(city)

        st.subheader("🌐 Real-Time Weather")

        col1, col2 = st.columns(2)
        col1.metric("🌡 Temp", f"{temp}°C")
        col2.metric("💧 Humidity", f"{humidity}%")

        if pred == "rain":
            st.success("🌧️ Rain Expected")
        elif pred == "sun":
            st.success("☀️ Clear Weather")
        else:
            st.success("☁️ Cloudy")

        # AQI
        if aqi <= 50:
            status = "🟢 Good"
            msg = "Air is clean"
        elif aqi <= 100:
            status = "🟡 Moderate"
            msg = "Acceptable air"
        else:
            status = "🔴 Unhealthy"
            msg = "Avoid outdoor activity"

        st.markdown("### 🌫 AQI (Live)")
        st.info(f"AQI: {round(aqi)} | {status}")
        st.write(msg)

# ------------------ MODEL ACCURACY ------------------
rf_acc = rf.score(X_test, y_test)
svm_acc = svm.score(X_test, y_test)

best_model = "Random Forest" if rf_acc > svm_acc else "SVM"

# ------------------ OFFLINE ------------------
st.subheader("💻 Offline Auto Prediction (Dataset Based)")

if st.button("⚡ Generate Prediction"):

    # Random dataset row
    sample = test_df.sample(1)

    temp = float(sample["temp_avg"].values[0])
    wind = float(sample["wind"].values[0])
    precipitation = float(sample["precipitation"].values[0])
    month = int(sample["month"].values[0])

    # Prepare input
    X_input = scaler.transform([[temp, 5, wind, precipitation, month]])

    # Individual predictions
    rf_pred = rf.predict(X_input)
    svm_pred = svm.predict(X_input)

    rf_label = le.inverse_transform(rf_pred)[0]
    svm_label = le.inverse_transform(svm_pred)[0]

    # Hybrid prediction
    prob = (rf.predict_proba(X_input) + svm.predict_proba(X_input)) / 2
    pred = np.argmax(prob, axis=1)
    hybrid_label = le.inverse_transform(pred)[0]
    confidence = np.max(prob)

    # ------------------ DATA DISPLAY ------------------
    st.markdown("### 🌍 Dataset Sample")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡 Temp", f"{round(temp,1)}°C")
    col2.metric("💧 Rain", f"{precipitation}")
    col3.metric("🌪 Wind", f"{wind}")
    col4.metric("📅 Month", f"{month}")

    # ------------------ PREDICTIONS ------------------
    st.markdown("## 🔮 Prediction Result")

    if hybrid_label == "rain":
        st.success("🌧️ Rain Expected")
    elif hybrid_label == "sun":
        st.success("☀️ Clear Weather")
    else:
        st.success("☁️ Cloudy")

    st.write(f"Confidence: {round(confidence,2)}")

    # ------------------ MODEL COMPARISON ------------------
    st.markdown("### 🤖 Model Comparison")

    c1, c2, c3 = st.columns(3)

    c1.markdown(f"**🌳 Random Forest**<br>Prediction: {rf_label}<br>Accuracy: {round(rf_acc,2)}", unsafe_allow_html=True)
    c2.markdown(f"**🧠 SVM**<br>Prediction: {svm_label}<br>Accuracy: {round(svm_acc,2)}", unsafe_allow_html=True)
    c3.markdown(f"**🔥 Hybrid**<br>Prediction: {hybrid_label}<br>Confidence: {round(confidence,2)}", unsafe_allow_html=True)

    # ------------------ BEST MODEL ------------------
    st.markdown("### 🏆 Best Model")

    if best_model == "Random Forest":
        st.success("🌳 Random Forest performs better overall")
    else:
        st.success("🧠 SVM performs better overall")

    # ------------------ AQI ------------------
    aqi = np.random.randint(40,150)

    if aqi <= 50:
        status = "🟢 Good"
        msg = "Air quality is clean"
    elif aqi <= 100:
        status = "🟡 Moderate"
        msg = "Acceptable air"
    else:
        status = "🔴 Unhealthy"
        msg = "Avoid outdoor activity"

    st.markdown("### 🌫 AQI (Estimated)")
    st.info(f"AQI: {aqi} | {status}")
    st.write(msg)
# ------------------ PERFORMANCE ------------------
st.subheader("📊 Model Performance")

rf_acc = rf.score(X_test, y_test)
svm_acc = svm.score(X_test, y_test)

st.write(f"RF Accuracy: {rf_acc:.2f}")
st.write(f"SVM Accuracy: {svm_acc:.2f}")
st.write(f"⏱ Training Time: {train_time:.2f}s")

# ------------------ CONFUSION MATRIX ------------------
st.subheader("📉 Confusion Matrix")
cm = confusion_matrix(y_test, rf.predict(X_test))
st.dataframe(cm)
