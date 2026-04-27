import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc

st.set_page_config(layout="wide")

# ------------------ PREMIUM CSS ------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f172a,#1e293b,#334155);
    color:white;
}

/* CARD */
.card {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 15px;
    text-align:center;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 20px rgba(0,0,0,0.5);
}

/* SUN GLOW */
.sun {
    font-size:70px;
    animation: glow 2s infinite alternate;
}
@keyframes glow {
    from { text-shadow:0 0 10px yellow; }
    to { text-shadow:0 0 30px orange; }
}

/* RAIN */
.rain {
    position: fixed;
    width:100%;
    height:100%;
    pointer-events:none;
}
.drop {
    position:absolute;
    width:2px;
    height:15px;
    background:cyan;
    animation: fall linear infinite;
}
@keyframes fall {
    from { transform: translateY(-100px); }
    to { transform: translateY(100vh); }
}
</style>
""", unsafe_allow_html=True)

st.title("🌦️ Weather AI PRO MAX")

API_KEY = "efd7a881ace6419480e100155251006"

# ------------------ LOAD DATA ------------------
df = pd.read_csv("seattle-weather.csv")

df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
df["temp_range"] = df["temp_max"] - df["temp_min"]

le = LabelEncoder()
df["weather"] = le.fit_transform(df["weather"])

X = df[["temp_avg","temp_range","wind"]]
y = df["weather"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ TRAIN ------------------
start = time.time()

rf = RandomForestClassifier(n_estimators=200)
svm = SVC(probability=True)

rf.fit(X_train,y_train)
svm.fit(X_train,y_train)

train_time = time.time() - start

# ------------------ HYBRID ------------------
def hybrid_predict(X_input):
    rf_p = rf.predict_proba(X_input)
    svm_p = svm.predict_proba(X_input)
    prob = (rf_p + svm_p)/2
    return np.argmax(prob,axis=1), prob

# ------------------ API ------------------
def get_weather(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
    data = requests.get(url).json()
    return data["current"]

# ------------------ UI ------------------
city = st.selectbox("City",["Delhi","Mumbai","Patna","Bangalore"])

if st.button("🚀 Predict"):

    data = get_weather(city)

    temp = data["temp_c"]
    humidity = data["humidity"]
    pressure = data["pressure_mb"]
    wind = data["wind_kph"]
    condition = data["condition"]["text"].lower()

    # ------------------ ICON ------------------
    if "rain" in condition:
        st.markdown("<div class='rain'>" + "".join(
            [f"<div class='drop' style='left:{np.random.randint(0,100)}%; animation-duration:{np.random.uniform(0.5,1.5)}s'></div>" for _ in range(80)]
        ) + "</div>", unsafe_allow_html=True)
        icon = "🌧️"
    elif "clear" in condition or "sun" in condition:
        icon = "<div class='sun'>☀️</div>"
    else:
        icon = "☁️"

    st.markdown(f"<h1 style='text-align:center'>{icon} {temp}°C</h1>", unsafe_allow_html=True)

    # ------------------ CARDS ------------------
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f"<div class='card'>🌡️<br>{temp}</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'>💧<br>{humidity}</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'>📊<br>{pressure}</div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='card'>🌪️<br>{wind}</div>", unsafe_allow_html=True)

    # ------------------ ML ------------------
    X_input = scaler.transform([[temp,4,wind]])
    ml_pred, prob = hybrid_predict(X_input)
    ml_label = le.inverse_transform(ml_pred)[0]

    # ------------------ API LOGIC ------------------
    if humidity>70 and pressure<1005:
        api_label="rain"
    else:
        api_label="sun"

    # ------------------ FINAL ------------------
    final = ml_label if ml_label==api_label else api_label

    st.subheader("🔥 Final Prediction")
    st.success(final.upper())

    # ------------------ CONFLICT ------------------
    st.subheader("🧠 Conflict Analysis")
    st.write(f"ML: {ml_label} | API: {api_label}")

    if ml_label != api_label:
        st.warning("⚠ Conflict → API prioritized (real-time stronger)")
    else:
        st.success("✔ Agreement → High confidence")

# ------------------ METRICS ------------------
st.subheader("📊 Model Performance")

rf_acc = rf.score(X_test,y_test)
svm_acc = svm.score(X_test,y_test)

st.write(f"RF: {rf_acc:.2f} | SVM: {svm_acc:.2f}")
st.write(f"⏱️ Training Time: {round(train_time,2)} sec")

# ------------------ CONFUSION MATRIX ------------------
cm = confusion_matrix(y_test, rf.predict(X_test))
st.subheader("📉 Confusion Matrix")
st.dataframe(cm)

# ------------------ ROC ------------------
st.subheader("📈 ROC Curve")

rain_label = le.transform(["rain"])[0]
y_bin = (y_test==rain_label).astype(int)

probs = rf.predict_proba(X_test)[:,rain_label]

fpr,tpr,_ = roc_curve(y_bin,probs)
roc_auc = auc(fpr,tpr)

st.line_chart(pd.DataFrame({"FPR":fpr,"TPR":tpr}))
st.write(f"AUC: {roc_auc:.2f}")
