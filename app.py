import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc

st.set_page_config(layout="wide")

API_KEY = "efd7a881ace6419480e100155251006"

# ------------------ CSS ------------------
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f172a,#1e293b,#334155); color:white;}
.card {background: rgba(255,255,255,0.08); padding:15px; border-radius:15px;
       text-align:center; backdrop-filter: blur(10px);}
.sun {font-size:70px; animation: glow 2s infinite alternate;}
@keyframes glow {from {text-shadow:0 0 10px yellow;} to {text-shadow:0 0 30px orange;}}
</style>
""", unsafe_allow_html=True)

st.title("🌦️ Weather AI PRO MAX")

# ------------------ LOAD TRAIN/TEST ------------------
@st.cache_data
def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    return train, test

train_df, test_df = load_data()

# Feature Engineering
for df in [train_df, test_df]:
    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
    df["temp_range"] = df["temp_max"] - df["temp_min"]

# Encode
le = LabelEncoder()
train_df["weather"] = le.fit_transform(train_df["weather"])
test_df["weather"] = le.transform(test_df["weather"])

X_train = train_df[["temp_avg","temp_range","wind"]]
y_train = train_df["weather"]

X_test = test_df[["temp_avg","temp_range","wind"]]
y_test = test_df["weather"]

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ TRAIN MODEL ------------------
@st.cache_resource
def train_models():
    start = time.time()

    rf = RandomForestClassifier(n_estimators=300, max_depth=12)
    svm = SVC(probability=True, C=10, gamma=0.1)

    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    t = time.time() - start
    return rf, svm, t

rf, svm, train_time = train_models()

# ------------------ HYBRID ------------------
def hybrid_predict(X):
    rf_p = rf.predict_proba(X)
    svm_p = svm.predict_proba(X)
    prob = (rf_p + svm_p)/2
    return np.argmax(prob,axis=1), prob

# ------------------ API ------------------
def get_weather(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
    return requests.get(url).json()["current"]

# ------------------ UI ------------------
city = st.selectbox("📍 City", ["Delhi","Mumbai","Patna","Bangalore"])

if st.button("🚀 Predict Weather"):

    data = get_weather(city)

    temp = data["temp_c"]
    humidity = data["humidity"]
    pressure = data["pressure_mb"]
    wind = data["wind_kph"]
    condition = data["condition"]["text"].lower()

    # ICON
    if "rain" in condition:
        icon = "🌧️"
    elif "clear" in condition or "sun" in condition:
        icon = "<div class='sun'>☀️</div>"
    else:
        icon = "☁️"

    st.markdown(f"<h1 style='text-align:center'>{icon} {temp}°C</h1>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f"<div class='card'>🌡️ {temp}</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'>💧 {humidity}</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'>📊 {pressure}</div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='card'>🌪️ {wind}</div>", unsafe_allow_html=True)

    # ML
    X_input = scaler.transform([[temp,4,wind]])
    ml_pred, prob = hybrid_predict(X_input)
    ml_label = le.inverse_transform(ml_pred)[0]

    # API rule
    api_label = "rain" if humidity>70 and pressure<1005 else "sun"

    # FINAL
    final = ml_label if ml_label==api_label else api_label

    st.subheader("🔥 Final Prediction")
    st.success(final.upper())

    # Conflict
    st.subheader("🧠 Conflict Analysis")
    st.write(f"ML: {ml_label} | API: {api_label}")

    if ml_label != api_label:
        st.warning("⚠ Conflict → API prioritized (real-time stronger)")
    else:
        st.success("✔ Agreement → High confidence")

# ------------------ PERFORMANCE ------------------
st.subheader("📊 Model Performance")

rf_acc = rf.score(X_test, y_test)
svm_acc = svm.score(X_test, y_test)

st.write(f"RF Accuracy: {rf_acc:.2f}")
st.write(f"SVM Accuracy: {svm_acc:.2f}")
st.write(f"⏱ Training Time: {train_time:.2f} sec")

# ------------------ CONFUSION MATRIX (HEATMAP) ------------------
st.subheader("📉 Confusion Matrix (Heatmap)")

cm = confusion_matrix(y_test, rf.predict(X_test))

fig, ax = plt.subplots()
ax.imshow(cm)
for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, cm[i,j], ha="center", va="center")

ax.set_title("Confusion Matrix")
st.pyplot(fig)

# ------------------ ROC ------------------
st.subheader("📈 ROC Curve")

rain_label = le.transform(["rain"])[0]
y_bin = (y_test == rain_label).astype(int)

probs = rf.predict_proba(X_test)[:, rain_label]

fpr, tpr, _ = roc_curve(y_bin, probs)
roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr)
ax2.set_title(f"ROC Curve (AUC={roc_auc:.2f})")

st.pyplot(fig2)
