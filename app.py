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
from sklearn.metrics import confusion_matrix, roc_curve, auc

st.set_page_config(layout="wide")

API_KEY = "efd7a881ace6419480e100155251006"

# ------------------ UI ------------------
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f172a,#1e293b,#334155); color:white;}
.card {background: rgba(255,255,255,0.08); padding:15px; border-radius:15px;
       text-align:center; backdrop-filter: blur(10px);}
</style>
""", unsafe_allow_html=True)

st.title("🌦️ Weather AI PRO MAX")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("seattle-weather.csv")
    return train_test_split(df, test_size=0.2, random_state=42)

train_df, test_df = load_data()

# ------------------ FEATURES ------------------
for df in [train_df, test_df]:
    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
    df["temp_range"] = df["temp_max"] - df["temp_min"]

le = LabelEncoder()
train_df["weather"] = le.fit_transform(train_df["weather"])
test_df["weather"] = le.transform(test_df["weather"])

X_train = train_df[["temp_avg","temp_range","wind"]]
y_train = train_df["weather"]

X_test = test_df[["temp_avg","temp_range","wind"]]
y_test = test_df["weather"]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ TRAIN ------------------
@st.cache_resource
def train_models():
    start = time.time()

    rf = RandomForestClassifier(n_estimators=300, max_depth=12)
    svm = SVC(probability=True, C=10, gamma=0.1)

    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    return rf, svm, time.time() - start

rf, svm, train_time = train_models()

# ------------------ ML PREDICT ------------------
def ml_predict(temp, wind):
    X_input = scaler.transform([[temp, 5, wind]])

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
        st.success(f"🤖 Prediction: {pred.upper()}")

# ------------------ PERFORMANCE ------------------
st.subheader("📊 Model Performance (Fixed)")

rf_acc = rf.score(X_test, y_test)
svm_acc = svm.score(X_test, y_test)

st.write(f"RF Accuracy: {rf_acc:.2f}")
st.write(f"SVM Accuracy: {svm_acc:.2f}")
st.write(f"⏱ Training Time: {train_time:.2f}s")

st.info("Metrics are computed on fixed test dataset (reproducible)")

# ------------------ DYNAMIC CONFUSION MATRIX ------------------
st.subheader("📉 Confusion Matrix (Dynamic Sample)")

idx = np.random.choice(len(X_test), 25)
sample_X = X_test[idx]
sample_y = y_test.iloc[idx]

preds = rf.predict(sample_X)

cm = confusion_matrix(sample_y, preds)

fig, ax = plt.subplots()
ax.imshow(cm)

for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, cm[i,j], ha="center", va="center")

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
ax2.set_title(f"AUC = {roc_auc:.2f}")

st.pyplot(fig2)
