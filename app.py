import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(layout="wide")

st.title("🌦️ Weather Prediction (Hybrid RF + SVM)")

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

# ------------------ SCALING ------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ TRAIN ------------------
@st.cache_resource
def train_models():
    start = time.time()

    rf = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42)
    svm = SVC(probability=True, C=20, gamma=0.05)

    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    train_time = time.time() - start
    return rf, svm, train_time

rf, svm, train_time = train_models()

# ------------------ HYBRID (PAPER METHOD) ------------------
def hybrid_predict(X):
    rf_prob = rf.predict_proba(X)
    svm_prob = svm.predict_proba(X)

    # Equal probability fusion (paper method)
    hybrid_prob = (rf_prob + svm_prob) / 2

    pred = np.argmax(hybrid_prob, axis=1)
    return pred, hybrid_prob

# ------------------ ACCURACY ------------------
rf_acc = rf.score(X_test, y_test)
svm_acc = svm.score(X_test, y_test)

hybrid_pred, _ = hybrid_predict(X_test)
hybrid_acc = accuracy_score(y_test, hybrid_pred)

# ------------------ OFFLINE AUTO MODE ------------------
st.subheader("💻 Offline Auto Prediction (Paper Hybrid Model)")

if st.button("⚡ Generate Prediction"):

    sample = test_df.sample(1)

    temp = float(sample["temp_avg"].values[0])
    wind = float(sample["wind"].values[0])
    precipitation = float(sample["precipitation"].values[0])
    month = int(sample["month"].values[0])

    X_input = scaler.transform([[temp, 5, wind, precipitation, month]])

    # Predictions
    rf_label = le.inverse_transform(rf.predict(X_input))[0]
    svm_label = le.inverse_transform(svm.predict(X_input))[0]

    pred, prob = hybrid_predict(X_input)
    hybrid_label = le.inverse_transform(pred)[0]
    confidence = np.max(prob)

    # ------------------ DISPLAY ------------------
    st.markdown("### 🌍 Dataset Sample")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌡 Temp", round(temp,1))
    c2.metric("💧 Rain", precipitation)
    c3.metric("🌪 Wind", wind)
    c4.metric("📅 Month", month)

    # Final Prediction
    st.markdown("## 🔮 Final Prediction (Hybrid)")
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

    c1.markdown(f"**🌳 RF**<br>{rf_label}<br>Acc: {rf_acc:.2f}", unsafe_allow_html=True)
    c2.markdown(f"**🧠 SVM**<br>{svm_label}<br>Acc: {svm_acc:.2f}", unsafe_allow_html=True)
    c3.markdown(f"**🔥 Hybrid**<br>{hybrid_label}<br>Acc: {hybrid_acc:.2f}", unsafe_allow_html=True)

# ------------------ PERFORMANCE ------------------
st.subheader("📊 Model Performance")

st.write(f"RF Accuracy: {rf_acc:.2f}")
st.write(f"SVM Accuracy: {svm_acc:.2f}")
st.write(f"Hybrid Accuracy: {hybrid_acc:.2f}")
st.write(f"⏱ Training Time: {train_time:.2f}s")

# ------------------ CONFUSION MATRIX ------------------
st.subheader("📉 Confusion Matrix")

cm = confusion_matrix(y_test, hybrid_pred)
st.dataframe(cm)
