import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(layout="wide")

st.title("🌦️ Weather AI PRO MAX (Stacking Hybrid)")

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

# ------------------ TRAIN STACKING MODEL ------------------
@st.cache_resource
def train_model():
    start = time.time()

    rf = RandomForestClassifier(n_estimators=300, max_depth=12)
    svm = SVC(probability=True, C=10, gamma=0.1)

    stack = StackingClassifier(
        estimators=[("rf", rf), ("svm", svm)],
        final_estimator=LogisticRegression(),
        passthrough=True
    )

    stack.fit(X_train, y_train)

    train_time = time.time() - start

    return rf, svm, stack, train_time

rf, svm, stack, train_time = train_model()

# ------------------ ACCURACY ------------------
rf_acc = rf.fit(X_train, y_train).score(X_test, y_test)
svm_acc = svm.fit(X_train, y_train).score(X_test, y_test)
stack_acc = stack.score(X_test, y_test)

# ------------------ BEST MODEL ------------------
if stack_acc > rf_acc and stack_acc > svm_acc:
    best_model = "🔥 Stacking Hybrid (Best)"
elif rf_acc > svm_acc:
    best_model = "🌳 Random Forest"
else:
    best_model = "🧠 SVM"

# ------------------ OFFLINE AUTO MODE ------------------
st.subheader("💻 Offline Auto Prediction (Stacking Model)")

if st.button("⚡ Generate Prediction"):

    sample = test_df.sample(1)

    temp = float(sample["temp_avg"].values[0])
    wind = float(sample["wind"].values[0])
    precipitation = float(sample["precipitation"].values[0])
    month = int(sample["month"].values[0])

    X_input = scaler.transform([[temp, 5, wind, precipitation, month]])

    # Predictions
    rf_pred = le.inverse_transform(rf.predict(X_input))[0]
    svm_pred = le.inverse_transform(svm.predict(X_input))[0]
    stack_pred = le.inverse_transform(stack.predict(X_input))[0]

    conf = np.max(stack.predict_proba(X_input))

    # ------------------ DISPLAY ------------------
    st.markdown("### 🌍 Dataset Sample")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌡 Temp", round(temp,1))
    c2.metric("💧 Rain", precipitation)
    c3.metric("🌪 Wind", wind)
    c4.metric("📅 Month", month)

    # Final Prediction
    st.markdown("## 🔮 Final Prediction (Stacking)")
    if stack_pred == "rain":
        st.success("🌧️ Rain Expected")
    elif stack_pred == "sun":
        st.success("☀️ Clear Weather")
    else:
        st.success("☁️ Cloudy")

    st.write(f"Confidence: {round(conf,2)}")

    # ------------------ MODEL COMPARISON ------------------
    st.markdown("### 🤖 Model Comparison")
    c1, c2, c3 = st.columns(3)

    c1.markdown(f"**🌳 RF**<br>{rf_pred}<br>Acc: {rf_acc:.2f}", unsafe_allow_html=True)
    c2.markdown(f"**🧠 SVM**<br>{svm_pred}<br>Acc: {svm_acc:.2f}", unsafe_allow_html=True)
    c3.markdown(f"**🔥 Hybrid**<br>{stack_pred}<br>Acc: {stack_acc:.2f}", unsafe_allow_html=True)

    # ------------------ BEST MODEL ------------------
    st.markdown("### 🏆 Best Model")
    st.success(best_model)

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

    st.markdown("### 🌫 AQI")
    st.info(f"AQI: {aqi} | {status}")
    st.write(msg)

# ------------------ PERFORMANCE ------------------
st.subheader("📊 Model Performance")

st.write(f"RF Accuracy: {rf_acc:.2f}")
st.write(f"SVM Accuracy: {svm_acc:.2f}")
st.write(f"Stacking Accuracy: {stack_acc:.2f}")
st.write(f"⏱ Training Time: {train_time:.2f}s")

# ------------------ CONFUSION MATRIX ------------------
st.subheader("📉 Confusion Matrix")

cm = confusion_matrix(y_test, stack.predict(X_test))
st.dataframe(cm)
