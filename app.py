import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Weather AI PRO", layout="wide")

st.markdown("""
<style>
body {background-color: #0f172a; color: white;}
.big-title {font-size: 42px; font-weight: bold;}
.card {
    padding: 15px;
    border-radius: 12px;
    background: #1e293b;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🌦️ Weather AI PRO (Offline ML)</div>", unsafe_allow_html=True)

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("seattle-weather.csv")

    # FIX column names (important)
    df.columns = df.columns.str.strip().str.lower()

    return df

df = load_data()

# -------------------------
# FEATURE ENGINEERING
# -------------------------
df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
df["temp_range"] = df["temp_max"] - df["temp_min"]

features = ["temp_avg", "temp_range", "humidity", "wind"]
target = "weather"

# -------------------------
# USER CONTROL
# -------------------------
st.markdown("### ⚙️ Settings")

col1, col2 = st.columns(2)

with col1:
    random_split = st.checkbox("🔀 Randomize Train Split", value=True)

with col2:
    retrain = st.button("🔄 Retrain Model")

# -------------------------
# TRAIN MODEL FUNCTION
# -------------------------
def train_model(df, random_state):
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=random_state
    )

    scaler = StandardScaler()
    le = LabelEncoder()

    X_train = scaler.fit_transform(train_df[features])
    X_test = scaler.transform(test_df[features])

    y_train = le.fit_transform(train_df[target])
    y_test = le.transform(test_df[target])

    rf = RandomForestClassifier(n_estimators=200)
    svm = SVC(probability=True)

    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    # Hybrid
    rf_p = rf.predict_proba(X_test)
    svm_p = svm.predict_proba(X_test)

    hybrid_p = (rf_p + svm_p) / 2
    hybrid_pred = np.argmax(hybrid_p, axis=1)

    acc = accuracy_score(y_test, hybrid_pred)
    cm = confusion_matrix(y_test, hybrid_pred)

    return {
        "rf": rf,
        "svm": svm,
        "scaler": scaler,
        "le": le,
        "acc": acc,
        "cm": cm,
        "X_test": X_test,
        "y_test": y_test
    }

# -------------------------
# TRAIN / RETRAIN LOGIC
# -------------------------
if "model_data" not in st.session_state or retrain:
    rs = None if random_split else 42
    st.session_state.model_data = train_model(df, rs)

model_data = st.session_state.model_data

# -------------------------
# RANDOM PREDICTION
# -------------------------
sample = df.sample(1)

X_sample = model_data["scaler"].transform(sample[features])

rf_p = model_data["rf"].predict_proba(X_sample)
svm_p = model_data["svm"].predict_proba(X_sample)

hybrid_p = (rf_p + svm_p) / 2
pred = np.argmax(hybrid_p, axis=1)[0]

label = model_data["le"].inverse_transform([pred])[0]
conf = np.max(hybrid_p)

# -------------------------
# UI DISPLAY
# -------------------------
st.markdown("### 🔮 Prediction")

emoji_map = {
    "rain": "🌧️",
    "sun": "☀️",
    "fog": "🌫️",
    "snow": "❄️",
    "drizzle": "🌦️"
}

emoji = emoji_map.get(label, "🌤️")

st.success(f"{emoji} **{label.upper()}** (Confidence: {conf:.2f})")

# -------------------------
# MODEL PERFORMANCE
# -------------------------
st.markdown("### 📊 Model Performance")

st.write(f"🎯 Hybrid Accuracy: **{model_data['acc']:.2f}**")

# -------------------------
# CONFUSION MATRIX (DYNAMIC)
# -------------------------
st.markdown("### 📊 Confusion Matrix")

fig, ax = plt.subplots()

ax.imshow(model_data["cm"])
ax.set_title("Confusion Matrix")

for i in range(model_data["cm"].shape[0]):
    for j in range(model_data["cm"].shape[1]):
        ax.text(j, i, model_data["cm"][i, j],
                ha="center", va="center", color="white")

st.pyplot(fig)

# -------------------------
# EXPLANATION
# -------------------------
st.markdown("### 🧠 Explanation")

st.info("""
✔ Model retrains when you click button  
✔ Random split makes results dynamic  
✔ Hybrid = RF + SVM average  
✔ Confusion matrix changes after retrain  

👉 If matrix doesn't change → model not retrained
""")
