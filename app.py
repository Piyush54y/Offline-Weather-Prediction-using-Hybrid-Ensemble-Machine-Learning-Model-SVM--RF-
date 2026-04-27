import streamlit as st
import pandas as pd
import numpy as np
import requests

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Weather Hybrid AI", layout="wide")

API_KEY = "efd7a881ace6419480e100155251006"

# ------------------ LOAD DATA ------------------
df = pd.read_csv("seattle-weather.csv")

le = LabelEncoder()
df["weather"] = le.fit_transform(df["weather"])

X = df[["temp_max", "temp_min", "wind"]]
y = df["weather"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ------------------ TRAIN MODELS ------------------
rf = RandomForestClassifier()
svm = SVC(probability=True)

rf.fit(X_train, y_train)
svm.fit(X_train, y_train)

# ------------------ HYBRID ------------------
def hybrid_predict(X_input):
    rf_prob = rf.predict_proba(X_input)
    svm_prob = svm.predict_proba(X_input)

    hybrid_prob = (rf_prob + svm_prob) / 2
    return np.argmax(hybrid_prob, axis=1), hybrid_prob

# ------------------ ONLINE FETCH ------------------
def get_weather(city):
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
        data = requests.get(url).json()

        return (
            data["current"]["temp_c"],
            data["current"]["humidity"],
            data["current"]["pressure_mb"],
            data["current"]["wind_kph"],
            data["current"]["condition"]["text"]
        )
    except:
        return None

# ------------------ UI ------------------
st.title("🌦️ Hybrid Weather Prediction System")

mode = st.radio("Select Mode", ["🌐 Online (API)", "📴 Offline (Manual)"])

# ------------------ INPUT ------------------
if mode == "🌐 Online (API)":
    city = st.selectbox("Select City", ["Delhi", "Mumbai", "Patna", "Bangalore"])
else:
    temp = st.slider("Temperature", 0, 40, 25)
    wind = st.slider("Wind Speed", 0, 30, 5)

# ------------------ PREDICTION ------------------
if st.button("Predict"):

    if mode == "🌐 Online (API)":
        result = get_weather(city)

        if result:
            temp, humidity, pressure, wind, condition = result
            st.write(f"🌡️ {temp}°C | 💧 {humidity}% | 🌪️ {wind}")
        else:
            st.error("API Failed")
            st.stop()
    else:
        humidity = 60
        pressure = 1010
        condition = "Manual Input"

    temp_min = temp - 5

    X_input = np.array([[temp, temp_min, wind]])

    pred, prob = hybrid_predict(X_input)
    label = le.inverse_transform(pred)[0]

    # ------------------ OUTPUT ------------------
    st.subheader("🔮 Prediction")

    if label == "rain":
        st.success(f"🌧️ Rain Expected (Confidence: {round(np.max(prob),2)})")
    elif label == "sun":
        st.info(f"☀️ Sunny Weather (Confidence: {round(np.max(prob),2)})")
    else:
        st.warning(f"🌤️ {label} (Confidence: {round(np.max(prob),2)})")

# ------------------ PERFORMANCE ------------------
st.markdown("## 📊 Model Performance")

rf_acc = rf.score(X_test, y_test)
svm_acc = svm.score(X_test, y_test)

st.write(f"Random Forest Accuracy: {round(rf_acc*100,2)}%")
st.write(f"SVM Accuracy: {round(svm_acc*100,2)}%")

# ------------------ CONFUSION MATRIX ------------------
st.markdown("## 📉 Confusion Matrix")

y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

st.dataframe(pd.DataFrame(cm))

# ------------------ ROC ------------------
st.markdown("## 📈 ROC Curve")

# Binary (rain vs others)
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

comp_df = pd.DataFrame({
    "Model": ["SVM", "Random Forest", "Hybrid"],
    "Accuracy": [svm_acc, rf_acc, (svm_acc+rf_acc)/2]
})

st.bar_chart(comp_df.set_index("Model"))
