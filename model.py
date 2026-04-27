import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

X = np.random.rand(200, 4)
y = np.random.randint(0, 2, 200)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

svm = SVC(probability=True)
rf = RandomForestClassifier()

svm.fit(X_scaled, y)
rf.fit(X_scaled, y)

def predict_weather(input_data):
    data = scaler.transform([input_data])

    svm_prob = svm.predict_proba(data)[0]
    rf_prob = rf.predict_proba(data)[0]

    hybrid_prob = 0.5 * svm_prob + 0.5 * rf_prob

    prediction = np.argmax(hybrid_prob)

    return prediction, hybrid_prob[prediction]
