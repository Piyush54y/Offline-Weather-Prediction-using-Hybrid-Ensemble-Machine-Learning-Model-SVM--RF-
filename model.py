import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Dummy training data (replace later if needed)
X = np.random.rand(200, 4)
y = np.random.randint(0, 2, 200)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
svm = SVC(probability=True)
rf = RandomForestClassifier()

svm.fit(X_scaled, y)
rf.fit(X_scaled, y)

# Hybrid prediction
def predict_weather(input_data):
    data = scaler.transform([input_data])

    svm_prob = svm.predict_proba(data)[0]
    rf_prob = rf.predict_proba(data)[0]

    # Hybrid (equal weight)
    hybrid_prob = 0.5 * svm_prob + 0.5 * rf_prob

    prediction = np.argmax(hybrid_prob)

    return prediction, hybrid_prob[prediction]
