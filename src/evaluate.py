import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# LOAD MODEL AND OBJECTS
# =========================
model = load_model("modelo.h5")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# =========================
# LOAD DATA
# =========================
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# =========================
# STANDARDIZATION (SAME TRAINING ROUTINE)
# =========================
X_test = scaler.transform(X_test)

# =========================
# EVALUATION
# =========================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Acurácia no teste: {accuracy:.4f}")
print(f"Loss: {loss:.4f}")

# =========================
# PREDICTION
# =========================
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# =========================
# REPORT
# =========================
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
