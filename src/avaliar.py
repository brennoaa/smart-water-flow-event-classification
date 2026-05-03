import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# CARREGAR MODELO E OBJETOS
# =========================
model = load_model("modelo.h5")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# =========================
# CARREGAR DADOS
# =========================
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# =========================
# NORMALIZAÇÃO (MESMO PADRÃO DO TREINO)
# =========================
X_test = scaler.transform(X_test)

# =========================
# AVALIAÇÃO
# =========================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Acurácia no teste: {accuracy:.4f}")
print(f"Loss: {loss:.4f}")

# =========================
# PREDIÇÃO
# =========================
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# =========================
# RELATÓRIO
# =========================
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))