# T2: Naive Bayes gaussiano vs. regresión logística (atributos estandarizados)
# Métricas en prueba: accuracy y F1 macro. División 70/30 estratificada, random_state=42.

import json

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# ------------------------------------------------------------------
# 1) Datos: Breast Cancer Wisconsin (Diagnostic)
# ------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target

# ------------------------------------------------------------------
# 2) División única 70/30 estratificada por la clase, random_state=42
#    (el conjunto de prueba no se usa para ajustar nada)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# ------------------------------------------------------------------
# 3) Modelo 1: Naive Bayes gaussiano (sin estandarización)
# ------------------------------------------------------------------
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

acc_gnb = float(accuracy_score(y_test, y_pred_gnb))
f1_gnb = float(f1_score(y_test, y_pred_gnb, average="macro"))

# ------------------------------------------------------------------
# 4) Modelo 2: regresión logística con atributos estandarizados
#    (el escalador se ajusta SOLO con el entrenamiento)
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)   # ajuste solo con train
X_test_std = scaler.transform(X_test)         # transformación de test

logreg = LogisticRegression(max_iter=5000, random_state=42)
logreg.fit(X_train_std, y_train)
y_pred_lr = logreg.predict(X_test_std)

acc_lr = float(accuracy_score(y_test, y_pred_lr))
f1_lr = float(f1_score(y_test, y_pred_lr, average="macro"))

# ------------------------------------------------------------------
# 5) Tabla resumen (también se guarda en resultados.json)
# ------------------------------------------------------------------
tabla = [
    {
        "modelo": "naive_bayes_gaussiano",
        "accuracy": acc_gnb,
        "f1_macro": f1_gnb,
    },
    {
        "modelo": "regresion_logistica_estandarizada",
        "accuracy": acc_lr,
        "f1_macro": f1_lr,
    },
]

resultados = {
    "n_ejemplos_train": int(X_train.shape[0]),
    "n_ejemplos_test": int(X_test.shape[0]),
    "naive_bayes_gaussiano": {
        "accuracy": acc_gnb,
        "f1_macro": f1_gnb,
    },
    "regresion_logistica_estandarizada": {
        "accuracy": acc_lr,
        "f1_macro": f1_lr,
    },
    "tabla_prueba": tabla,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ------------------------------------------------------------------
# 6) Resumen breve por consola
# ------------------------------------------------------------------
print("Subtarea T2 — Generativo vs. discriminativo (prueba 70/30, random_state=42)")
print(f"Ejemplos: train={X_train.shape[0]}, test={X_test.shape[0]}")
print("-" * 60)
print(f"{'Modelo':<38}{'Accuracy':>10}{'F1 macro':>10}")
print("-" * 60)
print(f"{'Naive Bayes gaussiano':<38}{acc_gnb:>10.4f}{f1_gnb:>10.4f}")
print(f"{'Regresión logística (estand.)':<38}{acc_lr:>10.4f}{f1_lr:>10.4f}")
print("-" * 60)
print("Resultados escritos en resultados.json")
