#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2: Entrenar sobre la división de T1 un Naive Bayes gaussiano y una regresión
logística (StandardScaler ajustado solo con el entrenamiento) y evaluar ambos
sobre el conjunto de prueba, produciendo una tabla con accuracy y F1 macro.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ----------------------------------------------------------------------
# 1) Recuperar la división única de T1 (índices train/test)
# ----------------------------------------------------------------------
RUTA_T1 = "entrada/T1/resultados.json"
with open(RUTA_T1, "r", encoding="utf-8") as f:
    t1 = json.load(f)

idx_train = np.asarray(t1["indices_train"], dtype=int)
idx_test = np.asarray(t1["indices_test"], dtype=int)

# ----------------------------------------------------------------------
# 2) Cargar el dataset y reconstruir exactamente la división de T1
# ----------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target

assert len(idx_train) == t1["n_train"], "n_train no coincide con T1"
assert len(idx_test) == t1["n_test"], "n_test no coincide con T1"

X_train, y_train = X[idx_train], y[idx_train]
X_test, y_test = X[idx_test], y[idx_test]

# ----------------------------------------------------------------------
# 3) Modelo 1: Naive Bayes gaussiano (atributos crudos, sin escalado)
# ----------------------------------------------------------------------
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

# ----------------------------------------------------------------------
# 4) Modelo 2: Regresión logística con StandardScaler ajustado SOLO con train
# ----------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit solo con entrenamiento
X_test_scaled = scaler.transform(X_test)        # solo transform en prueba

logreg = LogisticRegression(max_iter=5000, random_state=42)
logreg.fit(X_train_scaled, y_train)
y_pred_lr = logreg.predict(X_test_scaled)

# ----------------------------------------------------------------------
# 5) Métricas sobre el conjunto de prueba
# ----------------------------------------------------------------------
acc_gnb = float(accuracy_score(y_test, y_pred_gnb))
f1_gnb = float(f1_score(y_test, y_pred_gnb, average="macro"))
acc_lr = float(accuracy_score(y_test, y_pred_lr))
f1_lr = float(f1_score(y_test, y_pred_lr, average="macro"))

tabla = pd.DataFrame({
    "modelo": ["GaussianNB", "Regresion_logistica"],
    "accuracy_test": [acc_gnb, acc_lr],
    "f1_macro_test": [f1_gnb, f1_lr],
})

cm_gnb = confusion_matrix(y_test, y_pred_gnb).tolist()
cm_lr = confusion_matrix(y_test, y_pred_lr).tolist()

# ----------------------------------------------------------------------
# 6) Contrato: resultados.json con la tabla de métricas
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T2_nb_gaussiano_vs_regresion_logistica",
    "division_usada": (
        "División única de T1: train/test 70/30 estratificada, random_state=42 "
        "(índices leídos de entrada/T1/resultados.json)"
    ),
    "n_train": int(len(idx_train)),
    "n_test": int(len(idx_test)),
    "modelos": {
        "gaussian_nb": {
            "preprocesamiento": "ninguno (atributos crudos)",
            "accuracy_test": acc_gnb,
            "f1_macro_test": f1_gnb,
            "matriz_confusion_test": cm_gnb,
        },
        "regresion_logistica": {
            "preprocesamiento": "StandardScaler ajustado solo con el conjunto de entrenamiento",
            "parametros": {"max_iter": 5000, "random_state": 42, "solver": "lbfgs"},
            "accuracy_test": acc_lr,
            "f1_macro_test": f1_lr,
            "matriz_confusion_test": cm_lr,
        },
    },
    "tabla_metricas": tabla.to_dict(orient="records"),
    "tabla_metricas_por_modelo": {
        "GaussianNB": {"accuracy_test": acc_gnb, "f1_macro_test": f1_gnb},
        "Regresion_logistica": {"accuracy_test": acc_lr, "f1_macro_test": f1_lr},
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# Tabla también en CSV (valores sin redondear)
tabla.to_csv("T2_tabla_metricas.csv", index=False)

# ----------------------------------------------------------------------
# 7) Figura: tabla de métricas en PNG
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.8, 2.6))
ax.axis("off")
celdas = [
    [fila["modelo"],
     f"{fila['accuracy_test']:.4f}",
     f"{fila['f1_macro_test']:.4f}"]
    for fila in tabla.to_dict(orient="records")
]
tbl = ax.table(
    cellText=celdas,
    colLabels=["Modelo", "Accuracy (prueba)", "F1 macro (prueba)"],
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 1.7)
ax.set_title("T2: GaussianNB vs Regresión logística — métricas en prueba", fontsize=12)
plt.tight_layout()
plt.savefig("T2_tabla_metricas.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 8) Resumen breve
# ----------------------------------------------------------------------
print("T2 completada: NB gaussiano y regresión logística evaluados en prueba.")
print(f"División T1: n_train={len(idx_train)}, n_test={len(idx_test)}")
print(tabla.to_string(index=False))
print("Archivos generados: resultados.json, T2_tabla_metricas.csv, T2_tabla_metricas.png")
