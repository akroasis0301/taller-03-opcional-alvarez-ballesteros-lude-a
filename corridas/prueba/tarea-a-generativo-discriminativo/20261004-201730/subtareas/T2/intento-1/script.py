# -*- coding: utf-8 -*-
"""
T2 — Generativo contra discriminativo (Parte 2: dos clasificadores)
Entrena un Naive Bayes gaussiano y una regresión logística (con estandarización
ajustada SOLO con el entrenamiento) sobre la división 70/30 estratificada de T1,
y mide accuracy y F1 macro de cada uno sobre el conjunto de prueba, en una tabla.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

RANDOM_STATE = 42

# ------------------------------------------------------------------
# 1) Recuperar la división única de T1 (train_indices / test_indices)
# ------------------------------------------------------------------
t1_json = Path("entrada") / "T1" / "resultados.json"
train_idx, test_idx = None, None
t1_info = None
if t1_json.exists():
    with open(t1_json, "r", encoding="utf-8") as f:
        t1_info = json.load(f)
    div = t1_info.get("division", {}) if isinstance(t1_info, dict) else {}
    train_idx = div.get("train_indices")
    test_idx = div.get("test_indices")

# ------------------------------------------------------------------
# 2) Cargar el mismo dataset de T1 (incluido en scikit-learn, sin red)
# ------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target
nombres_clase = list(data.target_names)  # ['malignant', 'benign']

if train_idx is None or test_idx is None:
    # Respaldo: reproducir exactamente la división de T1
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )

train_idx = np.asarray(train_idx, dtype=int)
test_idx = np.asarray(test_idx, dtype=int)

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
n_train, n_test = len(y_train), len(y_test)

# Verificación suave contra lo reportado por T1
if isinstance(t1_info, dict):
    div = t1_info.get("division", {})
    if div.get("n_train") is not None and int(div["n_train"]) != n_train:
        print(f"[aviso] n_train ({n_train}) difiere de T1 ({div['n_train']})")
    if div.get("n_test") is not None and int(div["n_test"]) != n_test:
        print(f"[aviso] n_test ({n_test}) difiere de T1 ({div['n_test']})")

# ------------------------------------------------------------------
# 3) Modelos
#    - GaussianNB sobre los atributos crudos (no requiere estandarización)
#    - Regresión logística dentro de un Pipeline con StandardScaler ajustado
#      únicamente con el entrenamiento (el prueba nunca participa en el ajuste)
# ------------------------------------------------------------------
nb = GaussianNB()
nb.fit(X_train, y_train)
pred_nb = nb.predict(X_test)

logreg = Pipeline([
    ("escalador", StandardScaler()),  # se ajusta solo con X_train al hacer fit
    ("modelo", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
])
logreg.fit(X_train, y_train)
pred_lr = logreg.predict(X_test)

# ------------------------------------------------------------------
# 4) Métricas en prueba y tabla
# ------------------------------------------------------------------
filas = []
for nombre, pred in [
    ("Naive Bayes gaussiano", pred_nb),
    ("Regresión logística", pred_lr),
]:
    filas.append({
        "modelo": nombre,
        "accuracy_prueba": float(accuracy_score(y_test, pred)),
        "f1_macro_prueba": float(f1_score(y_test, pred, average="macro")),
    })

tabla = pd.DataFrame(filas)
tabla.to_csv("T2_tabla_metricas.csv", index=False, encoding="utf-8")

# ------------------------------------------------------------------
# 5) resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T2",
    "descripcion": (
        "Accuracy y F1 macro en el conjunto de prueba para Naive Bayes gaussiano y "
        "regresión logística (atributos estandarizados, escalador ajustado solo con "
        "el entrenamiento), sobre la división 70/30 estratificada de T1."
    ),
    "division_usada": {
        "fuente": "entrada/T1/resultados.json (train_indices / test_indices)",
        "test_size": 0.3,
        "estratificada": True,
        "random_state": RANDOM_STATE,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "conteo_por_clase_train": {
            "malignant": int((y_train == 0).sum()),
            "benign": int((y_train == 1).sum()),
        },
        "conteo_por_clase_test": {
            "malignant": int((y_test == 0).sum()),
            "benign": int((y_test == 1).sum()),
        },
    },
    "tabla_metricas": filas,
    "accuracy_naive_bayes_gaussiano": filas[0]["accuracy_prueba"],
    "f1_macro_naive_bayes_gaussiano": filas[0]["f1_macro_prueba"],
    "accuracy_regresion_logistica": filas[1]["accuracy_prueba"],
    "f1_macro_regresion_logistica": filas[1]["f1_macro_prueba"],
    "matriz_confusion_naive_bayes_gaussiano": confusion_matrix(y_test, pred_nb).tolist(),
    "matriz_confusion_regresion_logistica": confusion_matrix(y_test, pred_lr).tolist(),
    "notas": (
        "La estandarización (StandardScaler) se ajusta únicamente con el conjunto de "
        "entrenamiento dentro de un Pipeline; el conjunto de prueba no se usó para "
        "ajustar nada (ni escalador ni hiperparámetros)."
    ),
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# 6) Figura: comparación de métricas en prueba
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
xpos = np.arange(len(filas))
width = 0.35
acc = [r["accuracy_prueba"] for r in filas]
f1m = [r["f1_macro_prueba"] for r in filas]
b1 = ax.bar(xpos - width / 2, acc, width, label="Accuracy", color="#4C72B0")
b2 = ax.bar(xpos + width / 2, f1m, width, label="F1 macro", color="#DD8452")
ax.set_xticks(xpos)
ax.set_xticklabels([r["modelo"] for r in filas])
ax.set_ylim(0, 1.08)
ax.set_ylabel("Valor en el conjunto de prueba")
ax.set_title("T2: desempeño en prueba (división 70/30 de T1)")
for barras in (b1, b2):
    for b in barras:
        ax.annotate(f"{b.get_height():.3f}",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("T2_metricas_modelos.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 7) Resumen
# ------------------------------------------------------------------
print("División de T1: n_train =", n_train, "| n_test =", n_test)
print("\nTabla de métricas en prueba:")
print(tabla.to_string(index=False))
print("\nArchivos generados: resultados.json, T2_tabla_metricas.csv, T2_metricas_modelos.png")
