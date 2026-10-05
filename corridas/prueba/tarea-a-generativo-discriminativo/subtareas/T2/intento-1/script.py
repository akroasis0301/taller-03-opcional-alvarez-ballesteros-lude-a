# T2: Entrenar GaussianNB y Regresión Logística (atributos estandarizados) sobre la
# división 70/30 estratificada fija de T1 y evaluar accuracy y F1 macro en prueba.

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
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

# ------------------------------------------------------------------
# 1) Datos y recuperación de la división fija de T1
# ------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target

with open("entrada/T1/resultados.json", "r", encoding="utf-8") as f:
    t1 = json.load(f)

idx_train = np.asarray(t1["indices_train"], dtype=int)
idx_test = np.asarray(t1["indices_test"], dtype=int)

# Verificación cruzada con el artefacto persistido de T1 (si existe y es legible)
try:
    npz = np.load("entrada/T1/t1_split.npz")
    claves = set(npz.files)
    if {"indices_train", "indices_test"} <= claves:
        assert np.array_equal(np.asarray(npz["indices_train"], dtype=int), idx_train)
        assert np.array_equal(np.asarray(npz["indices_test"], dtype=int), idx_test)
        print("División verificada contra entrada/T1/t1_split.npz")
except FileNotFoundError:
    print("t1_split.npz no encontrado; se usan los índices de entrada/T1/resultados.json")

X_train, y_train = X[idx_train], y[idx_train]
X_test, y_test = X[idx_test], y[idx_test]
assert X_train.shape[0] == 398 and X_test.shape[0] == 171  # 70/30 de T1

# ------------------------------------------------------------------
# 2) Modelos
#    - GaussianNB sobre los atributos crudos de entrenamiento.
#    - Regresión logística dentro de un Pipeline con StandardScaler:
#      el escalador se ajusta ÚNICAMENTE con los datos de entrenamiento
#      (el conjunto de prueba solo se transforma con el escalador ya ajustado).
# ------------------------------------------------------------------
gnb = GaussianNB()

logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000, random_state=42)),
])

gnb.fit(X_train, y_train)
logreg.fit(X_train, y_train)

# ------------------------------------------------------------------
# 3) Evaluación sobre el conjunto de prueba (nunca usado para ajustar)
# ------------------------------------------------------------------
filas = []
for nombre, modelo in [("Naive Bayes gaussiano", gnb),
                       ("Regresión logística", logreg)]:
    y_pred = modelo.predict(X_test)
    filas.append({
        "modelo": nombre,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "F1 macro": float(f1_score(y_test, y_pred, average="macro")),
    })

tabla = pd.DataFrame(filas).set_index("modelo")[["accuracy", "F1 macro"]]

# ------------------------------------------------------------------
# 4) Figura comparativa
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
xpos = np.arange(len(tabla.index))
width = 0.35
b1 = ax.bar(xpos - width / 2, tabla["accuracy"], width, label="accuracy", color="#4C72B0")
b2 = ax.bar(xpos + width / 2, tabla["F1 macro"], width, label="F1 macro", color="#DD8452")
for barras in (b1, b2):
    for b in barras:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(xpos)
ax.set_xticklabels(tabla.index)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Puntaje en el conjunto de prueba")
ax.set_title("T2: GaussianNB vs Regresión logística (split 70/30 de T1, random_state=42)")
ax.legend(loc="lower right")
fig.tight_layout()
plt.savefig("t2_metricas_modelos.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 5) Contrato de resultados
# ------------------------------------------------------------------
salida = {
    "subtarea": "T2_gaussiannb_vs_logreg",
    "split_usado": "T1: 70/30 estratificado por clase, random_state=42",
    "n_train": int(X_train.shape[0]),
    "n_test": int(X_test.shape[0]),
    "escalador": "StandardScaler ajustado solo con datos de entrenamiento (Pipeline)",
    "tabla": filas,  # 2 filas: Naive Bayes gaussiano y Regresión logística; columnas accuracy y F1 macro
    "tabla_dict": {
        f["modelo"]: {"accuracy": f["accuracy"], "F1 macro": f["F1 macro"]} for f in filas
    },
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(salida, f, indent=2, ensure_ascii=False)

tabla.to_csv("t2_tabla_modelos.csv")

# ------------------------------------------------------------------
# 6) Resumen
# ------------------------------------------------------------------
print("\n=== T2: Resultados en el conjunto de prueba (n_test=171) ===")
print(tabla.to_string(float_format=lambda v: f"{v:.6f}"))
print("\nArchivos generados: resultados.json, t2_tabla_modelos.csv, t2_metricas_modelos.png")
