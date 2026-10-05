#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2 — Dos clasificadores sobre la división 70/30 de T1 (Breast Cancer Wisconsin).

Entrena:
  1) Naive Bayes gaussiano (GaussianNB)
  2) Regresión logística (LogisticRegression)
ambos sobre atributos estandarizados con StandardScaler ajustado SOLO con el
conjunto de entrenamiento, y reporta accuracy y F1 macro de cada uno sobre el
conjunto de prueba, en una tabla.

Salidas:
  - resultados.json            (contrato: todas las cifras de la subtarea)
  - t2_tabla_metricas.csv      (tabla de métricas)
  - t2_tabla_metricas.png      (tabla como figura)
  - t2_tabla_metricas.md       (tabla en Markdown para el reporte)
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ----------------------------------------------------------------------------
# 1) Cargar la división de T1 (train/test 70/30 estratificada, random_state=42)
# ----------------------------------------------------------------------------
def cargar_split_t1():
    d = Path("entrada") / "T1"
    nombres = ["X_train", "X_test", "y_train", "y_test"]
    rutas_npy = {n: d / f"t1_{n}.npy" for n in nombres}
    if all(p.exists() for p in rutas_npy.values()):
        datos = tuple(np.load(rutas_npy[n]) for n in nombres)
        return datos[0], datos[1], datos[2], datos[3], "archivos .npy de T1"

    # Respaldo: archivo .npz de T1
    npz_path = d / "t1_split_breast_cancer.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"No se encontraron ni los .npy ni {npz_path} con la división de T1."
        )
    z = np.load(npz_path)
    claves = list(z.keys())

    def buscar(*candidatos):
        for c in candidatos:
            for k in claves:
                if c.lower() in k.lower():
                    return z[k]
        raise KeyError(f"No se encontró ninguna de {candidatos} en {claves}")

    X_tr = buscar("X_train", "xtrain")
    X_te = buscar("X_test", "xtest")
    y_tr = buscar("y_train", "ytrain")
    y_te = buscar("y_test", "ytest")
    return X_tr, X_te, y_tr, y_te, f"archivo {npz_path.as_posix()}"


X_train, X_test, y_train, y_test, origen_split = cargar_split_t1()

n_train, n_test = X_train.shape[0], X_test.shape[0]
clases_unicas = np.unique(np.concatenate([y_train, y_test]))

# ----------------------------------------------------------------------------
# 2) Estandarización: StandardScaler ajustado SOLO con el entrenamiento
# ----------------------------------------------------------------------------
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)   # fit + transform SOLO en train
X_test_std = scaler.transform(X_test)         # el test solo se transforma

# ----------------------------------------------------------------------------
# 3) Entrenar los dos clasificadores y evaluar en prueba
# ----------------------------------------------------------------------------
modelos = {
    "Naive Bayes gaussiano (GaussianNB)": GaussianNB(),
    "Regresión logística": LogisticRegression(max_iter=10000, random_state=42),
}

filas = []
detalles = {}
for nombre, modelo in modelos.items():
    modelo.fit(X_train_std, y_train)          # ajuste solo con entrenamiento
    y_pred = modelo.predict(X_test_std)       # evaluación en prueba
    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))
    cm = confusion_matrix(y_test, y_pred)
    filas.append({"modelo": nombre, "accuracy": acc, "f1_macro": f1m})
    detalles[nombre] = {
        "accuracy": acc,
        "f1_macro": f1m,
        "matriz_confusion": cm.tolist(),
        "n_test": int(n_test),
    }

tabla = pd.DataFrame(filas, columns=["modelo", "accuracy", "f1_macro"])

# ----------------------------------------------------------------------------
# 4) Guardar la tabla (CSV, PNG y Markdown)
# ----------------------------------------------------------------------------
tabla.to_csv("t2_tabla_metricas.csv", index=False)

# Figura: tabla de métricas
fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.axis("off")
celdas = [
    [r["modelo"], f"{r['accuracy']:.4f}", f"{r['f1_macro']:.4f}"] for _, r in tabla.iterrows()
]
tbl = ax.table(
    cellText=celdas,
    colLabels=["Modelo", "Accuracy (prueba)", "F1 macro (prueba)"],
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.0, 1.6)
for j in range(3):
    tbl[0, j].set_facecolor("#d9e2f3")
    tbl[0, j].set_text_props(weight="bold")
ax.set_title(
    "T2 — Accuracy y F1 macro en prueba (división 70/30 de T1, escalador ajustado solo con train)",
    fontsize=10, pad=12,
)
plt.savefig("t2_tabla_metricas.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# Tabla en Markdown para el reporte
lineas_md = [
    "| Modelo | Accuracy (prueba) | F1 macro (prueba) |",
    "|---|---|---|",
]
for _, r in tabla.iterrows():
    lineas_md.append(f"| {r['modelo']} | {r['accuracy']:.4f} | {r['f1_macro']:.4f} |")
Path("t2_tabla_metricas.md").write_text("\n".join(lineas_md) + "\n", encoding="utf-8")

# ----------------------------------------------------------------------------
# 5) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------------
resultados = {
    "subtarea": "T2",
    "descripcion": (
        "Naive Bayes gaussiano y regresión logística entrenados sobre la división "
        "70/30 estratificada de T1 (random_state=42), con atributos estandarizados "
        "por un StandardScaler ajustado únicamente con el conjunto de entrenamiento; "
        "accuracy y F1 macro medidos sobre el conjunto de prueba."
    ),
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "division_usada": {
        "origen": origen_split,
        "split": "70/30 estratificada por clase, random_state=42 (creada en T1)",
        "tam_train": int(n_train),
        "tam_test": int(n_test),
        "numero_atributos": int(X_train.shape[1]),
        "clases": [str(c) for c in clases_unicas],
    },
    "escalador": {
        "tipo": "StandardScaler",
        "ajustado_con": "solo conjunto de entrenamiento",
        "test_usado_para_ajustar": False,
        "medias": scaler.mean_.tolist(),
        "escalas": scaler.scale_.tolist(),
    },
    "tabla": filas,
    "metricas_por_modelo": detalles,
    "archivos_guardados": [
        "t2_tabla_metricas.csv",
        "t2_tabla_metricas.png",
        "t2_tabla_metricas.md",
    ],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# 6) Resumen
# ----------------------------------------------------------------------------
print("=== T2: dos clasificadores sobre la división 70/30 de T1 ===")
print(f"Origen del split: {origen_split}")
print(f"Train: {n_train} ejemplos | Test: {n_test} ejemplos | Atributos: {X_train.shape[1]}")
print("Escalador: StandardScaler ajustado SOLO con el entrenamiento.")
print("\nTabla de métricas (conjunto de prueba):")
print(tabla.to_string(index=False))
print("\nArchivos guardados: resultados.json, t2_tabla_metricas.csv, "
      "t2_tabla_metricas.png, t2_tabla_metricas.md")
