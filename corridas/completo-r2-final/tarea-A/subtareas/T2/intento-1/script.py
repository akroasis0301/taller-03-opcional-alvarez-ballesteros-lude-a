# -*- coding: utf-8 -*-
"""
T2 — Dos clasificadores sobre la división 70/30 de la Parte 1 (Breast Cancer Wisconsin).

Entrena:
  1) Naive Bayes gaussiano (sobre los atributos originales; la estandarización no
     es necesaria para NB y no altera sus predicciones).
  2) Regresión logística con atributos estandarizados, donde el StandardScaler se
     ajusta ÚNICAMENTE con el conjunto de entrenamiento (Pipeline).

Evalúa accuracy y F1 macro de cada modelo sobre el conjunto de prueba y presenta
los resultados en una tabla (JSON + PNG + tabla Markdown para el reporte).

Entrada: entrada/T1/{train,test}.parquet (o .csv) generados en la subtarea T1.
Salidas: resultados.json, tabla_parte2_metricas.png, comparativa_parte2_barras.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42
DIR_ENTRADA = Path("entrada/T1")


# ----------------------------------------------------------------------
# 1. Cargar la división única de la Parte 1
# ----------------------------------------------------------------------
def leer_split(nombre: str) -> pd.DataFrame:
    for ext in ("parquet", "csv"):
        ruta = DIR_ENTRADA / f"{nombre}.{ext}"
        if ruta.exists():
            if ext == "parquet":
                return pd.read_parquet(ruta)
            return pd.read_csv(ruta)
    raise FileNotFoundError(
        f"No se encontró '{nombre}.parquet' ni '{nombre}.csv' en {DIR_ENTRADA}/"
    )


train_df = leer_split("train")
test_df = leer_split("test")

# Detectar la columna objetivo
if "target" in train_df.columns:
    target_col = "target"
else:
    target_col = train_df.columns[-1]

feature_cols = [c for c in train_df.columns if c != target_col]

X_train = train_df[feature_cols].to_numpy(dtype=float)
y_train = train_df[target_col].to_numpy()
X_test = test_df[feature_cols].to_numpy(dtype=float)
y_test = test_df[target_col].to_numpy()

n_train, n_test = X_train.shape[0], X_test.shape[0]
n_features = X_train.shape[1]

# ----------------------------------------------------------------------
# 2. Modelo 1: Naive Bayes gaussiano
# ----------------------------------------------------------------------
nb = GaussianNB()
nb.fit(X_train, y_train)
pred_nb = nb.predict(X_test)
acc_nb = float(accuracy_score(y_test, pred_nb))
f1_nb = float(f1_score(y_test, pred_nb, average="macro"))

# ----------------------------------------------------------------------
# 3. Modelo 2: Regresión logística con estandarización (scaler solo con train)
#    El Pipeline garantiza que el StandardScaler se ajuste únicamente con los
#    datos de entrenamiento y se aplique (fijo) al conjunto de prueba.
# ----------------------------------------------------------------------
logreg = Pipeline(
    steps=[
        ("escalador", StandardScaler()),
        (
            "regresion_logistica",
            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        ),
    ]
)
logreg.fit(X_train, y_train)
pred_lr = logreg.predict(X_test)
acc_lr = float(accuracy_score(y_test, pred_lr))
f1_lr = float(f1_score(y_test, pred_lr, average="macro"))

# ----------------------------------------------------------------------
# 4. Tabla de resultados
# ----------------------------------------------------------------------
filas = [
    {
        "modelo": "Naive Bayes gaussiano",
        "accuracy_prueba": acc_nb,
        "f1_macro_prueba": f1_nb,
    },
    {
        "modelo": "Regresión logística (atributos estandarizados)",
        "accuracy_prueba": acc_lr,
        "f1_macro_prueba": f1_lr,
    },
]

tabla_markdown = (
    "| Modelo | Accuracy (prueba) | F1 macro (prueba) |\n"
    "|---|---|---|\n"
    f"| Naive Bayes gaussiano | {acc_nb:.4f} | {f1_nb:.4f} |\n"
    f"| Regresión logística (atributos estandarizados) | {acc_lr:.4f} | {f1_lr:.4f} |"
)

# --- Figura 1: tabla como PNG ---
fig, ax = plt.subplots(figsize=(8.2, 2.8))
ax.axis("off")
col_labels = ["Modelo", "Accuracy (prueba)", "F1 macro (prueba)"]
cell_text = [
    ["Naive Bayes gaussiano", f"{acc_nb:.4f}", f"{f1_nb:.4f}"],
    ["Regresión logística\n(atributos estandarizados)", f"{acc_lr:.4f}", f"{f1_lr:.4f}"],
]
tabla = ax.table(
    cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center"
)
tabla.auto_set_font_size(False)
tabla.set_fontsize(11)
tabla.scale(1, 1.9)
for j in range(len(col_labels)):
    tabla[0, j].set_facecolor("#d9e2f3")
    tabla[0, j].set_text_props(weight="bold")
ax.set_title(
    "Parte 2 — Métricas en el conjunto de prueba\n"
    f"(Breast Cancer, división 70/30 estratificada, random_state={RANDOM_STATE})",
    fontsize=11,
)
plt.savefig("tabla_parte2_metricas.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# --- Figura 2: barras comparativas ---
fig, ax = plt.subplots(figsize=(6.4, 4.2))
x = np.arange(2)
ancho = 0.35
b1 = ax.bar(x - ancho / 2, [acc_nb, acc_lr], ancho, label="Accuracy", color="#4c72b0")
b2 = ax.bar(x + ancho / 2, [f1_nb, f1_lr], ancho, label="F1 macro", color="#dd8452")
for barras in (b1, b2):
    for b in barras:
        ax.annotate(
            f"{b.get_height():.4f}",
            xy=(b.get_x() + b.get_width() / 2, b.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
ax.set_xticks(x)
ax.set_xticklabels(["Naive Bayes\ngaussiano", "Regresión\nlogística"])
ax.set_ylim(0, 1.08)
ax.set_ylabel("Valor en prueba")
ax.set_title("Parte 2 — Comparación de clasificadores (conjunto de prueba)")
ax.legend(loc="lower right")
plt.savefig("comparativa_parte2_barras.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------
# 5. resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T2",
    "descripcion": (
        "Naive Bayes gaussiano y regresión logística (atributos estandarizados, "
        "escalador ajustado solo con el entrenamiento) evaluados con accuracy y "
        "F1 macro sobre el conjunto de prueba de la división 70/30 de la Parte 1."
    ),
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "division_usada": {
        "fuente": "entrada/T1 (división única de la Parte 1)",
        "test_size": 0.3,
        "estratificada": True,
        "random_state": RANDOM_STATE,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_atributos": int(n_features),
    },
    "modelos": {
        "naive_bayes_gaussiano": {
            "tipo": "generativo",
            "estandarizacion": "no requerida (invariante a transformaciones afines por atributo)",
            "accuracy_prueba": acc_nb,
            "f1_macro_prueba": f1_nb,
        },
        "regresion_logistica": {
            "tipo": "discriminativo",
            "estandarizacion": "StandardScaler ajustado solo con el conjunto de entrenamiento",
            "accuracy_prueba": acc_lr,
            "f1_macro_prueba": f1_lr,
        },
    },
    "tabla": filas,
    "tabla_markdown": tabla_markdown,
    "archivos_generados": [
        "tabla_parte2_metricas.png",
        "comparativa_parte2_barras.png",
    ],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 6. Resumen
# ----------------------------------------------------------------------
print("=" * 64)
print("T2 — Naive Bayes gaussiano vs. Regresión logística (prueba)")
print("=" * 64)
print(f"División usada: train={n_train}, test={n_test}, atributos={n_features}")
print("-" * 64)
print(f"{'Modelo':<44}{'Acc':>9}{'F1 macro':>11}")
print(f"{'Naive Bayes gaussiano':<44}{acc_nb:>9.4f}{f1_nb:>11.4f}")
print(f"{'Regresión logística (estandarizada)':<44}{acc_lr:>9.4f}{f1_lr:>11.4f}")
print("-" * 64)
print("Archivos: resultados.json, tabla_parte2_metricas.png, comparativa_parte2_barras.png")
