#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 — Curva de aprendizaje: Naive Bayes (generativo) vs. regresión logística
(discriminativo).

Sobre la MISMA división 70/30 estratificada de T1 (random_state=42), entrena
ambos modelos con submuestras estratificadas del 5 %, 10 %, 25 %, 50 % y 100 %
del conjunto de entrenamiento (random_state=42) y evalúa cada tamaño siempre
sobre el MISMO conjunto de prueba. Guarda la figura curva_aprendizaje.png con
una curva de exactitud por modelo y escribe resultados.json con los valores
numéricos de exactitud de cada punto de la curva.
"""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
FRACCIONES = [0.05, 0.10, 0.25, 0.50, 1.00]
ARCHIVO_FIGURA = "curva_aprendizaje.png"
ARCHIVO_RESULTADOS = "resultados.json"

# --------------------------------------------------------------------- #
# 1) Datos y MISMA división 70/30 estratificada de T1 (random_state=42)
# --------------------------------------------------------------------- #
data = load_breast_cancer()
X, y = data.data, data.target
nombres_clases = [str(c) for c in data.target_names]  # 0 = malignant, 1 = benign

try:
    with open("entrada/T1/resultados.json", "r", encoding="utf-8") as f:
        t1 = json.load(f)
    idx_train = np.asarray(t1["indices_train"], dtype=int)
    idx_test = np.asarray(t1["indices_test"], dtype=int)
    origen_division = "índices de la división 70/30 leídos de entrada/T1/resultados.json"
except Exception:
    # Respaldo: reproduce exactamente la división de la Parte 1 de T1
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    origen_division = ("división recomputada de forma idéntica a T1: "
                       "train_test_split(test_size=0.30, stratify=y, random_state=42)")

X_train, y_train = X[idx_train], y[idx_train]
X_test, y_test = X[idx_test], y[idx_test]  # conjunto de prueba FIJO para todos los puntos

# --------------------------------------------------------------------- #
# 2) Submuestras estratificadas del entrenamiento (random_state=42)
# --------------------------------------------------------------------- #
def submuestra_estratificada(frac):
    """Índices (posiciones dentro del train de T1) de una submuestra
    estratificada con la fracción `frac` del conjunto de entrenamiento."""
    if frac >= 1.0:
        return np.arange(len(y_train))
    idx_sub, _ = train_test_split(
        np.arange(len(y_train)),
        train_size=frac,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    return np.sort(idx_sub)

# --------------------------------------------------------------------- #
# 3) Entrenar por tamaño y evaluar siempre sobre el mismo conjunto de prueba
# --------------------------------------------------------------------- #
curva = []
for frac in FRACCIONES:
    idx_sub = submuestra_estratificada(frac)
    Xs, ys = X_train[idx_sub], y_train[idx_sub]
    n_sub = int(len(idx_sub))
    conteo_clases = {nombres_clases[c]: int(np.sum(ys == c)) for c in (0, 1)}

    # Modelo generativo: Naive Bayes gaussiano
    nb = GaussianNB()
    nb.fit(Xs, ys)
    acc_nb = float(accuracy_score(y_test, nb.predict(X_test)))

    # Modelo discriminativo: regresión logística
    # (el escalador se ajusta SOLO con la submuestra de entrenamiento, nunca con la prueba)
    lr = Pipeline([
        ("escalador", StandardScaler()),
        ("reglog", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ])
    lr.fit(Xs, ys)
    acc_lr = float(accuracy_score(y_test, lr.predict(X_test)))

    curva.append({
        "fraccion": frac,
        "n_entrenamiento": n_sub,
        "n_por_clase_entrenamiento": conteo_clases,
        "indices_submuestra_en_train": idx_sub.tolist(),
        "exactitud_prueba_naive_bayes": acc_nb,
        "exactitud_prueba_regresion_logistica": acc_lr,
    })

ns = [p["n_entrenamiento"] for p in curva]
accs_nb = [p["exactitud_prueba_naive_bayes"] for p in curva]
accs_lr = [p["exactitud_prueba_regresion_logistica"] for p in curva]

# --------------------------------------------------------------------- #
# 4) Figura: exactitud en prueba vs. número de ejemplos de entrenamiento
# --------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(8.0, 5.5))
ax.plot(ns, accs_nb, marker="o", color="tab:blue", label="Naive Bayes (generativo)")
ax.plot(ns, accs_lr, marker="s", color="tab:orange",
        label="Regresión logística (discriminativo)")

for n_pt, a in zip(ns, accs_nb):
    ax.annotate(f"{a:.3f}", (n_pt, a), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, color="tab:blue")
for n_pt, a in zip(ns, accs_lr):
    ax.annotate(f"{a:.3f}", (n_pt, a), textcoords="offset points",
                xytext=(0, -14), ha="center", fontsize=8, color="tab:orange")

ax.set_xticks(ns)
ax.set_xlabel("Número de ejemplos de entrenamiento")
ax.set_ylabel("Exactitud en el conjunto de prueba")
ax.set_title("Curva de aprendizaje: NB vs. regresión logística\n"
             f"Breast Cancer Wisconsin — prueba fija (n={len(y_test)}), "
             "división 70/30 de T1")
ax.set_ylim(min(min(accs_nb), min(accs_lr)) - 0.06, 1.01)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(ARCHIVO_FIGURA, dpi=120)
plt.close(fig)

# --------------------------------------------------------------------- #
# 5) resultados.json (contrato de la subtarea)
# --------------------------------------------------------------------- #
resultados = {
    "subtarea": "T3_curva_de_aprendizaje",
    "descripcion": ("Curvas de aprendizaje de GaussianNB y regresión logística sobre la "
                    "misma división 70/30 estratificada de T1; submuestras estratificadas "
                    "del 5 %, 10 %, 25 %, 50 % y 100 % del entrenamiento "
                    "(random_state=42); cada tamaño se evalúa siempre sobre el mismo "
                    "conjunto de prueba."),
    "origen_division": origen_division,
    "n_train_total": int(len(y_train)),
    "n_test": int(len(y_test)),
    "fracciones": FRACCIONES,
    "tamanos_entrenamiento": ns,
    "modelos": {
        "naive_bayes": "GaussianNB()",
        "regresion_logistica": ("Pipeline(StandardScaler -> LogisticRegression"
                                "(max_iter=5000, random_state=42)); el escalador se "
                                "ajusta solo con los datos de entrenamiento de cada punto"),
    },
    "exactitud_prueba_naive_bayes": accs_nb,
    "exactitud_prueba_regresion_logistica": accs_lr,
    "curva_aprendizaje": curva,
    "figura": ARCHIVO_FIGURA,
}

with open(ARCHIVO_RESULTADOS, "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# --------------------------------------------------------------------- #
# 6) Resumen breve
# --------------------------------------------------------------------- #
print("T3 — Curva de aprendizaje (NB vs. regresión logística)")
print(f"División usada: {origen_division}")
print(f"Entrenamiento total: {len(y_train)} ejemplos | Prueba fija: {len(y_test)} ejemplos")
print("\nExactitud en prueba por tamaño de entrenamiento:")
print(f"{'fracción':>9} | {'n_train':>7} | {'NB':>8} | {'RegLog':>8}")
for p in curva:
    print(f"{p['fraccion']:>8.0%} | {p['n_entrenamiento']:>7d} | "
          f"{p['exactitud_prueba_naive_bayes']:>8.4f} | "
          f"{p['exactitud_prueba_regresion_logistica']:>8.4f}")
print(f"\nFigura guardada en: {ARCHIVO_FIGURA}")
print(f"Resultados guardados en: {ARCHIVO_RESULTADOS}")
