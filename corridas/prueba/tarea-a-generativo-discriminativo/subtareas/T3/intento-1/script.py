#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 — Curva de aprendizaje: generativo (Naive Bayes gaussiano) vs discriminativo
(regresión logística) sobre la MISMA división 70/30 de T1 (Breast Cancer Wisconsin).

- Submuestras estratificadas del 5 %, 10 %, 25 %, 50 % y 100 % del entrenamiento
  (random_state=42).
- Cada punto se evalúa siempre sobre el mismo conjunto de prueba de T1.
- Figura: curva_aprendizaje.png | Resultados: resultados.json
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
FRACCIONES = [0.05, 0.10, 0.25, 0.50, 1.00]
NOMBRE_FIGURA = "curva_aprendizaje.png"
NOMBRE_RESULTADOS = "resultados.json"

# ------------------------------------------------------------------ datos
ds = load_breast_cancer()
X, y = ds.data, ds.target
n_total, n_atributos = X.shape

# ------------------------------------------ división fija de T1 (70/30)
def cargar_split():
    """Recupera la división 70/30 estratificada (random_state=42) fijada en T1."""
    # 1) archivo persistido por T1
    npz = Path("entrada") / "T1" / "t1_split.npz"
    if npz.exists():
        with np.load(npz) as z:
            files = set(z.files)
            if {"X_train", "X_test", "y_train", "y_test"} <= files:
                return (z["X_train"], z["X_test"], z["y_train"], z["y_test"],
                        "t1_split.npz (arreglos de T1)")
            for k_tr, k_te in [("indices_train", "indices_test"),
                               ("idx_train", "idx_test"),
                               ("train_indices", "test_indices")]:
                if {k_tr, k_te} <= files:
                    it = np.asarray(z[k_tr], dtype=int)
                    ie = np.asarray(z[k_te], dtype=int)
                    return (X[it], X[ie], y[it], y[ie],
                            f"t1_split.npz ({k_tr})")
    # 2) índices guardados en resultados.json de T1
    rj = Path("entrada") / "T1" / "resultados.json"
    if rj.exists():
        t1 = json.loads(rj.read_text(encoding="utf-8"))
        if "indices_train" in t1 and "indices_test" in t1:
            it = np.asarray(t1["indices_train"], dtype=int)
            ie = np.asarray(t1["indices_test"], dtype=int)
            return (X[it], X[ie], y[it], y[ie],
                    "índices de entrada/T1/resultados.json")
    # 3) recreación determinista con los mismos parámetros de T1
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    return X_tr, X_te, y_tr, y_te, "recreada (train_test_split 0.3, random_state=42, estratificada)"

X_train, X_test, y_train, y_test, origen_split = cargar_split()
n_train, n_test = int(len(y_train)), int(len(y_test))

# ------------------------------------------------------------------ modelos
def nuevos_modelos():
    """Instancias frescas por punto de la curva."""
    return {
        "naive_bayes_gaussiano": GaussianNB(),                       # generativo
        "regresion_logistica": LogisticRegression(max_iter=10000,    # discriminativo
                                                  random_state=RANDOM_STATE),
    }

# --------------------------------------------------- curva de aprendizaje
filas = []
curvas = {nombre: {"n_train": [], "exactitud_test": []}
          for nombre in nuevos_modelos()}

for frac in FRACCIONES:
    if frac >= 1.0:
        Xs, ys = X_train, y_train          # 100 %: entrenamiento completo
    else:
        Xs, _, ys, _ = train_test_split(   # submuestra estratificada
            X_train, y_train,
            train_size=frac, stratify=y_train, random_state=RANDOM_STATE)

    n_sub = int(len(ys))
    clases_sub = {str(int(c)): int(k)
                  for c, k in zip(*np.unique(ys, return_counts=True))}
    fila = {"fraccion": frac,
            "n_ejemplos_entrenamiento": n_sub,
            "clases_en_submuestra": clases_sub}

    for nombre, modelo in nuevos_modelos().items():
        modelo.fit(Xs, ys)                              # solo datos de entrenamiento
        acc = float(modelo.score(X_test, y_test))       # misma prueba fija siempre
        fila[f"exactitud_test_{nombre}"] = acc
        curvas[nombre]["n_train"].append(n_sub)
        curvas[nombre]["exactitud_test"].append(acc)
    filas.append(fila)

# ---------------------------------------------------------- resultados.json
resultados = {
    "subtarea": "T3_curva_aprendizaje",
    "descripcion": ("Exactitud en prueba contra tamaño de la submuestra estratificada "
                    "del entrenamiento (5 %, 10 %, 25 %, 50 %, 100 %); generativo "
                    "(GaussianNB) contra discriminativo (LogisticRegression); "
                    "evaluación siempre sobre el mismo conjunto de prueba de T1."),
    "dataset": "breast_cancer (sklearn.datasets.load_breast_cancer)",
    "n_ejemplos_total": int(n_total),
    "n_atributos": int(n_atributos),
    "split_t1": {
        "origen": origen_split,
        "test_size": 0.3,
        "estratificado": True,
        "random_state": RANDOM_STATE,
        "n_train": n_train,
        "n_test": n_test,
    },
    "modelos": {
        "generativo": "GaussianNB (Naive Bayes gaussiano)",
        "discriminativo": "LogisticRegression(max_iter=10000, random_state=42)",
    },
    "submuestreo": {
        "estratificado": True,
        "random_state": RANDOM_STATE,
        "fracciones": FRACCIONES,
    },
    "curva_aprendizaje": {
        nombre: {"n_ejemplos_entrenamiento": c["n_train"],
                 "exactitud_test": c["exactitud_test"]}
        for nombre, c in curvas.items()
    },
    "tabla_por_tamano": filas,
    "figura_png": NOMBRE_FIGURA,
}
with open(NOMBRE_RESULTADOS, "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------ figura
fig, ax = plt.subplots(figsize=(8.5, 5.5))
estilos = {
    "naive_bayes_gaussiano": ("o-", "Naive Bayes gaussiano (generativo)", 8),
    "regresion_logistica": ("s-", "Regresión logística (discriminativo)", -14),
}
for nombre, c in curvas.items():
    estilo, etiqueta, despl = estilos[nombre]
    ax.plot(c["n_train"], c["exactitud_test"], estilo, label=etiqueta)
    for n_v, a in zip(c["n_train"], c["exactitud_test"]):
        ax.annotate(f"{a:.3f}", (n_v, a), textcoords="offset points",
                    xytext=(0, despl), ha="center", fontsize=8)

n_vals = curvas["naive_bayes_gaussiano"]["n_train"]
ax.set_xticks(n_vals)
ax.set_xticklabels([f"{n}\n({int(f * 100)} %)" for n, f in zip(n_vals, FRACCIONES)])
ax.set_xlabel("Número de ejemplos de entrenamiento (submuestra estratificada)")
ax.set_ylabel("Exactitud en el conjunto de prueba")
ax.set_title("Curva de aprendizaje: generativo vs. discriminativo\n"
             f"Breast Cancer Wisconsin — prueba fija de {n_test} ejemplos")
todos = [a for c in curvas.values() for a in c["exactitud_test"]]
ax.set_ylim(min(todos) - 0.06, 1.015)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(NOMBRE_FIGURA, dpi=120)
plt.close(fig)

# ------------------------------------------------------------------ resumen
print("T3 — Curva de aprendizaje (generativo vs discriminativo)")
print(f"División T1 ({origen_split}): {n_train} entrenamiento / {n_test} prueba")
print(f"{'fracción':>9} {'n_train':>8} {'exact. NB':>10} {'exact. RL':>10}")
for fila in filas:
    print(f"{fila['fraccion'] * 100:>8.0f}% {fila['n_ejemplos_entrenamiento']:>8} "
          f"{fila['exactitud_test_naive_bayes_gaussiano']:>10.4f} "
          f"{fila['exactitud_test_regresion_logistica']:>10.4f}")
print(f"Figura guardada: {NOMBRE_FIGURA} | Resultados: {NOMBRE_RESULTADOS}")
