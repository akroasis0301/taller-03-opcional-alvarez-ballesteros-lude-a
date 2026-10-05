# -*- coding: utf-8 -*-
"""
T3 — Curva de aprendizaje (Tarea A, Parte 3).

Sobre la misma división 70/30 estratificada de T1, entrena los dos modelos de
T2 (Naive Bayes gaussiano y regresión logística con estandarización ajustada
solo con datos de entrenamiento) con submuestras estratificadas del 5 %, 10 %,
25 %, 50 % y 100 % del conjunto de entrenamiento (random_state=42), evalúa cada
tamaño siempre sobre el mismo conjunto de prueba, y grafica exactitud contra
número de ejemplos de entrenamiento con una curva por modelo.

Salidas:
  - curva_aprendizaje.png : figura con las dos curvas (una por modelo).
  - resultados.json       : exactitudes (y F1 macro) de cada modelo en cada tamaño.
"""

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEMILLA = 42
PORCENTAJES = [0.05, 0.10, 0.25, 0.50, 1.00]
ETIQUETAS_PCT = [5, 10, 25, 50, 100]

# ------------------------------------------------------------------
# 1. Recuperar la división única de T1 (el test no se usa para ajustar nada)
# ------------------------------------------------------------------
with open("entrada/T1/resultados.json", "r", encoding="utf-8") as f:
    t1 = json.load(f)

div = t1.get("division", t1)
train_idx = np.asarray(div["train_indices"], dtype=int)
test_idx = np.asarray(div["test_indices"], dtype=int)

data = load_breast_cancer()
X, y = data.data, data.target

X_train_full, y_train_full = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
n_train_full = int(X_train_full.shape[0])
n_test = int(X_test.shape[0])

# ------------------------------------------------------------------
# 2. Los mismos dos modelos de T2 (instancias nuevas para cada tamaño)
# ------------------------------------------------------------------
def construir_modelos():
    return {
        "Naive Bayes gaussiano": GaussianNB(),
        "Regresión logística": Pipeline(
            [
                ("escalador", StandardScaler()),
                ("modelo", LogisticRegression(max_iter=1000)),
            ]
        ),
    }

# ------------------------------------------------------------------
# 3. Entrenar con cada submuestra estratificada y evaluar en el test fijo
# ------------------------------------------------------------------
filas = []
tamanos = []
acc_nb, acc_lr = [], []
f1_nb, f1_lr = [], []

for pct, etiqueta in zip(PORCENTAJES, ETIQUETAS_PCT):
    if pct >= 1.0:
        idx_sub = np.arange(n_train_full)  # 100 %: entrenamiento completo
    else:
        idx_sub, _ = train_test_split(
            np.arange(n_train_full),
            train_size=pct,
            stratify=y_train_full,
            random_state=SEMILLA,
        )

    X_sub, y_sub = X_train_full[idx_sub], y_train_full[idx_sub]
    n_sub = int(X_sub.shape[0])
    tamanos.append(n_sub)

    accs, f1s = {}, {}
    for nombre, modelo in construir_modelos().items():
        modelo.fit(X_sub, y_sub)            # el escalador (pipeline) se ajusta solo con la submuestra
        y_pred = modelo.predict(X_test)     # evaluación siempre sobre el mismo conjunto de prueba
        accs[nombre] = float(accuracy_score(y_test, y_pred))
        f1s[nombre] = float(f1_score(y_test, y_pred, average="macro"))

    acc_nb.append(accs["Naive Bayes gaussiano"])
    acc_lr.append(accs["Regresión logística"])
    f1_nb.append(f1s["Naive Bayes gaussiano"])
    f1_lr.append(f1s["Regresión logística"])

    n_mal = int(np.sum(y_sub == 0))  # 0 = malignant
    n_ben = int(np.sum(y_sub == 1))  # 1 = benign

    filas.append(
        {
            "porcentaje": etiqueta,
            "n_ejemplos_entrenamiento": n_sub,
            "n_malignant_en_submuestra": n_mal,
            "n_benign_en_submuestra": n_ben,
            "accuracy_naive_bayes_gaussiano": accs["Naive Bayes gaussiano"],
            "accuracy_regresion_logistica": accs["Regresión logística"],
            "f1_macro_naive_bayes_gaussiano": f1s["Naive Bayes gaussiano"],
            "f1_macro_regresion_logistica": f1s["Regresión logística"],
        }
    )
    print(
        f"{etiqueta:>3} %  n_train={n_sub:>3}  "
        f"NB acc={accs['Naive Bayes gaussiano']:.4f}  "
        f"LR acc={accs['Regresión logística']:.4f}"
    )

# ------------------------------------------------------------------
# 4. Figura: exactitud contra número de ejemplos de entrenamiento
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(
    tamanos, acc_nb, marker="o", linewidth=2,
    label="Naive Bayes gaussiano (generativo)", color="#1f77b4",
)
ax.plot(
    tamanos, acc_lr, marker="s", linewidth=2,
    label="Regresión logística (discriminativo)", color="#d62728",
)

for desplazamiento, valores, color in [(-14, acc_nb, "#1f77b4"), (8, acc_lr, "#d62728")]:
    for n, a in zip(tamanos, valores):
        ax.annotate(
            f"{a:.3f}", (n, a), textcoords="offset points",
            xytext=(0, desplazamiento), ha="center", fontsize=8, color=color,
        )

ax.set_xticks(tamanos)
ax.set_xticklabels([f"{n}\n({p} %)" for n, p in zip(tamanos, ETIQUETAS_PCT)])
y_min = min(min(acc_nb), min(acc_lr))
ax.set_ylim(max(0.0, y_min - 0.06), 1.01)
ax.set_xlabel("Número de ejemplos de entrenamiento (submuestra estratificada)")
ax.set_ylabel("Exactitud (accuracy) en el conjunto de prueba")
ax.set_title(
    "Curva de aprendizaje — Breast Cancer (división 70/30 de T1, "
    f"test fijo de {n_test} ejemplos)"
)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("curva_aprendizaje.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 5. resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T3",
    "descripcion": (
        "Curva de aprendizaje: exactitud en el conjunto de prueba (fijo, de la división de T1) "
        "frente al número de ejemplos de entrenamiento, para Naive Bayes gaussiano y regresión "
        "logística entrenados con submuestras estratificadas del 5 %, 10 %, 25 %, 50 % y 100 % "
        "del entrenamiento (random_state=42)."
    ),
    "division_usada": {
        "fuente": "entrada/T1/resultados.json (train_indices / test_indices)",
        "n_train_completo": n_train_full,
        "n_test": n_test,
        "random_state_division": 42,
    },
    "porcentajes": ETIQUETAS_PCT,
    "tamanos_entrenamiento": tamanos,
    "curva_aprendizaje": filas,
    "accuracy_naive_bayes_gaussiano_por_tamano": acc_nb,
    "accuracy_regresion_logistica_por_tamano": acc_lr,
    "f1_macro_naive_bayes_gaussiano_por_tamano": f1_nb,
    "f1_macro_regresion_logistica_por_tamano": f1_lr,
    "figura": "curva_aprendizaje.png",
    "config_modelos": {
        "Naive Bayes gaussiano": "GaussianNB() (valores por defecto)",
        "Regresión logística": (
            "Pipeline(StandardScaler(), LogisticRegression(max_iter=1000)); "
            "el escalador se ajusta únicamente con la submuestra de entrenamiento de cada tamaño"
        ),
    },
    "protocolo_submuestreo": (
        "train_test_split(train_size=fraccion, stratify=y_train, random_state=42) por cada "
        "porcentaje; el 100 % usa el entrenamiento completo. El conjunto de prueba es siempre "
        "el mismo y no se usa para ajustar nada (ni escalador ni hiperparámetros)."
    ),
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# 6. Resumen
# ------------------------------------------------------------------
print("\nResumen T3 — exactitud en el conjunto de prueba (mismo test en todos los tamaños)")
print(f"{'%':>5} {'n_train':>8} {'NB':>8} {'LR':>8}")
for fila in filas:
    print(
        f"{fila['porcentaje']:>4}% {fila['n_ejemplos_entrenamiento']:>8} "
        f"{fila['accuracy_naive_bayes_gaussiano']:>8.4f} "
        f"{fila['accuracy_regresion_logistica']:>8.4f}"
    )
print("\nFigura guardada: curva_aprendizaje.png")
print("Resultados guardados: resultados.json")
