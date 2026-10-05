# -*- coding: utf-8 -*-
"""
T3 — Parte 3: Curva de aprendizaje (generativo vs discriminativo).

Sobre la única división 70/30 estratificada (random_state=42) de la Parte 1:
  * Para cada porcentaje p en {5, 10, 25, 50, 100} % se toma una submuestra
    ESTRATIFICADA del conjunto de entrenamiento con
    train_test_split(train_size=p, stratify=y_train, random_state=42).
  * Se entrenan los dos modelos de la Parte 2: Naive Bayes gaussiano
    (generativo, sin escalado) y regresión logística (discriminativo, con
    StandardScaler ajustado SOLO con la submuestra de entrenamiento).
  * Cada modelo se evalúa SIEMPRE sobre el mismo conjunto de prueba.
Se guarda la tabla de accuracy en prueba (5 tamaños x 2 modelos = 10 cifras)
en resultados.json y la figura con las dos curvas de aprendizaje como
curvas_aprendizaje.png (dpi=120).
"""

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

SEMILLA = 42
FIGURA_PNG = "curvas_aprendizaje.png"
PORCENTAJES = [0.05, 0.10, 0.25, 0.50, 1.00]  # fracciones del entrenamiento
PCT_ETIQUETAS = [5, 10, 25, 50, 100]          # etiquetas en %

# ----------------------------------------------------------------------
# 1) Datos y división única de la Parte 1 (70/30, estratificada, seed 42)
# ----------------------------------------------------------------------
datos = load_breast_cancer()
X, y = datos.data, datos.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEMILLA
)

n_train_total = int(X_train.shape[0])
n_test = int(X_test.shape[0])
conteo_train = {c: int((y_train == c).sum()) for c in (0, 1)}
conteo_test = {c: int((y_test == c).sum()) for c in (0, 1)}

# Verificación: debe ser EXACTAMENTE la misma división que produjo T1
ruta_t1 = Path("entrada/T1/resultados.json")
misma_division_t1 = None
if ruta_t1.exists():
    t1 = json.loads(ruta_t1.read_text(encoding="utf-8"))
    misma_division_t1 = bool(
        t1.get("n_train") == n_train_total
        and t1.get("n_test") == n_test
        and int(t1.get("conteo_train_por_clase", {}).get("malignant", -1)) == conteo_train[0]
        and int(t1.get("conteo_train_por_clase", {}).get("benign", -1)) == conteo_train[1]
        and int(t1.get("conteo_test_por_clase", {}).get("malignant", -1)) == conteo_test[0]
        and int(t1.get("conteo_test_por_clase", {}).get("benign", -1)) == conteo_test[1]
    )
    if not misma_division_t1:
        raise RuntimeError(
            "La división reproducida no coincide con la de la Parte 1 "
            "(entrada/T1/resultados.json)."
        )

# ----------------------------------------------------------------------
# 2) Curva de aprendizaje: submuestras estratificadas + los dos modelos
# ----------------------------------------------------------------------
filas = []          # tabla: una fila por tamaño de entrenamiento
composicion = []    # composición por clase de cada submuestra (verificación)
acc_nb, acc_lr = [], []
f1_nb, f1_lr = [], []
tamanos = []

for frac, pct in zip(PORCENTAJES, PCT_ETIQUETAS):
    if frac >= 1.0:
        X_sub, y_sub = X_train, y_train
    else:
        X_sub, _, y_sub, _ = train_test_split(
            X_train,
            y_train,
            train_size=frac,
            stratify=y_train,
            random_state=SEMILLA,
        )

    n_sub = int(X_sub.shape[0])
    tamanos.append(n_sub)
    composicion.append(
        {
            "porcentaje_pct": pct,
            "n_ejemplos": n_sub,
            "malignant": int((y_sub == 0).sum()),
            "benign": int((y_sub == 1).sum()),
        }
    )

    # --- Modelo generativo: Naive Bayes gaussiano (no requiere escalado) ---
    nb = GaussianNB()
    nb.fit(X_sub, y_sub)
    pred_nb = nb.predict(X_test)

    # --- Modelo discriminativo: regresión logística ---
    # El escalador se ajusta SOLO con la submuestra de entrenamiento;
    # el conjunto de prueba nunca se usa para ajustar nada.
    escalador = StandardScaler().fit(X_sub)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(escalador.transform(X_sub), y_sub)
    pred_lr = lr.predict(escalador.transform(X_test))

    a_nb = float(accuracy_score(y_test, pred_nb))
    a_lr = float(accuracy_score(y_test, pred_lr))
    f_nb = float(f1_score(y_test, pred_nb, average="macro"))
    f_lr = float(f1_score(y_test, pred_lr, average="macro"))

    acc_nb.append(a_nb)
    acc_lr.append(a_lr)
    f1_nb.append(f_nb)
    f1_lr.append(f_lr)

    filas.append(
        {
            "porcentaje_entrenamiento_pct": pct,
            "n_ejemplos_entrenamiento": n_sub,
            "accuracy_prueba_naive_bayes_gaussiano": a_nb,
            "accuracy_prueba_regresion_logistica": a_lr,
            "f1_macro_prueba_naive_bayes_gaussiano": f_nb,
            "f1_macro_prueba_regresion_logistica": f_lr,
        }
    )

# ----------------------------------------------------------------------
# 3) Verificación suave: el punto 100 % debe reproducir la Parte 2 (T2)
# ----------------------------------------------------------------------
coincide_t2 = None
ruta_t2 = Path("entrada/T2/resultados.json")
if ruta_t2.exists():
    t2 = json.loads(ruta_t2.read_text(encoding="utf-8"))
    m = t2.get("accuracy_f1_macro_prueba", {})

    def _acc(d, clave):
        v = d.get(clave, {}).get("accuracy")
        return None if v is None else float(v)

    a_nb_t2 = _acc(m, "naive_bayes_gaussiano")
    a_lr_t2 = _acc(m, "regresion_logistica")
    coincide_t2 = {
        "naive_bayes_gaussiano": bool(a_nb_t2 is not None and abs(a_nb_t2 - acc_nb[-1]) < 1e-12),
        "regresion_logistica": bool(a_lr_t2 is not None and abs(a_lr_t2 - acc_lr[-1]) < 1e-12),
    }

# ----------------------------------------------------------------------
# 4) Figura: exactitud vs número de ejemplos de entrenamiento (2 curvas)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.0, 5.5))
ax.plot(
    tamanos, acc_nb, marker="o", color="#1f77b4", linewidth=2,
    label="Naive Bayes gaussiano (generativo)",
)
ax.plot(
    tamanos, acc_lr, marker="s", color="#d62728", linewidth=2,
    label="Regresión logística (discriminativo)",
)

for n, a in zip(tamanos, acc_nb):
    ax.annotate(f"{a:.3f}", (n, a), textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=8, color="#1f77b4")
for n, a in zip(tamanos, acc_lr):
    ax.annotate(f"{a:.3f}", (n, a), textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=8, color="#d62728")

ax.set_xlabel("Número de ejemplos de entrenamiento")
ax.set_ylabel("Exactitud (accuracy) en el conjunto de prueba")
ax.set_title(
    "Curva de aprendizaje: generativo (NB gaussiano) vs discriminativo (regresión logística)\n"
    "Breast Cancer Wisconsin — división 70/30 estratificada, prueba fija (random_state=42)"
)
ax.set_xticks(tamanos)
lim_inf = max(0.0, min(min(acc_nb), min(acc_lr)) - 0.05)
ax.set_ylim(lim_inf, 1.02)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIGURA_PNG, dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 5) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T3",
    "descripcion": (
        "Parte 3: curva de aprendizaje. Naive Bayes gaussiano y regresión logística "
        "entrenados con el 5, 10, 25, 50 y 100 % del entrenamiento de la división 70/30 "
        "estratificada de la Parte 1 (submuestras estratificadas, random_state=42); "
        "evaluación de accuracy siempre sobre el mismo conjunto de prueba."
    ),
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "division_parte1": {
        "train_size": 0.7,
        "test_size": 0.3,
        "estratificada_por_clase": True,
        "random_state": 42,
        "n_train": n_train_total,
        "n_test": n_test,
    },
    "submuestreo_estratificado": {
        "metodo": "train_test_split(X_train, y_train, train_size=pct, stratify=y_train, random_state=42)",
        "random_state": 42,
        "escalador": "StandardScaler reajustado solo con cada submuestra de entrenamiento (nunca con la prueba)",
    },
    "porcentajes_entrenamiento_pct": PCT_ETIQUETAS,
    "n_ejemplos_entrenamiento": tamanos,
    "composicion_submuestras": composicion,
    "tabla_accuracy_prueba": filas,
    "accuracy_prueba_por_tamano": {
        "naive_bayes_gaussiano": acc_nb,
        "regresion_logistica": acc_lr,
    },
    "f1_macro_prueba_por_tamano": {
        "naive_bayes_gaussiano": f1_nb,
        "regresion_logistica": f1_lr,
    },
    "n_valores_accuracy_tabla": 10,
    "figura_png": FIGURA_PNG,
    "verificaciones": {
        "misma_division_que_T1": misma_division_t1,
        "accuracy_100pct_coincide_con_T2": coincide_t2,
    },
}

Path("resultados.json").write_text(
    json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
)

# ----------------------------------------------------------------------
# 6) Resumen
# ----------------------------------------------------------------------
print("=" * 74)
print(f"T3 — Curva de aprendizaje (conjunto de prueba fijo: n_test = {n_test})")
print("=" * 74)
encabezado = f"{'% ent.':>7} {'n_train':>8} {'Acc NB (gen.)':>15} {'Acc RL (disc.)':>16}"
print(encabezado)
print("-" * len(encabezado))
for f in filas:
    print(
        f"{f['porcentaje_entrenamiento_pct']:>6}% {f['n_ejemplos_entrenamiento']:>8} "
        f"{f['accuracy_prueba_naive_bayes_gaussiano']:>15.4f} "
        f"{f['accuracy_prueba_regresion_logistica']:>16.4f}"
    )
print("-" * len(encabezado))
print(f"Figura guardada en : {FIGURA_PNG}")
print("Tabla de accuracy (10 cifras) escrita en resultados.json")
if coincide_t2 is not None:
    print(f"Verificación 100 % vs T2: {coincide_t2}")
