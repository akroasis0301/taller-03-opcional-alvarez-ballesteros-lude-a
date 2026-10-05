# -*- coding: utf-8 -*-
"""
T3 — Curva de aprendizaje: generativo (Naive Bayes gaussiano) vs. discriminativo
(regresión logística) con pocos datos (Ng & Jordan, 2002).

Usa la ÚNICA división 70/30 estratificada (random_state=42) creada en T1.
Reentrena los dos modelos de T2 con submuestras ESTRATIFICADAS del 5 %, 10 %,
25 %, 50 % y 100 % del entrenamiento (random_state=42) y evalúa cada tamaño
SIEMPRE sobre el mismo conjunto de prueba fijo de T1 (nunca se usa para ajustar
nada: ni modelos ni escaladores).

Salidas: t3_curva_aprendizaje.png, t3_tabla_curva_aprendizaje.csv, resultados.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
FRACCIONES = [0.05, 0.10, 0.25, 0.50, 1.00]
NB = "Naive Bayes gaussiano (GaussianNB)"
LR = "Regresión logística"
NOMBRES_MODELOS = [NB, LR]

# ----------------------------------------------------------------------
# 1) División fija de la Parte 1 (T1). El test queda reservado y fijo.
# ----------------------------------------------------------------------
dir_t1 = Path("entrada") / "T1"
rutas_t1 = {k: dir_t1 / f"t1_{k}.npy" for k in ("X_train", "X_test", "y_train", "y_test")}

if all(r.exists() for r in rutas_t1.values()):
    X_train = np.load(rutas_t1["X_train"])
    X_test = np.load(rutas_t1["X_test"])
    y_train = np.load(rutas_t1["y_train"])
    y_test = np.load(rutas_t1["y_test"])
    origen_split = "archivos .npy de T1 (entrada/T1/)"
else:
    # Respaldo determinista: exactamente la misma división que T1
    datos = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        datos.data, datos.target,
        test_size=0.30, stratify=datos.target, random_state=RANDOM_STATE,
    )
    origen_split = ("recreada de forma determinista: load_breast_cancer + "
                    "train_test_split(test_size=0.3, stratify, random_state=42)")

n_train_total = int(X_train.shape[0])
n_test = int(X_test.shape[0])

# ----------------------------------------------------------------------
# 2) Entrenar con cada submuestra estratificada y evaluar en el test fijo
# ----------------------------------------------------------------------
def entrenar_y_evaluar(X_sub, y_sub):
    """Pipeline de T2 (StandardScaler + modelos) ajustado SOLO con la submuestra
    de entrenamiento de la corrida; el conjunto de prueba solo se transforma y
    se predice (jamás se usa para ajustar escalador ni modelos)."""
    escalador = StandardScaler().fit(X_sub)      # ajustado solo con entrenamiento
    X_sub_z = escalador.transform(X_sub)
    X_test_z = escalador.transform(X_test)       # el test solo se transforma

    fabricas = {NB: GaussianNB, LR: lambda: LogisticRegression(max_iter=1000)}
    metricas = {}
    for nombre, fabrica in fabricas.items():
        modelo = fabrica()
        modelo.fit(X_sub_z, y_sub)
        pred = modelo.predict(X_test_z)
        metricas[nombre] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1_macro": float(f1_score(y_test, pred, average="macro")),
        }
    return metricas


corridas = []
for frac in FRACCIONES:
    if frac >= 1.0:
        X_sub, y_sub = X_train, y_train          # 100 %: entrenamiento completo
    else:
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train,
            train_size=frac, stratify=y_train, random_state=RANDOM_STATE,
        )
    n_sub = int(X_sub.shape[0])
    conteos = np.bincount(np.asarray(y_sub).astype(int), minlength=2)
    metricas = entrenar_y_evaluar(X_sub, y_sub)
    corridas.append({
        "fraccion": float(frac),
        "n_train": n_sub,
        "conteos_clase_submuestra": {"0 (malignant)": int(conteos[0]),
                                     "1 (benign)": int(conteos[1])},
        "metricas": metricas,
    })

# ----------------------------------------------------------------------
# 3) Curvas y tabla
# ----------------------------------------------------------------------
curvas = {
    m: {
        "fracciones": [c["fraccion"] for c in corridas],
        "n_train": [c["n_train"] for c in corridas],
        "accuracy": [c["metricas"][m]["accuracy"] for c in corridas],
        "f1_macro": [c["metricas"][m]["f1_macro"] for c in corridas],
    }
    for m in NOMBRES_MODELOS
}

tabla = []
for c in corridas:
    for m in NOMBRES_MODELOS:
        tabla.append({
            "modelo": m,
            "fraccion_train": c["fraccion"],
            "porcentaje_train": f"{c['fraccion']:.0%}",
            "n_ejemplos_train": c["n_train"],
            "accuracy_test": c["metricas"][m]["accuracy"],
            "f1_macro_test": c["metricas"][m]["f1_macro"],
        })
tabla_df = pd.DataFrame(tabla)
tabla_df.to_csv("t3_tabla_curva_aprendizaje.csv", index=False)

accuracy_por_tamano = {
    f"{c['fraccion']:.0%}": {
        "n_train": c["n_train"],
        NB: c["metricas"][NB]["accuracy"],
        LR: c["metricas"][LR]["accuracy"],
    }
    for c in corridas
}
f1_por_tamano = {
    f"{c['fraccion']:.0%}": {
        "n_train": c["n_train"],
        NB: c["metricas"][NB]["f1_macro"],
        LR: c["metricas"][LR]["f1_macro"],
    }
    for c in corridas
}

# ----------------------------------------------------------------------
# 4) Figura: exactitud vs. número de ejemplos de entrenamiento (2 curvas)
# ----------------------------------------------------------------------
n_vals = [c["n_train"] for c in corridas]
colores = {NB: "#1f77b4", LR: "#d62728"}
marcadores = {NB: "o", LR: "s"}
etiquetas_leyenda = {NB: "Naive Bayes gaussiano (generativo)",
                     LR: "Regresión logística (discriminativo)"}

fig, ax = plt.subplots(figsize=(9.0, 5.8))
for m in NOMBRES_MODELOS:
    ax.plot(n_vals, curvas[m]["accuracy"], marker=marcadores[m],
            color=colores[m], linewidth=2, markersize=6,
            label=etiquetas_leyenda[m])

# Anotar valores, desplazados al lado opuesto de la otra curva para no encimar
for i, m in enumerate(NOMBRES_MODELOS):
    otras = curvas[NOMBRES_MODELOS[1 - i]]["accuracy"]
    for n, a, o in zip(n_vals, curvas[m]["accuracy"], otras):
        dy = 9 if a >= o else -16
        ax.annotate(f"{a:.3f}", (n, a), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=8, color=colores[m])

ax.set_xticks(n_vals)
ax.set_xticklabels([f"{n}\n({c['fraccion']:.0%})" for n, c in zip(n_vals, corridas)])
todas_acc = [a for m in NOMBRES_MODELOS for a in curvas[m]["accuracy"]]
ax.set_ylim(min(0.5, min(todas_acc) - 0.08), 1.04)
ax.set_xlabel("Número de ejemplos de entrenamiento (submuestra estratificada del train de T1)")
ax.set_ylabel("Exactitud (accuracy) sobre el conjunto de prueba fijo")
ax.set_title("Curva de aprendizaje — Breast Cancer Wisconsin (Diagnostic)\n"
             f"División 70/30 de T1 (random_state=42); test fijo de {n_test} ejemplos")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("t3_curva_aprendizaje.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 5) Verificación: con el 100 % se debe reproducir el resultado de T2
# ----------------------------------------------------------------------
t2_accuracy_test = {NB: 0.935672514619883, LR: 0.9883040935672515}
verificacion_t2 = {
    m: {
        "accuracy_100pct_T3": curvas[m]["accuracy"][-1],
        "accuracy_T2": t2_accuracy_test[m],
        "diferencia_abs": abs(curvas[m]["accuracy"][-1] - t2_accuracy_test[m]),
    }
    for m in NOMBRES_MODELOS
}

# ----------------------------------------------------------------------
# 6) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T3",
    "descripcion": ("Curva de aprendizaje: los dos modelos de T2 (Naive Bayes gaussiano y "
                    "regresión logística con atributos estandarizados) reentrenados con "
                    "submuestras estratificadas del 5 %, 10 %, 25 %, 50 % y 100 % del "
                    "entrenamiento de T1 (random_state=42) y evaluados siempre sobre el "
                    "mismo conjunto de prueba fijo de T1."),
    "division_usada": {
        "origen": origen_split,
        "split": "70/30 estratificada por clase, random_state=42 (creada en T1)",
        "tam_train": n_train_total,
        "tam_test": n_test,
    },
    "protocolo": {
        "fracciones_train": [c["fraccion"] for c in corridas],
        "submuestreo": ("train_test_split(train_size=fraccion, stratify=y_train, "
                        "random_state=42) sobre el entrenamiento de T1; el 100 % usa el "
                        "entrenamiento completo"),
        "escalador": ("StandardScaler ajustado únicamente con la submuestra de "
                      "entrenamiento de cada corrida (nunca con el conjunto de prueba); "
                      "con el 100 % coincide con el escalador de T2"),
        "evaluacion": "accuracy (y F1 macro) sobre el test fijo de T1 en todos los tamaños",
        "modelos": {NB: "GaussianNB()", LR: "LogisticRegression(max_iter=1000)"},
        "random_state": RANDOM_STATE,
    },
    "tamanos_train": [c["n_train"] for c in corridas],
    "curva_aprendizaje": curvas,
    "accuracy_por_tamano": accuracy_por_tamano,
    "f1_macro_por_tamano": f1_por_tamano,
    "tabla": tabla,
    "detalle_submuestras": [
        {"fraccion": c["fraccion"], "n_train": c["n_train"],
         "conteos_clase_submuestra": c["conteos_clase_submuestra"]}
        for c in corridas
    ],
    "verificacion_contra_T2_100pct": verificacion_t2,
    "figura": "t3_curva_aprendizaje.png",
    "archivos_guardados": ["t3_curva_aprendizaje.png",
                           "t3_tabla_curva_aprendizaje.csv",
                           "resultados.json"],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 7) Resumen
# ----------------------------------------------------------------------
print("=" * 78)
print(f"T3 — Curva de aprendizaje (test fijo de T1: {n_test} ejemplos)")
print("-" * 78)
print(tabla_df.to_string(index=False))
print("-" * 78)
for m in NOMBRES_MODELOS:
    v = verificacion_t2[m]
    print(f"100 % vs T2 | {m}: T3={v['accuracy_100pct_T3']:.6f} "
          f"T2={v['accuracy_T2']:.6f} (|diff|={v['diferencia_abs']:.2e})")
print("-" * 78)
print("Figura guardada: t3_curva_aprendizaje.png")
print("Tabla guardada:  t3_tabla_curva_aprendizaje.csv")
print("Resultados guardados: resultados.json")
