# T3 — Curva de aprendizaje: GNB y Regresión Logística sobre la división única de T1
# Entrena con submuestras estratificadas del 5/10/25/50/100 % del entrenamiento
# (random_state=42) y evalúa SIEMPRE sobre el mismo conjunto de prueba de T1.

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
FRACCIONES = [0.05, 0.10, 0.25, 0.50, 1.00]
ETIQUETAS = ["5 %", "10 %", "25 %", "50 %", "100 %"]

# ------------------------------------------------------------------
# 1) Cargar la división única de T1 (entrada/T1)
# ------------------------------------------------------------------
t1_dir = Path("entrada") / "T1"
X_train = np.load(t1_dir / "X_train.npy")
X_test = np.load(t1_dir / "X_test.npy")
y_train = np.load(t1_dir / "y_train.npy")
y_test = np.load(t1_dir / "y_test.npy")

n_train_total = int(X_train.shape[0])
n_test = int(X_test.shape[0])
n_atributos = int(X_train.shape[1])

MODELOS = ["naive_bayes_gaussiano", "regresion_logistica"]
curvas = {m: {"n": [], "acc": [], "f1": []} for m in MODELOS}
composicion = []
filas = []

# ------------------------------------------------------------------
# 2) Bucle sobre los cinco tamaños de entrenamiento
# ------------------------------------------------------------------
for frac, etiq in zip(FRACCIONES, ETIQUETAS):
    if frac >= 1.0:
        X_sub, y_sub = X_train, y_train  # 100 %: entrenamiento completo
    else:
        # Submuestra ESTRATIFICADA del entrenamiento, random_state=42
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train,
            train_size=frac,
            stratify=y_train,
            random_state=SEED,
        )
    n_sub = int(len(y_sub))

    composicion.append({
        "fraccion": frac,
        "etiqueta": etiq,
        "n_train": n_sub,
        "n_clase_0_malignant": int(np.sum(y_sub == 0)),
        "n_clase_1_benign": int(np.sum(y_sub == 1)),
    })

    # --- Modelo 1: Naive Bayes gaussiano (atributos originales, como en T2) ---
    gnb = GaussianNB()
    gnb.fit(X_sub, y_sub)
    pred_gnb = gnb.predict(X_test)
    acc_gnb = float(accuracy_score(y_test, pred_gnb))
    f1_gnb = float(f1_score(y_test, pred_gnb, average="macro"))

    # --- Modelo 2: Regresión logística con atributos estandarizados ---
    # El escalador se ajusta SOLO con la submuestra de entrenamiento de este
    # tamaño (nunca con el conjunto de prueba).
    scaler = StandardScaler().fit(X_sub)
    rl = LogisticRegression(max_iter=1000, random_state=SEED)
    rl.fit(scaler.transform(X_sub), y_sub)
    pred_rl = rl.predict(scaler.transform(X_test))
    acc_rl = float(accuracy_score(y_test, pred_rl))
    f1_rl = float(f1_score(y_test, pred_rl, average="macro"))

    curvas["naive_bayes_gaussiano"]["n"].append(n_sub)
    curvas["naive_bayes_gaussiano"]["acc"].append(acc_gnb)
    curvas["naive_bayes_gaussiano"]["f1"].append(f1_gnb)
    curvas["regresion_logistica"]["n"].append(n_sub)
    curvas["regresion_logistica"]["acc"].append(acc_rl)
    curvas["regresion_logistica"]["f1"].append(f1_rl)

    filas.append({
        "fraccion": etiq,
        "n_train": n_sub,
        "accuracy_naive_bayes_gaussiano": acc_gnb,
        "accuracy_regresion_logistica": acc_rl,
        "f1_macro_naive_bayes_gaussiano": f1_gnb,
        "f1_macro_regresion_logistica": f1_rl,
    })

# ------------------------------------------------------------------
# 3) Figura: exactitud en prueba vs. número de ejemplos de entrenamiento
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.5))

ax.plot(curvas["naive_bayes_gaussiano"]["n"],
        curvas["naive_bayes_gaussiano"]["acc"],
        marker="o", color="tab:blue", linewidth=2,
        label="Naive Bayes gaussiano")
ax.plot(curvas["regresion_logistica"]["n"],
        curvas["regresion_logistica"]["acc"],
        marker="s", color="tab:red", linewidth=2,
        label="Regresión logística (atributos estandarizados)")

for n, a in zip(curvas["naive_bayes_gaussiano"]["n"],
                curvas["naive_bayes_gaussiano"]["acc"]):
    ax.annotate(f"{a:.3f}", (n, a), textcoords="offset points",
                xytext=(0, -16), ha="center", fontsize=8, color="tab:blue")
for n, a in zip(curvas["regresion_logistica"]["n"],
                curvas["regresion_logistica"]["acc"]):
    ax.annotate(f"{a:.3f}", (n, a), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, color="tab:red")

ax.set_xticks(curvas["naive_bayes_gaussiano"]["n"])
y_min = min(min(curvas["naive_bayes_gaussiano"]["acc"]),
            min(curvas["regresion_logistica"]["acc"]))
ax.set_ylim(max(0.0, y_min - 0.06), 1.02)
ax.set_xlabel("Número de ejemplos de entrenamiento")
ax.set_ylabel("Exactitud en el conjunto de prueba")
ax.set_title("Curva de aprendizaje — Breast Cancer Wisconsin "
             "(división 70/30 estratificada de T1, prueba fija de "
             f"{n_test} ejemplos)")
ax.grid(True, alpha=0.4)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("curva_aprendizaje.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 4) Tabla CSV y resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
df = pd.DataFrame(filas)
df.to_csv("tabla_exactitudes_por_tamano.csv", index=False)

resultados = {
    "subtarea": "T3",
    "descripcion": ("Curva de aprendizaje: GNB y regresión logística (modelos de T2) "
                    "entrenados con submuestras estratificadas del 5/10/25/50/100 % "
                    "del entrenamiento de T1 (random_state=42) y evaluados siempre "
                    "sobre el mismo conjunto de prueba de T1."),
    "division": {
        "origen": "entrada/T1 (única división 70/30 estratificada, random_state=42)",
        "n_train_total": n_train_total,
        "n_test": n_test,
        "n_atributos": n_atributos,
    },
    "protocolo": {
        "fracciones_entrenamiento": FRACCIONES,
        "submuestreo": ("train_test_split estratificado con random_state=42; "
                        "para 100 % se usa el entrenamiento completo"),
        "conjunto_de_evaluacion": (f"mismo conjunto de prueba de T1 en todos los "
                                   f"tamaños ({n_test} ejemplos); nunca se usa "
                                   f"para ajustar nada"),
        "escalado_regresion_logistica": ("StandardScaler ajustado solo con la "
                                         "submuestra de entrenamiento de cada "
                                         "tamaño (nunca con el conjunto de prueba)"),
        "modelos": {
            "naive_bayes_gaussiano": "GaussianNB sobre atributos originales (como en T2)",
            "regresion_logistica": ("LogisticRegression(max_iter=1000, random_state=42) "
                                    "sobre atributos estandarizados (como en T2)"),
        },
    },
    "tamanos_entrenamiento": curvas["naive_bayes_gaussiano"]["n"],
    "composicion_submuestras": composicion,
    "exactitud_prueba_por_tamano": {
        "naive_bayes_gaussiano": curvas["naive_bayes_gaussiano"]["acc"],
        "regresion_logistica": curvas["regresion_logistica"]["acc"],
    },
    "f1_macro_prueba_por_tamano": {
        "naive_bayes_gaussiano": curvas["naive_bayes_gaussiano"]["f1"],
        "regresion_logistica": curvas["regresion_logistica"]["f1"],
    },
    "tabla_por_tamano": filas,
    "figura": "curva_aprendizaje.png",
    "archivos_generados": [
        "resultados.json",
        "curva_aprendizaje.png",
        "tabla_exactitudes_por_tamano.csv",
    ],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# 5) Resumen
# ------------------------------------------------------------------
print("T3 — Curva de aprendizaje (evaluación fija en el conjunto de prueba "
      f"de {n_test} ejemplos)")
print(f"{'fracción':>8} {'n_train':>8} {'acc GNB':>10} {'acc RL':>10}")
for fila in filas:
    print(f"{fila['fraccion']:>8} {fila['n_train']:>8} "
          f"{fila['accuracy_naive_bayes_gaussiano']:>10.4f} "
          f"{fila['accuracy_regresion_logistica']:>10.4f}")
print("Figura guardada: curva_aprendizaje.png")
print("Tablas guardadas: resultados.json, tabla_exactitudes_por_tamano.csv")
