# -*- coding: utf-8 -*-
"""
T3 — Curva de aprendizaje: generativo vs. discriminativo con pocos datos.

Sobre la MISMA división 70/30 estratificada de T1 (random_state=42), entrena:
  - Naive Bayes gaussiano (atributos originales, como en T2)
  - Regresión logística (atributos estandarizados; el escalador se ajusta SOLO
    con la submuestra de entrenamiento de cada corrida, como en T2)
con submuestras estratificadas del 5 %, 10 %, 25 %, 50 % y 100 % del
entrenamiento (random_state=42) y evalúa cada modelo SIEMPRE sobre el mismo
conjunto de prueba de T1 (171 ejemplos). El conjunto de prueba no se usa para
ajustar nada (ni modelos ni escalador).

Salidas: resultados.json, t3_curva_aprendizaje.png, t3_tabla_curva_aprendizaje.csv
"""

import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
PORCENTAJES = [0.05, 0.10, 0.25, 0.50, 1.00]
ETIQUETAS_PCT = [5, 10, 25, 50, 100]

NOMBRES = {
    "gaussian_nb": "Naive Bayes gaussiano (generativo)",
    "logistic_regression": "Regresión logística (discriminativo)",
}

# ------------------------------------------------------------------ #
# 1) Cargar la división de T1 (la única división de toda la tarea)   #
# ------------------------------------------------------------------ #
with open("entrada/T1/resultados.json", "r", encoding="utf-8") as f:
    t1 = json.load(f)

X_train = np.load("entrada/T1/X_train.npy")
X_test = np.load("entrada/T1/X_test.npy")
y_train = np.load("entrada/T1/y_train.npy")
y_test = np.load("entrada/T1/y_test.npy")
idx_train_global = np.load("entrada/T1/idx_train.npy")

assert X_train.shape[0] == t1["n_train"] == 398, "X_train no coincide con T1"
assert X_test.shape[0] == t1["n_test"] == 171, "X_test no coincide con T1"

# ------------------------------------------------------------------ #
# 2) Entrenar con submuestras estratificadas y evaluar en prueba     #
# ------------------------------------------------------------------ #
curvas = {m: {"n_ejemplos": [], "accuracy": [], "f1_macro": []} for m in NOMBRES}
filas_tabla = []
detalle_submuestras = []
acc_por_pct = {}

for pct, frac in zip(ETIQUETAS_PCT, PORCENTAJES):
    if frac >= 1.0:
        # 100 %: entrenamiento completo (idéntico a la corrida de T2)
        X_sub, y_sub = X_train, y_train
        idx_sub_pos = np.arange(len(y_train))
    else:
        idx_pos = np.arange(len(y_train))
        X_sub, _, y_sub, _, idx_sub_pos, _ = train_test_split(
            X_train, y_train, idx_pos,
            train_size=frac,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )
    n_sub = int(len(y_sub))
    assert len(np.unique(y_sub)) == 2, "La submuestra debe tener las dos clases"
    idx_originales = sorted(int(i) for i in idx_train_global[idx_sub_pos])

    detalle_submuestras.append(
        {
            "porcentaje": pct,
            "n_ejemplos": n_sub,
            "malignant": int(np.sum(y_sub == 0)),
            "benign": int(np.sum(y_sub == 1)),
            "indices_en_dataset_original": idx_originales,
        }
    )

    # --- Generativo: GaussianNB sobre atributos originales (como en T2) ---
    gnb = GaussianNB()
    gnb.fit(X_sub, y_sub)
    pred_gnb = gnb.predict(X_test)
    acc_gnb = float(accuracy_score(y_test, pred_gnb))
    f1_gnb = float(f1_score(y_test, pred_gnb, average="macro"))

    # --- Discriminativo: LR con escalador ajustado SOLO con la submuestra ---
    scaler = StandardScaler().fit(X_sub)
    lr = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    lr.fit(scaler.transform(X_sub), y_sub)
    pred_lr = lr.predict(scaler.transform(X_test))
    acc_lr = float(accuracy_score(y_test, pred_lr))
    f1_lr = float(f1_score(y_test, pred_lr, average="macro"))

    acc_por_pct[f"{pct}pct"] = {
        "gaussian_nb": acc_gnb,
        "logistic_regression": acc_lr,
    }
    for modelo, acc, f1 in (
        ("gaussian_nb", acc_gnb, f1_gnb),
        ("logistic_regression", acc_lr, f1_lr),
    ):
        curvas[modelo]["n_ejemplos"].append(n_sub)
        curvas[modelo]["accuracy"].append(acc)
        curvas[modelo]["f1_macro"].append(f1)
        filas_tabla.append(
            {
                "porcentaje": pct,
                "n_ejemplos_entrenamiento": n_sub,
                "modelo": NOMBRES[modelo],
                "accuracy_prueba": acc,
                "f1_macro_prueba": f1,
            }
        )

# ------------------------------------------------------------------ #
# 3) Verificación: el punto 100 % debe reproducir los resultados T2  #
# ------------------------------------------------------------------ #
verificacion = {"comparado_con": "entrada/T2/resultados.json"}
try:
    with open("entrada/T2/resultados.json", "r", encoding="utf-8") as f:
        t2 = json.load(f)
    t2_gnb = t2["metricas_prueba"]["gaussian_nb"]["accuracy"]
    t2_lr = t2["metricas_prueba"]["logistic_regression"]["accuracy"]
    verificacion["accuracy_T2_gaussian_nb"] = t2_gnb
    verificacion["accuracy_T2_logistic_regression"] = t2_lr
    verificacion["accuracy_T3_100pct_gaussian_nb"] = curvas["gaussian_nb"]["accuracy"][-1]
    verificacion["accuracy_T3_100pct_logistic_regression"] = curvas["logistic_regression"]["accuracy"][-1]
    verificacion["coincide_con_T2"] = bool(
        np.isclose(t2_gnb, curvas["gaussian_nb"]["accuracy"][-1])
        and np.isclose(t2_lr, curvas["logistic_regression"]["accuracy"][-1])
    )
except (OSError, KeyError):
    verificacion["coincide_con_T2"] = None

# ------------------------------------------------------------------ #
# 4) Figura: exactitud vs. número de ejemplos (una curva por modelo) #
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(8.5, 5.5))
estilos = {
    "gaussian_nb": dict(marker="o", color="tab:blue", linestyle="-"),
    "logistic_regression": dict(marker="s", color="tab:orange", linestyle="-"),
}
offsets = {"gaussian_nb": (0, -16), "logistic_regression": (0, 8)}

for modelo in NOMBRES:
    n = curvas[modelo]["n_ejemplos"]
    a = curvas[modelo]["accuracy"]
    ax.plot(n, a, label=NOMBRES[modelo], **estilos[modelo])
    for x, v in zip(n, a):
        ax.annotate(
            f"{v:.3f}", (x, v), textcoords="offset points",
            xytext=offsets[modelo], ha="center", fontsize=8,
        )

ax.set_xlabel("Número de ejemplos de entrenamiento")
ax.set_ylabel("Exactitud (accuracy) en el conjunto de prueba")
ax.set_title(
    "Curva de aprendizaje: generativo vs. discriminativo\n"
    "Breast Cancer — división 70/30 de T1, prueba fija (171 ejemplos)"
)
ax.set_xticks(curvas["gaussian_nb"]["n_ejemplos"])
todos = [v for m in NOMBRES for v in curvas[m]["accuracy"]]
ax.set_ylim(min(todos) - 0.06, 1.01)
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("t3_curva_aprendizaje.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------ #
# 5) Tabla CSV y resultados.json (contrato de la subtarea)           #
# ------------------------------------------------------------------ #
tabla_df = pd.DataFrame(filas_tabla)
tabla_df.to_csv("t3_tabla_curva_aprendizaje.csv", index=False, encoding="utf-8")

resultados = {
    "subtarea": "T3",
    "descripcion": (
        "Curva de aprendizaje: GaussianNB y regresión logística entrenados con "
        "submuestras estratificadas del 5/10/25/50/100 % del entrenamiento de T1 "
        "(random_state=42) y evaluados SIEMPRE sobre el mismo conjunto de prueba "
        "de T1 (171 ejemplos)."
    ),
    "division_usada": "T1: train_test_split(test_size=0.3, stratify=y, random_state=42)",
    "submuestreo": (
        "train_test_split(train_size=frac, stratify=y_train, random_state=42) "
        "sobre el entrenamiento de T1; 100 % usa el entrenamiento completo (398)"
    ),
    "porcentajes": ETIQUETAS_PCT,
    "n_ejemplos_entrenamiento": curvas["gaussian_nb"]["n_ejemplos"],
    "detalle_submuestras": detalle_submuestras,
    "accuracy_por_porcentaje": acc_por_pct,
    "curva_aprendizaje": {
        "gaussian_nb": curvas["gaussian_nb"],
        "logistic_regression": curvas["logistic_regression"],
    },
    "tabla": filas_tabla,
    "figura": "t3_curva_aprendizaje.png",
    "archivos_generados": [
        "resultados.json",
        "t3_curva_aprendizaje.png",
        "t3_tabla_curva_aprendizaje.csv",
    ],
    "notas": (
        "El escalador de la regresión logística se ajusta únicamente con la "
        "submuestra de entrenamiento de cada corrida (protocolo de curva de "
        "aprendizaje: cada modelo solo ve sus n ejemplos); el conjunto de "
        "prueba no se usa para ajustar nada (ni modelos ni escalador)."
    ),
    "verificacion_con_T2": verificacion,
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------ #
# 6) Resumen breve                                                   #
# ------------------------------------------------------------------ #
print("T3 — Curva de aprendizaje (evaluación fija sobre el conjunto de prueba de T1)")
print(tabla_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
if verificacion.get("coincide_con_T2") is not None:
    print(
        "\nVerificación del punto 100 % contra T2:",
        "coincide" if verificacion["coincide_con_T2"] else "NO coincide",
    )
print("\nFigura guardada: t3_curva_aprendizaje.png")
print("Tabla guardada:  t3_tabla_curva_aprendizaje.csv")
print("Resultados guardados: resultados.json")
