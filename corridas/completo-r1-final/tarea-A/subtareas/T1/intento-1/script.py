# T1: Carga de Breast Cancer Wisconsin (Diagnostic), reporte de cifras básicas
# y creación de la división única entrenamiento/prueba (70/30, estratificada, random_state=42).

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# 1) Carga del conjunto de datos
# ------------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=list(data.feature_names))
y = pd.Series(data.target.astype(int), name="target")

n_ejemplos = int(X.shape[0])
n_atributos = int(X.shape[1])
nombres_atributos = list(data.feature_names)
nombres_clases = list(data.target_names)          # ['malignant', 'benign']
codificacion = {nombres_clases[i]: int(i) for i in range(len(nombres_clases))}

conteo_total = y.value_counts().sort_index()
conteo_clases_total = {nombres_clases[int(c)]: int(conteo_total.loc[c]) for c in conteo_total.index}

# ------------------------------------------------------------------
# 2) División única 70/30 estratificada por clase, random_state=42
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, train_size=0.70, stratify=y, random_state=42
)

n_train = int(X_train.shape[0])
n_test = int(X_test.shape[0])

def conteos(y_serie):
    c = y_serie.value_counts().sort_index()
    return {nombres_clases[int(k)]: int(v) for k, v in c.items()}

conteo_clases_train = conteos(y_train)
conteo_clases_test = conteos(y_test)

# ------------------------------------------------------------------
# 3) Guardar los conjuntos para las subtareas siguientes (archivos CSV)
# ------------------------------------------------------------------
train_df = X_train.copy()
train_df["target"] = y_train.values
test_df = X_test.copy()
test_df["target"] = y_test.values

Path("t1_train.csv").parent.mkdir(parents=True, exist_ok=True)
train_df.to_csv("t1_train.csv", index=False)
test_df.to_csv("t1_test.csv", index=False)

# ------------------------------------------------------------------
# 4) Figura: distribución de clases en total, entrenamiento y prueba
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
grupos = ["Total", "Entrenamiento (70%)", "Prueba (30%)"]
valores_malignant = [
    conteo_clases_total["malignant"],
    conteo_clases_train["malignant"],
    conteo_clases_test["malignant"],
]
valores_benign = [
    conteo_clases_total["benign"],
    conteo_clases_train["benign"],
    conteo_clases_test["benign"],
]
xpos = np.arange(len(grupos))
ancho = 0.35
ax.bar(xpos - ancho / 2, valores_malignant, ancho, label="malignant (0)", color="#c44e52")
ax.bar(xpos + ancho / 2, valores_benign, ancho, label="benign (1)", color="#4c72b0")
for i, (m, b) in enumerate(zip(valores_malignant, valores_benign)):
    ax.text(i - ancho / 2, m + 3, str(m), ha="center", fontsize=9)
    ax.text(i + ancho / 2, b + 3, str(b), ha="center", fontsize=9)
ax.set_xticks(xpos)
ax.set_xticklabels(grupos)
ax.set_ylabel("Número de ejemplos")
ax.set_title("Breast Cancer Wisconsin (Diagnostic): conteo por clase\n"
             "División 70/30 estratificada, random_state=42")
ax.legend()
fig.tight_layout()
plt.savefig("t1_distribucion_clases.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 5) resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T1",
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "n_ejemplos_total": n_ejemplos,
    "n_atributos": n_atributos,
    "nombres_atributos": nombres_atributos,
    "nombres_clases": nombres_clases,
    "codificacion_clases": codificacion,
    "conteo_clases_total": conteo_clases_total,
    "division": {
        "test_size": 0.30,
        "train_size": 0.70,
        "estratificada": True,
        "random_state": 42,
    },
    "n_train": n_train,
    "n_test": n_test,
    "fraccion_train": n_train / n_ejemplos,
    "fraccion_test": n_test / n_ejemplos,
    "conteo_clases_train": conteo_clases_train,
    "conteo_clases_test": conteo_clases_test,
    "archivos_generados": {
        "train": "t1_train.csv",
        "test": "t1_test.csv",
        "figura": "t1_distribucion_clases.png",
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# 6) Resumen
# ------------------------------------------------------------------
print("=== T1: Carga y división de Breast Cancer Wisconsin (Diagnostic) ===")
print(f"Ejemplos totales : {n_ejemplos}")
print(f"Atributos        : {n_atributos}")
print(f"Clases           : {nombres_clases} (codificación {codificacion})")
print(f"Conteo total     : {conteo_clases_total}")
print(f"División         : 70/30 estratificada, random_state=42")
print(f"Entrenamiento    : {n_train} ejemplos -> {conteo_clases_train}")
print(f"Prueba           : {n_test} ejemplos -> {conteo_clases_test}")
print("Archivos escritos: t1_train.csv, t1_test.csv, t1_distribucion_clases.png, resultados.json")
