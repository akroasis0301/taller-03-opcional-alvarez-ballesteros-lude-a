# T1: Carga de Breast Cancer Wisconsin (Diagnostic) y división 70/30 estratificada (random_state=42)
from pathlib import Path
import json
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
X = pd.DataFrame(data.data, columns=[str(f) for f in data.feature_names])
y = pd.Series(data.target.astype(int), name="target")

n_ejemplos = int(X.shape[0])
n_atributos = int(X.shape[1])
nombres_clases = [str(c) for c in data.target_names]  # ['malignant', 'benign']
n_clases = int(len(nombres_clases))

conteo_por_clase = {
    nombres_clases[i]: int((y == i).sum()) for i in range(n_clases)
}
proporcion_por_clase = {
    k: v / n_ejemplos for k, v in conteo_por_clase.items()
}

# ------------------------------------------------------------------
# 2) División única entrenamiento/prueba: 70/30, estratificada por clase,
#    random_state=42 (el conjunto de prueba no se usa para ajustar nada)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

n_train = int(X_train.shape[0])
n_test = int(X_test.shape[0])
n_train_por_clase = {
    nombres_clases[i]: int((y_train == i).sum()) for i in range(n_clases)
}
n_test_por_clase = {
    nombres_clases[i]: int((y_test == i).sum()) for i in range(n_clases)
}

# Verificación de estratificación (proporciones similares en train y test)
prop_train = {k: v / n_train for k, v in n_train_por_clase.items()}
prop_test = {k: v / n_test for k, v in n_test_por_clase.items()}

# ------------------------------------------------------------------
# 3) Guardar los conjuntos para las subtareas siguientes
# ------------------------------------------------------------------
train_df = X_train.copy()
train_df["target"] = y_train.values
test_df = X_test.copy()
test_df["target"] = y_test.values

train_df.to_parquet("train.parquet", index=False)
test_df.to_parquet("test.parquet", index=False)
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

# ------------------------------------------------------------------
# 4) resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T1",
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "n_ejemplos": n_ejemplos,
    "n_atributos": n_atributos,
    "n_clases": n_clases,
    "nombres_clases": nombres_clases,
    "mapeo_clases": {str(i): nombres_clases[i] for i in range(n_clases)},
    "conteo_por_clase": conteo_por_clase,
    "proporcion_por_clase": proporcion_por_clase,
    "division": {
        "test_size": 0.30,
        "estratificada": True,
        "random_state": 42,
        "n_train": n_train,
        "n_test": n_test,
        "n_train_por_clase": n_train_por_clase,
        "n_test_por_clase": n_test_por_clase,
        "proporcion_train_por_clase": prop_train,
        "proporcion_test_por_clase": prop_test,
    },
    "nombres_atributos": [str(f) for f in data.feature_names],
    "archivos_generados": ["train.parquet", "test.parquet", "train.csv", "test.csv"],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# 5) Figura: distribución de clases (referencia para el reporte)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
valores = [conteo_por_clase[c] for c in nombres_clases]
ax.bar(nombres_clases, valores, color=["#b34444", "#4a9a4a"])
for i, v in enumerate(valores):
    ax.text(i, v + 3, str(v), ha="center", fontsize=10)
ax.set_ylabel("Número de ejemplos")
ax.set_title("Breast Cancer Wisconsin (Diagnostic): casos por clase")
plt.tight_layout()
plt.savefig("distribucion_clases.png", dpi=120)
plt.close()

# ------------------------------------------------------------------
# 6) Resumen
# ------------------------------------------------------------------
print("=== T1: Carga y división de datos ===")
print(f"Ejemplos totales: {n_ejemplos}")
print(f"Atributos: {n_atributos}")
print(f"Clases: {conteo_por_clase}")
print(f"División 70/30 estratificada (random_state=42):")
print(f"  Entrenamiento: {n_train} ejemplos -> {n_train_por_clase}")
print(f"  Prueba:        {n_test} ejemplos -> {n_test_por_clase}")
print("Archivos escritos: resultados.json, train.parquet, test.parquet, "
      "train.csv, test.csv, distribucion_clases.png")
