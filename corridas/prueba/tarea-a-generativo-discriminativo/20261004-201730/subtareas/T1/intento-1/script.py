"""Subtarea T1: carga de Breast Cancer Wisconsin (Diagnostic), cifras básicas y
creación de la única división train/test 70/30 estratificada (random_state=42).

La partición se guarda por índices (y también como arrays .npz) para que las
demás subtareas la reutilicen exactamente sin volver a dividir.
"""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# 1. Carga del conjunto de datos
# ------------------------------------------------------------------
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = [str(f) for f in data.feature_names]
target_names = [str(t) for t in data.target_names]

n_ejemplos = int(X.shape[0])
n_atributos = int(X.shape[1])

clases_unicas, conteos = np.unique(y, return_counts=True)
conteo_por_clase = {target_names[int(c)]: int(n) for c, n in zip(clases_unicas, conteos)}

# ------------------------------------------------------------------
# 2. División ÚNICA train/test 70/30, estratificada por clase,
#    random_state=42. Se divide sobre índices para poder reconstruir
#    la partición idéntica en las subtareas siguientes.
# ------------------------------------------------------------------
indices = np.arange(n_ejemplos)
train_idx, test_idx = train_test_split(
    indices, test_size=0.30, stratify=y, random_state=42
)
train_idx = np.sort(train_idx)
test_idx = np.sort(test_idx)

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

clases_tr, conteos_tr = np.unique(y_train, return_counts=True)
clases_te, conteos_te = np.unique(y_test, return_counts=True)
conteo_train = {target_names[int(c)]: int(n) for c, n in zip(clases_tr, conteos_tr)}
conteo_test = {target_names[int(c)]: int(n) for c, n in zip(clases_te, conteos_te)}

# ------------------------------------------------------------------
# 3. Guardar partición reutilizable (.npz) y cifras del contrato (json)
# ------------------------------------------------------------------
np.savez(
    "datos_division_T1.npz",
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    train_idx=train_idx, test_idx=test_idx,
)

resultados = {
    "subtarea": "T1",
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "n_ejemplos_total": n_ejemplos,
    "n_atributos": n_atributos,
    "n_clases": int(len(clases_unicas)),
    "nombres_clases": target_names,
    "nombres_atributos": feature_names,
    "conteo_por_clase": conteo_por_clase,
    "proporcion_por_clase": {k: float(v) / n_ejemplos for k, v in conteo_por_clase.items()},
    "division": {
        "test_size": 0.30,
        "estratificada": True,
        "random_state": 42,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "fraccion_train": float(len(train_idx) / n_ejemplos),
        "fraccion_test": float(len(test_idx) / n_ejemplos),
        "conteo_por_clase_train": conteo_train,
        "conteo_por_clase_test": conteo_test,
        "train_indices": [int(i) for i in train_idx],
        "test_indices": [int(i) for i in test_idx],
    },
    "archivo_particion": "datos_division_T1.npz",
}

with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

# ------------------------------------------------------------------
# 4. Figura: distribución de clases
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
nombres = list(conteo_por_clase.keys())
valores = list(conteo_por_clase.values())
ax.bar(nombres, valores, color=["#d95f02", "#1b9e77"])
for i, v in enumerate(valores):
    ax.text(i, v + 3, str(v), ha="center")
ax.set_ylabel("Número de ejemplos")
ax.set_title("Breast Cancer Wisconsin (Diagnostic): casos por clase")
fig.tight_layout()
fig.savefig("distribucion_clases_T1.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 5. Resumen
# ------------------------------------------------------------------
print("=== T1: Carga de datos y división única train/test ===")
print(f"Ejemplos totales : {n_ejemplos}")
print(f"Atributos        : {n_atributos}")
print(f"Clases           : {conteo_por_clase}")
print(f"Train            : {len(train_idx)} ejemplos ({len(train_idx)/n_ejemplos:.1%}) -> {conteo_train}")
print(f"Test             : {len(test_idx)} ejemplos ({len(test_idx)/n_ejemplos:.1%}) -> {conteo_test}")
print("Parámetros de la división: test_size=0.30, stratify=y, random_state=42")
print("Archivos generados: resultados.json, datos_division_T1.npz, distribucion_clases_T1.png")
