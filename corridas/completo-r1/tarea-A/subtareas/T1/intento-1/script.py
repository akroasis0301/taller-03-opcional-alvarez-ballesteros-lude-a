# T1 — Carga de datos, reporte descriptivo y división 70/30 estratificada
# Breast Cancer Wisconsin (Diagnostic) · random_state=42 · MMIA 6013 Taller 03 v2

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# 1) Carga del conjunto de datos
# ------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target
class_names = [str(c) for c in data.target_names]  # ['malignant', 'benign']

n_ejemplos, n_atributos = X.shape
clases_unicas, conteos = np.unique(y, return_counts=True)
casos_por_clase_total = {
    class_names[int(c)]: int(n) for c, n in zip(clases_unicas, conteos)
}

# ------------------------------------------------------------------
# 2) División entrenamiento/prueba 70/30, estratificada por clase,
#    random_state=42 (única división usada en toda la tarea)
# ------------------------------------------------------------------
indices = np.arange(n_ejemplos)
idx_train, idx_test, y_train, y_test = train_test_split(
    indices, y, test_size=0.30, stratify=y, random_state=42
)
X_train, X_test = X[idx_train], X[idx_test]

cl_tr, co_tr = np.unique(y_train, return_counts=True)
cl_te, co_te = np.unique(y_test, return_counts=True)
casos_por_clase_train = {class_names[int(c)]: int(n) for c, n in zip(cl_tr, co_tr)}
casos_por_clase_test = {class_names[int(c)]: int(n) for c, n in zip(cl_te, co_te)}

# ------------------------------------------------------------------
# 3) Guardar la división para reutilizarse en T2 y T3
# ------------------------------------------------------------------
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
np.save("idx_train.npy", idx_train)
np.save("idx_test.npy", idx_test)
np.save("X_full.npy", X)
np.save("y_full.npy", y)

# ------------------------------------------------------------------
# 4) resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T1",
    "dataset": "Breast Cancer Wisconsin (Diagnostic) — sklearn.datasets.load_breast_cancer",
    "n_ejemplos_totales": int(n_ejemplos),
    "n_atributos": int(n_atributos),
    "nombres_clases": class_names,
    "casos_por_clase_total": casos_por_clase_total,
    "n_train": int(len(idx_train)),
    "n_test": int(len(idx_test)),
    "proporcion_test": float(len(idx_test) / n_ejemplos),
    "casos_por_clase_train": casos_por_clase_train,
    "casos_por_clase_test": casos_por_clase_test,
    "parametros_split": {
        "test_size": 0.30,
        "estratificado_por_clase": True,
        "random_state": 42,
    },
    "indices_train": idx_train.tolist(),
    "indices_test": idx_test.tolist(),
    "archivos_guardados": [
        "X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy",
        "idx_train.npy", "idx_test.npy", "X_full.npy", "y_full.npy",
    ],
}

with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

# ------------------------------------------------------------------
# 5) Figura: distribución de clases (total / train / test)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
xpos = np.arange(len(class_names))
width = 0.25
tot = [casos_por_clase_total[c] for c in class_names]
tr = [casos_por_clase_train[c] for c in class_names]
te = [casos_por_clase_test[c] for c in class_names]
ax.bar(xpos - width, tot, width, label="Total")
ax.bar(xpos, tr, width, label="Entrenamiento (70%)")
ax.bar(xpos + width, te, width, label="Prueba (30%)")
ax.set_xticks(xpos)
ax.set_xticklabels(class_names)
ax.set_ylabel("Número de casos")
ax.set_title("Breast Cancer Wisconsin (Diagnostic): distribución de clases")
ax.legend()
plt.tight_layout()
plt.savefig("t1_distribucion_clases.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 6) Resumen
# ------------------------------------------------------------------
print("=== T1: Carga de datos y división 70/30 estratificada ===")
print(f"Ejemplos totales: {n_ejemplos}")
print(f"Atributos: {n_atributos}")
print(f"Clases: {class_names}")
print(f"Casos por clase (total): {casos_por_clase_total}")
print(f"Entrenamiento: {len(idx_train)} ejemplos -> {casos_por_clase_train}")
print(f"Prueba:        {len(idx_test)} ejemplos -> {casos_por_clase_test}")
print("División (random_state=42, estratificada) guardada en .npy y en resultados.json")
print("Archivos: resultados.json, t1_distribucion_clases.png, X_*/y_*/idx_*.npy")
