# -*- coding: utf-8 -*-
"""
T1: Cargar el dataset Breast Cancer Wisconsin (Diagnostic), reportar cifras
básicas (n ejemplos, n atributos, conteo por clase) y crear la división
entrenamiento/prueba 70/30 estratificada por clase con random_state=42.
La división se guarda de forma reutilizable para las demás subtareas.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# 1) Carga del dataset
# ----------------------------------------------------------------------
ds = load_breast_cancer()
X, y = ds.data, ds.target
feature_names = [str(f) for f in ds.feature_names]
target_names = [str(t) for t in ds.target_names]  # 0 = malignant, 1 = benign

n_ejemplos, n_atributos = X.shape

# Conteo de casos por clase en el dataset completo
clases_unicas, conteos = np.unique(y, return_counts=True)
conteo_por_clase = {target_names[int(c)]: int(n) for c, n in zip(clases_unicas, conteos)}
conteo_por_clase_codigo = {int(c): int(n) for c, n in zip(clases_unicas, conteos)}

# ----------------------------------------------------------------------
# 2) División 70/30 estratificada por clase, random_state=42
#    (única división usada en toda la tarea; el test no se usa para ajustar)
# ----------------------------------------------------------------------
indices = np.arange(n_ejemplos)
idx_train, idx_test = train_test_split(
    indices, test_size=0.30, stratify=y, random_state=42
)
X_train, X_test = X[idx_train], X[idx_test]
y_train, y_test = y[idx_train], y[idx_test]

n_train, n_test = int(len(idx_train)), int(len(idx_test))


def conteos_por_clase(y_vec):
    cls, cnt = np.unique(y_vec, return_counts=True)
    return {target_names[int(c)]: int(n) for c, n in zip(cls, cnt)}


conteo_train = conteos_por_clase(y_train)
conteo_test = conteos_por_clase(y_test)

# ----------------------------------------------------------------------
# 3) Contrato de salida: resultados.json con todas las cifras pedidas
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T1_carga_y_division",
    "dataset": "Breast Cancer Wisconsin (Diagnostic) - sklearn.datasets.load_breast_cancer",
    "n_ejemplos": int(n_ejemplos),
    "n_atributos": int(n_atributos),
    "nombres_atributos": feature_names,
    "clases": {str(int(c)): target_names[int(c)] for c in clases_unicas},
    "conteo_por_clase_total": conteo_por_clase,
    "conteo_por_clase_total_codigo": conteo_por_clase_codigo,
    "parametros_division": {
        "test_size": 0.30,
        "train_size": 0.70,
        "estratificado": True,
        "random_state": 42,
    },
    "n_train": n_train,
    "n_test": n_test,
    "proporcion_train": n_train / n_ejemplos,
    "proporcion_test": n_test / n_ejemplos,
    "conteo_por_clase_train": conteo_train,
    "conteo_por_clase_test": conteo_test,
    "indices_train": idx_train.tolist(),
    "indices_test": idx_test.tolist(),
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------------
# 4) Guardar la división de forma reutilizable para las demás subtareas
# ----------------------------------------------------------------------
np.savez(
    "division_train_test.npz",
    idx_train=idx_train,
    idx_test=idx_test,
    X=X,
    y=y,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

# ----------------------------------------------------------------------
# 5) Figura de verificación: distribución de clases total/train/test
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.2))
etiquetas = target_names  # ["malignant", "benign"]
totales = [conteo_por_clase.get(e, 0) for e in etiquetas]
tr = [conteo_train.get(e, 0) for e in etiquetas]
te = [conteo_test.get(e, 0) for e in etiquetas]
xpos = np.arange(len(etiquetas))
w = 0.25
b1 = ax.bar(xpos - w, totales, w, label=f"Total (n={n_ejemplos})")
b2 = ax.bar(xpos, tr, w, label=f"Train (n={n_train})")
b3 = ax.bar(xpos + w, te, w, label=f"Test (n={n_test})")
for bars in (b1, b2, b3):
    ax.bar_label(bars, padding=2)
ax.set_xticks(xpos)
ax.set_xticklabels(etiquetas)
ax.set_ylabel("Número de casos")
ax.set_title(
    "Breast Cancer Wisconsin: distribución de clases\n"
    "(división 70/30 estratificada, random_state=42)"
)
ax.legend()
fig.tight_layout()
plt.savefig("t1_distribucion_clases.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 6) Resumen breve
# ----------------------------------------------------------------------
print("=== T1: Carga y división del dataset Breast Cancer Wisconsin ===")
print(f"Ejemplos totales : {n_ejemplos}")
print(f"Atributos        : {n_atributos}")
print(f"Clases           : {target_names} (códigos 0 y 1)")
print(f"Conteo por clase : {conteo_por_clase}")
print("División 70/30 estratificada por clase, random_state=42")
print(f"  Train: {n_train} ejemplos ({n_train / n_ejemplos:.4f} del total) -> {conteo_train}")
print(f"  Test : {n_test} ejemplos ({n_test / n_ejemplos:.4f} del total) -> {conteo_test}")
print("Archivos generados: resultados.json, division_train_test.npz, t1_distribucion_clases.png")
