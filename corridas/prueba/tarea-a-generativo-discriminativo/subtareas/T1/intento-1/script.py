"""T1: Carga del dataset Breast Cancer Wisconsin y creación de la división única 70/30
estratificada por clase con random_state=42. Persiste cifras en resultados.json y la
división en t1_split.npz para las subtareas siguientes."""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# 1) Carga del dataset
# ------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target
target_names = [str(t) for t in data.target_names]

n_ejemplos = int(X.shape[0])
n_atributos = int(X.shape[1])

clases, conteos = np.unique(y, return_counts=True)
conteo_por_clase = {target_names[int(c)]: int(n) for c, n in zip(clases, conteos)}

# ------------------------------------------------------------------
# 2) División única 70/30 estratificada por clase, random_state=42
#    (el conjunto de prueba no se usa para ajustar nada en la tarea)
# ------------------------------------------------------------------
indices = np.arange(n_ejemplos)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices,
    test_size=0.30,
    train_size=0.70,
    stratify=y,
    random_state=42,
)

cl_tr, co_tr = np.unique(y_train, return_counts=True)
cl_te, co_te = np.unique(y_test, return_counts=True)
conteo_train = {target_names[int(c)]: int(n) for c, n in zip(cl_tr, co_tr)}
conteo_test = {target_names[int(c)]: int(n) for c, n in zip(cl_te, co_te)}

# ------------------------------------------------------------------
# 3) Persistencia de la división (contrato para subtareas siguientes)
# ------------------------------------------------------------------
np.savez(
    "t1_split.npz",
    X_train=X_train, X_test=X_test,
    y_train=y_train, y_test=y_test,
    idx_train=idx_train, idx_test=idx_test,
)

resultados = {
    "subtarea": "T1_carga_y_split",
    "dataset": "breast_cancer (sklearn.datasets.load_breast_cancer)",
    "n_ejemplos_total": n_ejemplos,
    "n_atributos": n_atributos,
    "nombres_atributos": [str(f) for f in data.feature_names],
    "clases": {str(int(c)): target_names[int(c)] for c in clases},
    "conteo_por_clase_total": conteo_por_clase,
    "split": {
        "test_size": 0.30,
        "train_size": 0.70,
        "estratificado_por_clase": True,
        "random_state": 42,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "proporcion_test": float(X_test.shape[0]) / n_ejemplos,
        "conteo_por_clase_train": conteo_train,
        "conteo_por_clase_test": conteo_test,
    },
    "indices_train": idx_train.tolist(),
    "indices_test": idx_test.tolist(),
    "archivos_persistidos": ["t1_split.npz"],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ------------------------------------------------------------------
# 4) Figura resumen (PNG)
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
nombres = [target_names[int(c)] for c in clases]

axes[0].bar(nombres, [conteo_por_clase[n] for n in nombres],
            color=["#c44e52", "#4c72b0"])
axes[0].set_title(f"Casos por clase (total = {n_ejemplos})")
axes[0].set_ylabel("Número de ejemplos")
for i, n in enumerate(nombres):
    axes[0].text(i, conteo_por_clase[n], str(conteo_por_clase[n]),
                 ha="center", va="bottom")

etiquetas = ["Entrenamiento", "Prueba"]
tamanos = [int(X_train.shape[0]), int(X_test.shape[0])]
axes[1].bar(etiquetas, tamanos, color=["#55a868", "#8172b2"])
axes[1].set_title("División 70/30 estratificada (random_state=42)")
axes[1].set_ylabel("Número de ejemplos")
for i, v in enumerate(tamanos):
    axes[1].text(i, v, str(v), ha="center", va="bottom")

fig.tight_layout()
fig.savefig("t1_resumen_datos.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 5) Resumen impreso
# ------------------------------------------------------------------
print("=" * 62)
print("T1 — Breast Cancer Wisconsin: carga y división 70/30")
print("=" * 62)
print(f"Ejemplos totales : {n_ejemplos}")
print(f"Atributos        : {n_atributos}")
print("Casos por clase  :")
for n in nombres:
    print(f"  - {n:<10}: {conteo_por_clase[n]}")
print(f"Entrenamiento    : {int(X_train.shape[0])} ejemplos {conteo_train}")
print(f"Prueba           : {int(X_test.shape[0])} ejemplos {conteo_test}")
print("División persistida en 't1_split.npz'; cifras en 'resultados.json';")
print("figura en 't1_resumen_datos.png'.")
