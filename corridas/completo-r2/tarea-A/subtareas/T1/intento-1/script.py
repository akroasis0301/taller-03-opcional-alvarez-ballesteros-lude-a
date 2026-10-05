# -*- coding: utf-8 -*-
"""
Subtarea T1 — Tarea A: Generativo contra discriminativo con pocos datos
=======================================================================
Carga el conjunto Breast Cancer Wisconsin (Diagnostic) de scikit-learn,
reporta el número de ejemplos, el número de atributos y los casos por clase,
y crea la ÚNICA división train/test 70/30 estratificada por clase con
random_state=42 que se usará en toda la tarea.

Salidas (carpeta actual, rutas relativas):
  - resultados.json              -> cifras de la subtarea (contrato)
  - division_train_test.npz      -> X_train, X_test, y_train, y_test + índices (reutilizable)
  - division_train_test.json     -> índices y parámetros de la división (reutilizable)
  - t1_distribucion_clases.png   -> figura con la distribución de clases
"""

import json

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Parámetros fijados por el enunciado (no se modifican)
RANDOM_STATE = 42
TEST_SIZE = 0.3


def main():
    # ------------------------------------------------------------------
    # 1) Carga del dataset
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = [str(f) for f in data.feature_names]
    target_names = [str(t) for t in data.target_names]  # ['malignant', 'benign']

    n_ejemplos = int(X.shape[0])
    n_atributos = int(X.shape[1])

    # ------------------------------------------------------------------
    # 2) Distribución de clases en el conjunto completo
    # ------------------------------------------------------------------
    etiquetas, conteos = np.unique(y, return_counts=True)
    conteo_clases_total = {
        target_names[int(et)]: int(ct) for et, ct in zip(etiquetas, conteos)
    }
    conteo_clases_total_por_etiqueta = {
        int(et): int(ct) for et, ct in zip(etiquetas, conteos)
    }

    # ------------------------------------------------------------------
    # 3) División ÚNICA train/test 70/30, estratificada por clase, seed=42
    #    (el conjunto de prueba no se usa para ajustar nada en la tarea)
    # ------------------------------------------------------------------
    indices = np.arange(n_ejemplos)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    tam_train = int(X_train.shape[0])
    tam_test = int(X_test.shape[0])

    et_train, ct_train = np.unique(y_train, return_counts=True)
    et_test, ct_test = np.unique(y_test, return_counts=True)
    conteo_train = {target_names[int(e)]: int(c) for e, c in zip(et_train, ct_train)}
    conteo_test = {target_names[int(e)]: int(c) for e, c in zip(et_test, ct_test)}

    # ------------------------------------------------------------------
    # 4) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T1",
        "dataset": "breast_cancer_wisconsin_diagnostic (sklearn.datasets.load_breast_cancer)",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "estratificado_por_clase": True,
        # Cifras pedidas por la subtarea:
        "total_ejemplos": n_ejemplos,
        "numero_atributos": n_atributos,
        "nombres_clases": target_names,
        "conteo_clases_total": conteo_clases_total,
        "conteo_clases_total_por_etiqueta_numerica": conteo_clases_total_por_etiqueta,
        "tamano_entrenamiento": tam_train,
        "tamano_prueba": tam_test,
        "fraccion_entrenamiento": tam_train / n_ejemplos,
        "fraccion_prueba": tam_test / n_ejemplos,
        # Información complementaria de la división estratificada:
        "conteo_clases_entrenamiento": conteo_train,
        "conteo_clases_prueba": conteo_test,
        "nombres_atributos": feature_names,
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 5) División guardada y reutilizable para las demás subtareas
    # ------------------------------------------------------------------
    np.savez(
        "division_train_test.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        indices_train=idx_train,
        indices_test=idx_test,
    )
    with open("division_train_test.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "estratificado_por_clase": True,
                "indices_entrenamiento": idx_train.tolist(),
                "indices_prueba": idx_test.tolist(),
            },
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # 6) Figura: distribución de clases (total, entrenamiento, prueba)
    # ------------------------------------------------------------------
    fig, ejes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    paneles = [
        ("Total", conteo_clases_total),
        ("Entrenamiento (70%)", conteo_train),
        ("Prueba (30%)", conteo_test),
    ]
    for ax, (titulo, conteo) in zip(ejes, paneles):
        valores = [conteo[n] for n in target_names]
        barras = ax.bar(target_names, valores, color=["#c44e52", "#4c72b0"])
        ax.bar_label(barras)
        ax.set_title(titulo)
        ax.set_ylabel("Número de casos")
    fig.suptitle("Breast Cancer Wisconsin (Diagnostic) — distribución de clases")
    fig.tight_layout()
    plt.savefig("t1_distribucion_clases.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 7) Resumen breve por consola
    # ------------------------------------------------------------------
    print("=== Subtarea T1 — Resumen ===")
    print(f"Dataset: Breast Cancer Wisconsin (Diagnostic)")
    print(f"Total de ejemplos: {n_ejemplos}")
    print(f"Número de atributos: {n_atributos}")
    print(f"Conteo por clase (total): {conteo_clases_total}")
    print(f"División 70/30 estratificada, random_state={RANDOM_STATE}")
    print(f"  Entrenamiento: {tam_train} ejemplos ({tam_train / n_ejemplos:.4f}) -> {conteo_train}")
    print(f"  Prueba:        {tam_test} ejemplos ({tam_test / n_ejemplos:.4f}) -> {conteo_test}")
    print("Archivos generados: resultados.json, division_train_test.npz, "
          "division_train_test.json, t1_distribucion_clases.png")


if __name__ == "__main__":
    main()
