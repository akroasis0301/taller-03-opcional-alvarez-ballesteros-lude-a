# T1: Carga del dataset Breast Cancer Wisconsin (Diagnostic), reporte de cifras
# y creación de la división entrenamiento/prueba 70/30 estratificada (random_state=42).

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def main():
    # ------------------------------------------------------------------
    # 1) Carga del conjunto de datos
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=list(data.feature_names))
    y = pd.Series(data.target, name="target")

    n_ejemplos = int(X.shape[0])
    n_atributos = int(X.shape[1])
    nombres_clases = [str(c) for c in data.target_names]  # ['malignant', 'benign']

    # Conteo de casos por clase (por nombre y por código numérico)
    conteo_por_clase = {
        nombres_clases[i]: int((y == i).sum()) for i in range(len(nombres_clases))
    }
    conteo_por_codigo = {
        int(code): int(cnt) for code, cnt in y.value_counts().sort_index().items()
    }

    # ------------------------------------------------------------------
    # 2) División 70/30 estratificada por clase, random_state=42
    #    (única división usada en toda la tarea; la prueba no se usa para ajustar)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, train_size=0.70, stratify=y, random_state=42
    )

    conteo_train = {
        nombres_clases[i]: int((y_train == i).sum()) for i in range(len(nombres_clases))
    }
    conteo_test = {
        nombres_clases[i]: int((y_test == i).sum()) for i in range(len(nombres_clases))
    }

    # ------------------------------------------------------------------
    # 3) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T1",
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "n_ejemplos_totales": n_ejemplos,
        "n_atributos": n_atributos,
        "nombres_atributos": list(data.feature_names),
        "nombres_clases": nombres_clases,
        "mapa_nombre_a_codigo": {nombres_clases[i]: i for i in range(len(nombres_clases))},
        "conteo_por_clase": conteo_por_clase,
        "conteo_por_clase_codigo": conteo_por_codigo,
        "division": {
            "train_size": 0.70,
            "test_size": 0.30,
            "estratificada_por_clase": True,
            "random_state": 42,
        },
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "conteo_train_por_clase": conteo_train,
        "conteo_test_por_clase": conteo_test,
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 4) Guardar los conjuntos de entrenamiento y prueba para subtareas siguientes
    #    (parquet, con el índice original preservado para trazabilidad)
    # ------------------------------------------------------------------
    X_train.to_parquet("X_entrenamiento.parquet", index=True)
    X_test.to_parquet("X_prueba.parquet", index=True)
    y_train.to_frame("target").to_parquet("y_entrenamiento.parquet", index=True)
    y_test.to_frame("target").to_parquet("y_prueba.parquet", index=True)

    # ------------------------------------------------------------------
    # 5) Figura: distribución de clases
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    valores = [conteo_por_clase[n] for n in nombres_clases]
    ax.bar(nombres_clases, valores, color=["#4c72b0", "#dd8452"])
    for i, v in enumerate(valores):
        ax.text(i, v + 3, str(v), ha="center", fontsize=11)
    ax.set_ylabel("Número de casos")
    ax.set_title("Breast Cancer Wisconsin (Diagnostic): casos por clase")
    plt.tight_layout()
    plt.savefig("distribucion_clases.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6) Resumen
    # ------------------------------------------------------------------
    print("=== T1 completada ===")
    print(f"Ejemplos totales: {n_ejemplos}")
    print(f"Atributos: {n_atributos}")
    print(f"Clases: {conteo_por_clase}")
    print(
        f"División 70/30 estratificada (random_state=42): "
        f"train={len(X_train)} {conteo_train} | test={len(X_test)} {conteo_test}"
    )
    print("Archivos escritos: resultados.json, X_entrenamiento.parquet, "
          "X_prueba.parquet, y_entrenamiento.parquet, y_prueba.parquet, "
          "distribucion_clases.png")


if __name__ == "__main__":
    main()
