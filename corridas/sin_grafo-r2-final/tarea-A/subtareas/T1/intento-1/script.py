# Subtarea T1: Carga del dataset Breast Cancer Wisconsin (Diagnostic),
# reporte de cifras básicas y creación de la única división 70/30
# estratificada (random_state=42) que se usará en toda la tarea.

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def main():
    # ------------------------------------------------------------------
    # 1) Carga del dataset
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = [str(f) for f in data.feature_names]
    target_names = [str(t) for t in data.target_names]

    n_ejemplos = int(X.shape[0])
    n_atributos = int(X.shape[1])

    # Conteo de casos por clase en el dataset completo
    valores, conteos = np.unique(y, return_counts=True)
    conteo_por_clase = {
        f"clase_{int(v)}_{target_names[int(v)]}": int(c)
        for v, c in zip(valores, conteos)
    }

    # ------------------------------------------------------------------
    # 2) División 70/30 estratificada por clase, random_state=42
    #    (única división de la tarea; el test no se usa para ajustar nada)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    def conteos(etiquetas):
        v, c = np.unique(etiquetas, return_counts=True)
        return {
            f"clase_{int(vv)}_{target_names[int(vv)]}": int(cc)
            for vv, cc in zip(v, c)
        }

    conteo_train = conteos(y_train)
    conteo_test = conteos(y_test)

    # ------------------------------------------------------------------
    # 3) Contrato de resultados: resultados.json
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T1",
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "n_ejemplos_totales": n_ejemplos,
        "n_atributos": n_atributos,
        "nombres_atributos": feature_names,
        "clases": {str(i): nombre for i, nombre in enumerate(target_names)},
        "conteo_por_clase": conteo_por_clase,
        "division": {
            "test_size": 0.30,
            "train_size": 0.70,
            "estratificado": True,
            "random_state": 42,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "conteo_train_por_clase": conteo_train,
            "conteo_test_por_clase": conteo_test,
        },
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 4) Guardar los conjuntos de entrenamiento y prueba para las
    #    subtareas siguientes (misma división en toda la tarea)
    # ------------------------------------------------------------------
    np.savez(
        "split_train_test.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=np.array(feature_names),
        target_names=np.array(target_names),
    )

    # ------------------------------------------------------------------
    # 5) Resumen breve
    # ------------------------------------------------------------------
    print("=== T1: Breast Cancer Wisconsin (Diagnostic) ===")
    print(f"Ejemplos totales: {n_ejemplos}")
    print(f"Atributos: {n_atributos}")
    print(f"Conteo por clase (total): {conteo_por_clase}")
    print(f"Train: {X_train.shape[0]} ejemplos | Test: {X_test.shape[0]} ejemplos")
    print(f"Conteo train: {conteo_train}")
    print(f"Conteo test:  {conteo_test}")
    print("Archivos escritos: resultados.json, split_train_test.npz")


if __name__ == "__main__":
    main()
