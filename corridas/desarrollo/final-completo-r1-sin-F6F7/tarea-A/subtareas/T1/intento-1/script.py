# T1: Cargar Breast Cancer Wisconsin (Diagnostic), reportar cifras básicas
# y crear la única división train/test 70/30 estratificada (random_state=42).

from pathlib import Path
import json

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.30


def main():
    # ------------------------------------------------------------------
    # 1) Carga del dataset
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=list(data.feature_names))
    y = pd.Series(data.target, name="target")
    target_names = list(data.target_names)  # ['malignant', 'benign']

    total_ejemplos = int(X.shape[0])
    num_atributos = int(X.shape[1])

    # Conteo de casos por clase en el dataset completo
    conteo_clases = {
        nombre: int((y == codigo).sum()) for codigo, nombre in enumerate(target_names)
    }
    proporcion_clases = {k: v / total_ejemplos for k, v in conteo_clases.items()}

    # ------------------------------------------------------------------
    # 2) Única división train/test 70/30, estratificada por clase
    #    (el conjunto de prueba queda intacto para toda la tarea)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # ------------------------------------------------------------------
    # 3) Guardar la división en archivos reutilizables por las demás subtareas
    # ------------------------------------------------------------------
    train_df = X_train.copy()
    train_df["target"] = y_train.values
    test_df = X_test.copy()
    test_df["target"] = y_test.values

    train_df.to_parquet("train.parquet", index=False)
    test_df.to_parquet("test.parquet", index=False)

    # ------------------------------------------------------------------
    # 4) Cifras de la división (verificación de estratificación)
    # ------------------------------------------------------------------
    conteo_train = {
        nombre: int((y_train == codigo).sum()) for codigo, nombre in enumerate(target_names)
    }
    conteo_test = {
        nombre: int((y_test == codigo).sum()) for codigo, nombre in enumerate(target_names)
    }

    # ------------------------------------------------------------------
    # 5) Contrato de resultados
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T1",
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "total_ejemplos": total_ejemplos,
        "num_atributos": num_atributos,
        "nombres_atributos": list(data.feature_names),
        "clases": target_names,
        "conteo_clases": conteo_clases,
        "proporcion_clases": proporcion_clases,
        "division": {
            "test_size": TEST_SIZE,
            "train_size": 1.0 - TEST_SIZE,
            "estratificada_por_clase": True,
            "random_state": RANDOM_STATE,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "conteo_clases_train": conteo_train,
            "conteo_clases_test": conteo_test,
        },
        "archivos_generados": ["train.parquet", "test.parquet"],
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 6) Resumen
    # ------------------------------------------------------------------
    print("=== T1 completada ===")
    print(f"Total de ejemplos: {total_ejemplos}")
    print(f"Número de atributos: {num_atributos}")
    print(f"Conteo por clase: {conteo_clases}")
    print(
        f"División 70/30 estratificada (random_state={RANDOM_STATE}): "
        f"train={len(X_train)} ejemplos, test={len(X_test)} ejemplos"
    )
    print(f"Conteo train: {conteo_train}")
    print(f"Conteo test:  {conteo_test}")
    print("División guardada en train.parquet y test.parquet; cifras en resultados.json")


if __name__ == "__main__":
    main()
