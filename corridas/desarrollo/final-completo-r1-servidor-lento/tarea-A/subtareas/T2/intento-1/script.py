# ============================================================================
# T2 — Parte 2: Dos clasificadores (Naive Bayes gaussiano vs Regresión
# logística) sobre la división 70/30 estratificada de la Parte 1.
# Métricas en el conjunto de prueba: accuracy y F1 macro, en una tabla.
# ============================================================================
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42  # única semilla fijada en la Parte 1


def main():
    # ------------------------------------------------------------------
    # 1) Datos y división de la Parte 1 (70/30 estratificada por clase,
    #    random_state=42) — la misma división única de toda la tarea.
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    # Verificación defensiva contra los resultados de T1 (si existen)
    t1_file = Path("entrada/T1/resultados.json")
    if t1_file.exists():
        t1 = json.loads(t1_file.read_text(encoding="utf-8"))
        assert int(t1["n_train"]) == X_train.shape[0], "n_train no coincide con T1"
        assert int(t1["n_test"]) == X_test.shape[0], "n_test no coincide con T1"

    # ------------------------------------------------------------------
    # 2) Escalador ajustado SOLO con el conjunto de entrenamiento
    #    (el conjunto de prueba no se usa para ajustar nada).
    # ------------------------------------------------------------------
    scaler = StandardScaler().fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 3) Modelos
    #    - Naive Bayes gaussiano: atributos originales (la estandarización
    #      es una transformación afín por atributo y no cambia sus
    #      predicciones; se entrena sin ella por simplicidad).
    #    - Regresión logística: atributos estandarizados.
    # ------------------------------------------------------------------
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    lr = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    lr.fit(X_train_std, y_train)

    # ------------------------------------------------------------------
    # 4) Evaluación sobre el conjunto de prueba: accuracy y F1 macro
    # ------------------------------------------------------------------
    filas = []
    for nombre, modelo, Xte in [
        ("Naive Bayes gaussiano", nb, X_test),
        ("Regresión logística", lr, X_test_std),
    ]:
        y_pred = modelo.predict(Xte)
        filas.append(
            {
                "modelo": nombre,
                "accuracy_prueba": float(accuracy_score(y_test, y_pred)),
                "f1_macro_prueba": float(f1_score(y_test, y_pred, average="macro")),
            }
        )

    tabla = pd.DataFrame(filas)

    # ------------------------------------------------------------------
    # 5) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T2",
        "descripcion": (
            "Parte 2: Naive Bayes gaussiano y regresión logística (atributos "
            "estandarizados, escalador ajustado solo con el entrenamiento) "
            "entrenados sobre la división 70/30 estratificada de la Parte 1; "
            "accuracy y F1 macro medidos en el conjunto de prueba."
        ),
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "division": {
            "train_size": 0.7,
            "test_size": 0.3,
            "estratificada_por_clase": True,
            "random_state": RANDOM_STATE,
        },
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "escalador": "StandardScaler ajustado únicamente con X_train",
        "tabla_metricas_prueba": filas,
        "accuracy_f1_macro_prueba": {
            "naive_bayes_gaussiano": {
                "accuracy": filas[0]["accuracy_prueba"],
                "f1_macro": filas[0]["f1_macro_prueba"],
            },
            "regresion_logistica": {
                "accuracy": filas[1]["accuracy_prueba"],
                "f1_macro": filas[1]["f1_macro_prueba"],
            },
        },
    }
    Path("resultados.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # 6) Figura: tabla de métricas en PNG
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.6, 2.7))
    ax.axis("off")

    cell_text = [
        [f["modelo"], f"{f['accuracy_prueba']:.4f}", f"{f['f1_macro_prueba']:.4f}"]
        for f in filas
    ]
    tbl = ax.table(
        cellText=cell_text,
        colLabels=["Modelo", "Accuracy (prueba)", "F1 macro (prueba)"],
        colWidths=[0.46, 0.27, 0.27],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#D9E2F3")

    ax.set_title(
        "Parte 2 — Accuracy y F1 macro en el conjunto de prueba "
        f"(n_train={X_train.shape[0]}, n_test={X_test.shape[0]}; "
        "RL con atributos estandarizados)",
        pad=14,
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig("tabla_parte2.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 7) Resumen
    # ------------------------------------------------------------------
    print("=== T2 — Parte 2: dos clasificadores (métricas en PRUEBA) ===")
    print(tabla.to_string(index=False))
    print("\nArchivos escritos: resultados.json, tabla_parte2.png")


if __name__ == "__main__":
    main()
