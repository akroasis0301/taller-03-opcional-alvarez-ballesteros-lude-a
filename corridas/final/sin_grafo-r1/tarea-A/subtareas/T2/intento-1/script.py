from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

RUTA_T1 = Path("entrada") / "T1"


def main():
    # ------------------------------------------------------------------
    # 1. Cargar la división única fijada por T1 (70/30 estratificada, rs=42)
    # ------------------------------------------------------------------
    X_train = np.load(RUTA_T1 / "X_train.npy")
    X_test = np.load(RUTA_T1 / "X_test.npy")
    y_train = np.load(RUTA_T1 / "y_train.npy")
    y_test = np.load(RUTA_T1 / "y_test.npy")

    assert X_train.shape[0] == y_train.shape[0], "X_train e y_train no coinciden"
    assert X_test.shape[0] == y_test.shape[0], "X_test e y_test no coinciden"

    n_train, n_atributos = X_train.shape
    n_test = X_test.shape[0]

    # ------------------------------------------------------------------
    # 2. Modelo 1: Naive Bayes gaussiano (atributos originales)
    # ------------------------------------------------------------------
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    pred_gnb = gnb.predict(X_test)
    acc_gnb = float(accuracy_score(y_test, pred_gnb))
    f1_gnb = float(f1_score(y_test, pred_gnb, average="macro"))
    cm_gnb = confusion_matrix(y_test, pred_gnb).tolist()

    # ------------------------------------------------------------------
    # 3. Modelo 2: Regresión logística con atributos estandarizados
    #    (StandardScaler ajustado SOLO con el conjunto de entrenamiento)
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)  # ajuste solo con train
    X_test_std = scaler.transform(X_test)        # misma transformación en prueba

    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train_std, y_train)
    pred_lr = logreg.predict(X_test_std)
    acc_lr = float(accuracy_score(y_test, pred_lr))
    f1_lr = float(f1_score(y_test, pred_lr, average="macro"))
    cm_lr = confusion_matrix(y_test, pred_lr).tolist()

    # ------------------------------------------------------------------
    # 4. Tabla comparativa
    # ------------------------------------------------------------------
    tabla = pd.DataFrame(
        {
            "modelo": [
                "Naive Bayes gaussiano",
                "Regresión logística (atributos estandarizados)",
            ],
            "accuracy_prueba": [acc_gnb, acc_lr],
            "f1_macro_prueba": [f1_gnb, f1_lr],
        }
    )
    tabla.to_csv("tabla_comparativa.csv", index=False)

    # Figura con la tabla comparativa
    celdas = [
        [fila["modelo"],
         f"{fila['accuracy_prueba']:.4f}",
         f"{fila['f1_macro_prueba']:.4f}"]
        for _, fila in tabla.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(8.5, 2.4))
    ax.axis("off")
    tbl = ax.table(
        cellText=celdas,
        colLabels=["Modelo", "Accuracy (prueba)", "F1 macro (prueba)"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    plt.savefig("tabla_comparativa.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Contrato de resultados
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T2",
        "descripcion": (
            "Naive Bayes gaussiano y regresión logística (atributos estandarizados) "
            "entrenados sobre la división 70/30 estratificada de T1 (random_state=42); "
            "métricas medidas en el conjunto de prueba."
        ),
        "division": {
            "origen": "entrada/T1 (única división de la tarea)",
            "n_train": int(n_train),
            "n_test": int(n_test),
            "n_atributos": int(n_atributos),
        },
        "modelos": {
            "naive_bayes_gaussiano": {
                "tipo": "GaussianNB (atributos originales, sin estandarizar)",
                "accuracy_prueba": acc_gnb,
                "f1_macro_prueba": f1_gnb,
                "matriz_confusion_prueba": cm_gnb,
            },
            "regresion_logistica": {
                "tipo": "LogisticRegression",
                "estandarizacion": "StandardScaler ajustado solo con el conjunto de entrenamiento",
                "max_iter": 1000,
                "random_state": 42,
                "accuracy_prueba": acc_lr,
                "f1_macro_prueba": f1_lr,
                "matriz_confusion_prueba": cm_lr,
            },
        },
        "tabla_comparativa": tabla.to_dict(orient="records"),
        "archivos_generados": [
            "resultados.json",
            "tabla_comparativa.csv",
            "tabla_comparativa.png",
        ],
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 6. Resumen
    # ------------------------------------------------------------------
    print("=== T2: Comparación de clasificadores (conjunto de prueba) ===")
    print(f"División de T1: {n_train} entrenamiento / {n_test} prueba, {n_atributos} atributos")
    print(tabla.to_string(index=False))
    print("Mejor accuracy :",
          tabla.loc[tabla["accuracy_prueba"].idxmax(), "modelo"])
    print("Mejor F1 macro :",
          tabla.loc[tabla["f1_macro_prueba"].idxmax(), "modelo"])
    print("Archivos: resultados.json, tabla_comparativa.csv, tabla_comparativa.png")


if __name__ == "__main__":
    main()
