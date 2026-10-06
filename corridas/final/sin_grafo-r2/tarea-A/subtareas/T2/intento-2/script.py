# T2 — Entrenar Naive Bayes gaussiano y regresión logística sobre la división 70/30
# de la Parte 1 (estratificada, random_state=42) y medir accuracy y F1 macro en PRUEBA.
# La regresión logística usa atributos estandarizados con un escalador ajustado
# SOLO con el conjunto de entrenamiento.

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


def main():
    # ------------------------------------------------------------------
    # 1) Dataset y reproducción EXACTA de la división de la Parte 1 (T1)
    #    Misma llamada que en T1: test_size=0.3, estratificada, random_state=42.
    #    La división es determinista, por lo que se reproduce idéntica.
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    verif = {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "train_clase_0_malignant": int((y_train == 0).sum()),
        "train_clase_1_benign": int((y_train == 1).sum()),
        "test_clase_0_malignant": int((y_test == 0).sum()),
        "test_clase_1_benign": int((y_test == 1).sum()),
    }

    # Verificación contra los conteos reportados por T1 (si están disponibles)
    t1_path = Path("entrada/T1/resultados.json")
    if t1_path.exists():
        t1 = json.loads(t1_path.read_text(encoding="utf-8"))
        div = t1.get("division", {})
        esperado = {
            "n_train": div.get("n_train"),
            "n_test": div.get("n_test"),
            "train_clase_0_malignant": div.get("conteo_train_por_clase", {}).get("clase_0_malignant"),
            "train_clase_1_benign": div.get("conteo_train_por_clase", {}).get("clase_1_benign"),
            "test_clase_0_malignant": div.get("conteo_test_por_clase", {}).get("clase_0_malignant"),
            "test_clase_1_benign": div.get("conteo_test_por_clase", {}).get("clase_1_benign"),
        }
    else:
        # Valores reportados por T1 (respaldo)
        esperado = {
            "n_train": 398, "n_test": 171,
            "train_clase_0_malignant": 148, "train_clase_1_benign": 250,
            "test_clase_0_malignant": 64, "test_clase_1_benign": 107,
        }

    coincide = all(esperado[k] == verif[k] for k in esperado)
    if not coincide:
        raise RuntimeError(f"La división reproducida no coincide con T1: {verif} vs {esperado}")

    # ------------------------------------------------------------------
    # 2) Escalador ajustado SOLO con el conjunto de entrenamiento
    #    (el conjunto de prueba no se usa para ajustar nada)
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)  # fit solo con train
    X_test_std = scaler.transform(X_test)        # test solo se transforma

    # ------------------------------------------------------------------
    # 3) Modelos
    # ------------------------------------------------------------------
    # Naive Bayes gaussiano sobre los atributos originales (el enunciado solo
    # exige estandarización para la regresión logística).
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    # Regresión logística sobre atributos estandarizados
    lr = LogisticRegression(max_iter=5000, random_state=42)
    lr.fit(X_train_std, y_train)

    # ------------------------------------------------------------------
    # 4) Evaluación en el conjunto de PRUEBA (accuracy y F1 macro)
    # ------------------------------------------------------------------
    filas = []
    for nombre, modelo, X_eval in [
        ("Naive Bayes gaussiano", nb, X_test),
        ("Regresión logística (atributos estandarizados)", lr, X_test_std),
    ]:
        y_pred = modelo.predict(X_eval)
        filas.append(
            {
                "modelo": nombre,
                "accuracy_test": float(accuracy_score(y_test, y_pred)),
                "f1_macro_test": float(f1_score(y_test, y_pred, average="macro")),
            }
        )

    tabla = pd.DataFrame(filas)

    # ------------------------------------------------------------------
    # 5) resultados.json (contrato de la subtarea: las 4 cifras + tabla)
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T2",
        "descripcion": (
            "Naive Bayes gaussiano y regresión logística (atributos estandarizados, "
            "escalador ajustado solo con el entrenamiento) entrenados sobre la división "
            "70/30 estratificada de la Parte 1; accuracy y F1 macro medidos en el "
            "conjunto de prueba."
        ),
        "division_usada": {
            "test_size": 0.3,
            "estratificado": True,
            "random_state": 42,
            "n_train": verif["n_train"],
            "n_test": verif["n_test"],
            "coincide_con_T1": bool(coincide),
        },
        "preprocesado": {
            "naive_bayes_gaussiano": "atributos originales",
            "regresion_logistica": "StandardScaler ajustado únicamente con el entrenamiento",
        },
        "tabla_prueba": filas,
        "accuracy_f1_macro_prueba": {
            "naive_bayes_gaussiano": {
                "accuracy": filas[0]["accuracy_test"],
                "f1_macro": filas[0]["f1_macro_test"],
            },
            "regresion_logistica": {
                "accuracy": filas[1]["accuracy_test"],
                "f1_macro": filas[1]["f1_macro_test"],
            },
        },
    }
    Path("resultados.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # 6) Tabla de resultados como PNG (respaldo visual)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    ax.axis("off")
    tbl = ax.table(
        cellText=[
            [r["modelo"], f"{r['accuracy_test']:.4f}", f"{r['f1_macro_test']:.4f}"]
            for r in filas
        ],
        colLabels=["Modelo", "Accuracy (prueba)", "F1 macro (prueba)"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.7)
    ax.set_title("Parte 2 — Exactitud y F1 macro en el conjunto de prueba", pad=14)
    fig.tight_layout()
    plt.savefig("T2_tabla_prueba.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 7) Resumen
    # ------------------------------------------------------------------
    print("T2 — Resultados en el conjunto de prueba (división 70/30 estratificada, random_state=42):")
    print(tabla.to_string(index=False))
    print("Verificación de la división contra T1:", "OK" if coincide else "FALLÓ")
    print("Archivos escritos: resultados.json, T2_tabla_prueba.png")


if __name__ == "__main__":
    main()
