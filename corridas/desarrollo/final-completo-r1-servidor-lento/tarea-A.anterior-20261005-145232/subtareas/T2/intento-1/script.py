# -*- coding: utf-8 -*-
"""
T2 — Parte 2 de la Tarea A (Ng & Jordan, 2002): generativo vs discriminativo.

Sobre la división 70/30 estratificada (random_state=42) fijada en T1:
  * Entrena un Naive Bayes gaussiano y una regresión logística con atributos
    estandarizados (StandardScaler ajustado SOLO con el conjunto de entrenamiento).
  * Mide accuracy y F1 macro de cada modelo sobre el conjunto de prueba.
  * Construye la tabla comparativa y la guarda en resultados.json,
    tabla_comparativa_T2.csv y la figura comparativa_modelos_T2.png.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

RUTA_T1 = Path("entrada") / "T1" / "resultados.json"


def main():
    # ------------------------------------------------------------------
    # 1) Recuperar la división única fijada por T1
    # ------------------------------------------------------------------
    with open(RUTA_T1, "r", encoding="utf-8") as f:
        t1 = json.load(f)

    idx_train = np.asarray(t1["indices_train"], dtype=int)
    idx_test = np.asarray(t1["indices_test"], dtype=int)

    # ------------------------------------------------------------------
    # 2) Datos y reconstrucción EXACTA de la división de T1
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X, y = data.data, data.target
    target_names = list(data.target_names)  # 0: malignant, 1: benign

    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]

    n_train, n_test = int(len(idx_train)), int(len(idx_test))
    conteo_train = {target_names[c]: int((y_train == c).sum()) for c in (0, 1)}
    conteo_test = {target_names[c]: int((y_test == c).sum()) for c in (0, 1)}

    verificacion = {
        "n_train_coincide_con_T1": n_train == t1["division"]["n_train"],
        "n_test_coincide_con_T1": n_test == t1["division"]["n_test"],
        "conteo_por_clase_train_coincide": conteo_train == t1["division"]["conteo_por_clase_train"],
        "conteo_por_clase_test_coincide": conteo_test == t1["division"]["conteo_por_clase_test"],
    }
    assert all(verificacion.values()), "La división reconstruida no coincide con T1"

    # ------------------------------------------------------------------
    # 3) Estandarización: el escalador se ajusta SOLO con el entrenamiento
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)  # fit + transform únicamente en train
    X_test_s = scaler.transform(X_test)        # el test nunca participa en el ajuste

    # ------------------------------------------------------------------
    # 4) Modelos
    # ------------------------------------------------------------------
    # Generativo: Naive Bayes gaussiano (sobre atributos estandarizados, como
    # indica el enunciado; por invarianza de escala sus predicciones no cambian)
    nb = GaussianNB()
    nb.fit(X_train_s, y_train)
    pred_nb = nb.predict(X_test_s)

    # Discriminativo: regresión logística
    logreg = LogisticRegression(max_iter=5000, random_state=42)
    logreg.fit(X_train_s, y_train)
    pred_lr = logreg.predict(X_test_s)

    # ------------------------------------------------------------------
    # 5) Métricas sobre el conjunto de prueba
    # ------------------------------------------------------------------
    predicciones = {
        "naive_bayes_gaussiano": ("Naive Bayes gaussiano", pred_nb),
        "regresion_logistica": ("Regresión logística", pred_lr),
    }

    metricas = {}
    for clave, (etiqueta, pred) in predicciones.items():
        metricas[clave] = {
            "modelo": etiqueta,
            "accuracy_prueba": float(accuracy_score(y_test, pred)),
            "f1_macro_prueba": float(f1_score(y_test, pred, average="macro")),
            "matriz_confusion_prueba": confusion_matrix(y_test, pred).tolist(),
        }

    # ------------------------------------------------------------------
    # 6) Tabla comparativa
    # ------------------------------------------------------------------
    tabla_registros = [
        {
            "modelo": metricas[c]["modelo"],
            "accuracy_prueba": metricas[c]["accuracy_prueba"],
            "f1_macro_prueba": metricas[c]["f1_macro_prueba"],
        }
        for c in ("naive_bayes_gaussiano", "regresion_logistica")
    ]
    tabla = pd.DataFrame(tabla_registros)
    tabla.to_csv("tabla_comparativa_T2.csv", index=False)

    # ------------------------------------------------------------------
    # 7) Figura comparativa
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xpos = np.arange(len(tabla))
    w = 0.35
    b1 = ax.bar(xpos - w / 2, tabla["accuracy_prueba"], w, label="Accuracy", color="#4C72B0")
    b2 = ax.bar(xpos + w / 2, tabla["f1_macro_prueba"], w, label="F1 macro", color="#DD8452")
    for barras in (b1, b2):
        for b in barras:
            ax.annotate(
                f"{b.get_height():.4f}",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=9,
            )
    ax.set_xticks(xpos)
    ax.set_xticklabels(["Naive Bayes\ngaussiano", "Regresión\nlogística"])
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Valor en el conjunto de prueba")
    ax.set_title("Parte 2 — Comparación en prueba (división 70/30 de T1, random_state=42)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("comparativa_modelos_T2.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 8) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T2",
        "descripcion": (
            "Naive Bayes gaussiano y regresión logística entrenados sobre la división "
            "70/30 estratificada (random_state=42) de T1; atributos estandarizados con "
            "StandardScaler ajustado solo con el entrenamiento; accuracy y F1 macro "
            "medidos sobre el conjunto de prueba."
        ),
        "division_usada": {
            "fuente": "entrada/T1/resultados.json",
            "random_state": t1["division"]["random_state"],
            "test_size": t1["division"]["test_size"],
            "n_train": n_train,
            "n_test": n_test,
            "conteo_por_clase_train": conteo_train,
            "conteo_por_clase_test": conteo_test,
        },
        "verificacion_division_vs_T1": verificacion,
        "preprocesamiento": {
            "escalador": "StandardScaler",
            "ajustado_solo_con_entrenamiento": True,
            "aplicado_a": ["naive_bayes_gaussiano", "regresion_logistica"],
            "nota": (
                "Para el NB gaussiano la estandarización es invariante de escala y no "
                "altera sus predicciones; se aplica por consistencia con el enunciado."
            ),
        },
        "hiperparametros": {
            "GaussianNB": "valores por defecto (var_smoothing=1e-9)",
            "LogisticRegression": "solver=lbfgs, C=1.0, max_iter=5000, random_state=42",
        },
        "metricas_prueba": metricas,
        "tabla_comparativa": tabla_registros,
        "archivos_generados": [
            "tabla_comparativa_T2.csv",
            "comparativa_modelos_T2.png",
        ],
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 9) Resumen
    # ------------------------------------------------------------------
    print("=" * 64)
    print("T2 — Comparación generativo vs discriminativo (conjunto de prueba)")
    print("=" * 64)
    print(tabla.to_string(index=False))
    print("-" * 64)
    for clave, m in metricas.items():
        print(
            f"{m['modelo']}: accuracy={m['accuracy_prueba']:.4f} | "
            f"F1 macro={m['f1_macro_prueba']:.4f}"
        )
    print(f"División usada: {n_train} train / {n_test} test (de T1, random_state=42)")
    print("Archivos generados: resultados.json, tabla_comparativa_T2.csv, "
          "comparativa_modelos_T2.png")


if __name__ == "__main__":
    main()
