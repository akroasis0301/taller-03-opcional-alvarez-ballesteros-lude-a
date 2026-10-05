# -*- coding: utf-8 -*-
"""
T2 — Dos clasificadores sobre la división de T1 (Breast Cancer Wisconsin).

Entrena:
  1) Naive Bayes gaussiano (GaussianNB, atributos originales).
  2) Regresión logística con atributos estandarizados
     (StandardScaler ajustado SOLO con el conjunto de entrenamiento).

Mide accuracy y F1 macro de cada modelo ÚNICAMENTE sobre el conjunto de prueba
(70/30 estratificada, random_state=42, definida en T1) y escribe la tabla en
resultados.json. El conjunto de prueba no se usa para ajustar nada y no se
reportan métricas sobre los datos de entrenamiento (evita evaluar sobre las
mismas variables con las que se ajustó el modelo).
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

SEED = 42
DIR_T1 = os.path.join("entrada", "T1")


def cargar_np(nombre):
    return np.load(os.path.join(DIR_T1, nombre))


def cargar_division():
    """Carga la división 70/30 estratificada de T1 (con reserva de reconstrucción)."""
    # 1) Archivos directos de train/test guardados por T1
    try:
        return (cargar_np("X_train.npy"), cargar_np("X_test.npy"),
                cargar_np("y_train.npy"), cargar_np("y_test.npy"))
    except FileNotFoundError:
        pass
    # 2) Reconstruir con X_full/y_full + índices de T1
    try:
        X_full, y_full = cargar_np("X_full.npy"), cargar_np("y_full.npy")
        idx_tr = cargar_np("idx_train.npy").astype(int)
        idx_te = cargar_np("idx_test.npy").astype(int)
        return X_full[idx_tr], X_full[idx_te], y_full[idx_tr], y_full[idx_te]
    except FileNotFoundError:
        pass
    # 3) Última reserva: regenerar el dataset y repetir el split de T1
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    data = load_breast_cancer()
    idx_tr, idx_te = train_test_split(
        np.arange(data.target.shape[0]),
        test_size=0.3, stratify=data.target, random_state=SEED,
    )
    return (data.data[idx_tr], data.data[idx_te],
            data.target[idx_tr], data.target[idx_te])


def metricas(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def main():
    # ---------- Datos: división de T1 ----------
    X_train, X_test, y_train, y_test = cargar_division()
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # ---------- Modelo 1: Naive Bayes gaussiano ----------
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    # Evaluación EXCLUSIVAMENTE en el conjunto de prueba (nunca visto en el ajuste)
    pred_gnb_test = gnb.predict(X_test)

    # ---------- Modelo 2: Regresión logística con estandarización ----------
    # El escalador se ajusta ÚNICAMENTE con el entrenamiento.
    scaler = StandardScaler().fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)  # test solo se transforma, nunca se ajusta
    logreg = LogisticRegression(max_iter=5000, random_state=SEED)
    logreg.fit(X_train_std, y_train)
    # Evaluación EXCLUSIVAMENTE en el conjunto de prueba (nunca visto en el ajuste)
    pred_lr_test = logreg.predict(X_test_std)

    # ---------- Métricas en prueba (única evaluación reportada) ----------
    m_gnb = metricas(y_test, pred_gnb_test)
    m_lr = metricas(y_test, pred_lr_test)

    tabla = pd.DataFrame([
        {"modelo": "Naive Bayes gaussiano",
         "accuracy": m_gnb["accuracy"], "f1_macro": m_gnb["f1_macro"]},
        {"modelo": "Regresión logística (atributos estandarizados)",
         "accuracy": m_lr["accuracy"], "f1_macro": m_lr["f1_macro"]},
    ])

    # ---------- resultados.json (contrato de la subtarea) ----------
    resultados = {
        "subtarea": "T2",
        "descripcion": (
            "Naive Bayes gaussiano y regresión logística (con estandarización "
            "ajustada solo con el entrenamiento) evaluados ÚNICAMENTE sobre el "
            "conjunto de prueba de la división de T1 (70/30 estratificada, "
            "random_state=42). No se reportan métricas sobre los datos de "
            "entrenamiento."
        ),
        "dataset": "Breast Cancer Wisconsin (Diagnostic) — sklearn.datasets.load_breast_cancer",
        "division_usada": "T1: train_test_split(test_size=0.3, stratify=y, random_state=42)",
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_atributos": int(X_train.shape[1]),
        "conjunto_evaluacion": "test (los modelos nunca se evalúan sobre train)",
        "tabla_prueba": tabla.to_dict(orient="records"),
        "metricas_prueba": {
            "gaussian_nb": m_gnb,
            "logistic_regression": m_lr,
        },
        "matrices_confusion_prueba": {
            "gaussian_nb": confusion_matrix(y_test, pred_gnb_test).tolist(),
            "logistic_regression": confusion_matrix(y_test, pred_lr_test).tolist(),
        },
        "detalles_modelos": {
            "gaussian_nb": {
                "clase": "sklearn.naive_bayes.GaussianNB",
                "parametros": "por defecto; atributos originales (sin estandarizar)",
            },
            "logistic_regression": {
                "clase": "sklearn.linear_model.LogisticRegression",
                "parametros": {"max_iter": 5000, "random_state": SEED,
                               "resto": "por defecto"},
                "estandarizacion": ("StandardScaler ajustado únicamente con "
                                    "X_train; X_test solo transformado"),
            },
        },
        "archivos_generados": [
            "resultados.json",
            "t2_tabla_metricas_prueba.csv",
            "t2_metricas_prueba.png",
        ],
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ---------- Tabla en CSV (comodín) ----------
    tabla.to_csv("t2_tabla_metricas_prueba.csv", index=False, encoding="utf-8")

    # ---------- Figura: barras agrupadas accuracy / F1 macro ----------
    etiquetas = ["Naive Bayes\ngaussiano", "Regresión\nlogística"]
    x = np.arange(len(etiquetas))
    ancho = 0.35
    accs = [m_gnb["accuracy"], m_lr["accuracy"]]
    f1s = [m_gnb["f1_macro"], m_lr["f1_macro"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - ancho / 2, accs, ancho, label="Accuracy", color="#4C72B0")
    b2 = ax.bar(x + ancho / 2, f1s, ancho, label="F1 macro", color="#DD8452")
    ax.bar_label(b1, fmt="%.3f")
    ax.bar_label(b2, fmt="%.3f")
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Valor en el conjunto de prueba")
    ax.set_title("T2 — Desempeño en prueba (división T1: 70/30 estratificada)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("t2_metricas_prueba.png", dpi=120)
    plt.close(fig)

    # ---------- Resumen ----------
    print("T2 — Métricas en el conjunto de prueba "
          "(división T1: 70/30 estratificada, random_state=42)")
    print(tabla.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\nMatrices de confusión en prueba (clases: 0=malignant, 1=benign):")
    print("  GaussianNB:", resultados["matrices_confusion_prueba"]["gaussian_nb"])
    print("  LogReg    :", resultados["matrices_confusion_prueba"]["logistic_regression"])
    print("\nArchivos escritos: resultados.json, "
          "t2_tabla_metricas_prueba.csv, t2_metricas_prueba.png")


if __name__ == "__main__":
    main()
