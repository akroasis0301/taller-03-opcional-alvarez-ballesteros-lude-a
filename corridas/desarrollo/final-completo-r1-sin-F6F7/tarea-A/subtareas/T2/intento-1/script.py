#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2 — Dos clasificadores sobre la división de T1 (70/30 estratificada, random_state=42).

Entrena:
  1) Naive Bayes gaussiano
  2) Regresión logística
ambos sobre atributos estandarizados (StandardScaler ajustado SOLO con el
conjunto de entrenamiento) y reporta accuracy y F1 macro de cada uno sobre el
conjunto de prueba, en una tabla.

Salidas:
  - resultados.json          (contrato de la subtarea: tabla con accuracy y F1 macro)
  - t2_tabla_metricas.csv    (la tabla en CSV)
  - t2_metricas_prueba.png   (figura comparativa)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

RUTA_TRAIN = Path("entrada/T1/train.parquet")
RUTA_TEST = Path("entrada/T1/test.parquet")
RUTA_T1_JSON = Path("entrada/T1/resultados.json")


def cargar_conjunto(ruta_parquet):
    """Carga un parquet de T1 y separa (X, y) de forma robusta."""
    df = pd.read_parquet(ruta_parquet)

    # Identificar la columna objetivo
    candidatos = ["target", "y", "label", "clase", "class", "target_name", "diagnosis"]
    target_col = next((c for c in candidatos if c in df.columns), df.columns[-1])

    y_raw = df[target_col]
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.to_numpy().astype(int)
        nombres_clases = [str(v) for v in sorted(np.unique(y))]
    else:
        nombres_clases = sorted(y_raw.astype(str).unique().tolist())
        mapa = {c: i for i, c in enumerate(nombres_clases)}
        y = y_raw.astype(str).map(mapa).to_numpy().astype(int)

    # Atributos: todas las columnas numéricas distintas del objetivo
    feature_cols = [
        c for c in df.columns
        if c != target_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[feature_cols].to_numpy(dtype=float)
    return X, y, feature_cols, nombres_clases


def main():
    # ---------- 1. Cargar la única división de T1 ----------
    X_train, y_train, feats_tr, clases_cod = cargar_conjunto(RUTA_TRAIN)
    X_test, y_test, feats_te, clases_cod_te = cargar_conjunto(RUTA_TEST)

    if feats_tr != feats_te:
        raise ValueError("Los atributos de train y test de T1 no coinciden.")
    if clases_cod != clases_cod_te:
        raise ValueError("Las clases de train y test de T1 no coinciden.")

    # Metadatos de la división de T1 (si están disponibles)
    info_t1 = {}
    nombres_clases = clases_cod
    if RUTA_T1_JSON.exists():
        try:
            t1 = json.loads(RUTA_T1_JSON.read_text(encoding="utf-8"))
            info_t1 = t1.get("division", {})
            if "clases" in t1 and len(t1["clases"]) == len(clases_cod):
                nombres_clases = t1["clases"]
        except Exception:
            info_t1 = {}

    # ---------- 2. Estandarización ajustada SOLO con el entrenamiento ----------
    scaler = StandardScaler().fit(X_train)   # el test nunca participa en el ajuste
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ---------- 3. Entrenar y evaluar los dos modelos ----------
    modelos = {
        "Naive Bayes gaussiano": GaussianNB(),
        "Regresión logística": LogisticRegression(max_iter=5000, random_state=42),
    }

    filas = []
    for nombre, modelo in modelos.items():
        modelo.fit(X_train_s, y_train)
        y_pred = modelo.predict(X_test_s)
        acc = float(accuracy_score(y_test, y_pred))
        f1m = float(f1_score(y_test, y_pred, average="macro"))
        cm = confusion_matrix(y_test, y_pred, labels=sorted(np.unique(y_test)))
        filas.append({
            "modelo": nombre,
            "accuracy": acc,
            "f1_macro": f1m,
            "matriz_confusion_prueba": cm.tolist(),
        })

    # ---------- 4. Tabla y resultados.json ----------
    tabla = pd.DataFrame(filas)[["modelo", "accuracy", "f1_macro"]]
    tabla.to_csv("t2_tabla_metricas.csv", index=False, encoding="utf-8")

    lineas = ["| modelo | accuracy | f1_macro |", "|---|---|---|"]
    for f in filas:
        lineas.append(f"| {f['modelo']} | {f['accuracy']:.6f} | {f['f1_macro']:.6f} |")

    resultados = {
        "subtarea": "T2",
        "descripcion": (
            "Naive Bayes gaussiano y regresión logística entrenados sobre la división "
            "70/30 estratificada de T1 (random_state=42); accuracy y F1 macro medidos "
            "sobre el conjunto de prueba."
        ),
        "division_usada": {
            "origen": "entrada/T1 (train.parquet, test.parquet)",
            "test_size": 0.3,
            "train_size": 0.7,
            "estratificada_por_clase": True,
            "random_state": 42,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "info_t1": info_t1,
        },
        "preprocesamiento": {
            "estandarizacion": (
                "StandardScaler ajustado únicamente con el conjunto de entrenamiento; "
                "el mismo escalador transforma el conjunto de prueba."
            ),
            "nota_nb": (
                "GaussianNB es invariante a transformaciones afines por atributo, por lo "
                "que estandarizar no altera sus predicciones; se usa la misma matriz "
                "estandarizada para ambos modelos."
            ),
        },
        "clases": nombres_clases,
        "clases_codificadas": clases_cod,
        "num_atributos": int(X_train.shape[1]),
        "tabla_metricas_prueba": filas,
        "tabla_markdown": "\n".join(lineas),
        "archivos_generados": ["t2_tabla_metricas.csv", "t2_metricas_prueba.png"],
    }
    Path("resultados.json").write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------- 5. Figura comparativa ----------
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(filas))
    ancho = 0.35
    accs = [f["accuracy"] for f in filas]
    f1s = [f["f1_macro"] for f in filas]
    b1 = ax.bar(x - ancho / 2, accs, ancho, label="Accuracy", color="#4C72B0")
    b2 = ax.bar(x + ancho / 2, f1s, ancho, label="F1 macro", color="#DD8452")
    for barras in (b1, b2):
        for b in barras:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f["modelo"] for f in filas])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Valor en el conjunto de prueba")
    ax.set_title("T2 — Accuracy y F1 macro en prueba\n(división 70/30 de T1, random_state=42)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("t2_metricas_prueba.png", dpi=120)
    plt.close(fig)

    # ---------- 6. Resumen ----------
    print("T2 — Métricas en el conjunto de prueba (división 70/30 de T1, random_state=42)")
    print(f"n_train={X_train.shape[0]}  n_test={X_test.shape[0]}  "
          f"atributos={X_train.shape[1]}  clases={nombres_clases}")
    print(f"{'modelo':<25}{'accuracy':>12}{'f1_macro':>12}")
    for f in filas:
        print(f"{f['modelo']:<25}{f['accuracy']:>12.6f}{f['f1_macro']:>12.6f}")
    print("Archivos escritos: resultados.json, t2_tabla_metricas.csv, t2_metricas_prueba.png")


if __name__ == "__main__":
    main()
