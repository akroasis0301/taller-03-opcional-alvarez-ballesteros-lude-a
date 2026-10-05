#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 — Curva de aprendizaje: generativo vs discriminativo con pocos datos.

Sobre la MISMA división 70/30 estratificada de T1 (entrada/T1/train.parquet y
entrada/T1/test.parquet) y los MISMOS dos modelos de T2 (Naive Bayes gaussiano y
regresión logística con estandarización ajustada solo con datos de entrenamiento),
se entrena con submuestras estratificadas del 5 %, 10 %, 25 %, 50 % y 100 % del
conjunto de entrenamiento (random_state=42) y se evalúa cada tamaño SIEMPRE sobre
el mismo conjunto de prueba. Se grafica exactitud contra número de ejemplos de
entrenamiento (una curva por modelo) y se guarda la figura como PNG.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42
FRACCIONES = [0.05, 0.10, 0.25, 0.50, 1.00]
NOMBRE_CLASE = {0: "malignant", 1: "benign"}
RUTA_TRAIN = Path("entrada/T1/train.parquet")
RUTA_TEST = Path("entrada/T1/test.parquet")
RUTA_FIGURA = "t3_curva_aprendizaje.png"
RUTA_RESULTADOS = "resultados.json"


def cargar_parquet(ruta, columnas_ref=None):
    """Lee un parquet de T1 y devuelve (X, y, nombres_de_columnas). y: 0=malignant, 1=benign."""
    df = pd.read_parquet(ruta)
    tcol = None
    for cand in ["target", "target_name", "diagnosis", "clase", "label", "y"]:
        if cand in df.columns:
            tcol = cand
            break
    if tcol is None:
        tcol = df.columns[-1]
    y_raw = df[tcol]
    X_df = df.drop(columns=[tcol])
    if columnas_ref is not None:
        X_df = X_df[columnas_ref]
    if y_raw.dtype == object:
        y = y_raw.map({"malignant": 0, "benign": 1})
        if y.isna().any():
            uniq = sorted(y_raw.astype(str).unique())
            y = y_raw.astype(str).map({u: i for i, u in enumerate(uniq)})
    else:
        y = y_raw
    y = y.astype(int).to_numpy()
    X_num = X_df.select_dtypes(include=[np.number])
    X = X_num.to_numpy(dtype=float)
    return X, y, list(X_num.columns)


def nuevos_modelos():
    """Los mismos dos modelos de T2 (instancias frescas por corrida)."""
    return [
        ("Naive Bayes gaussiano", GaussianNB()),
        ("Regresión logística", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]


def main():
    if not RUTA_TRAIN.exists() or not RUTA_TEST.exists():
        raise FileNotFoundError("No se encontraron entrada/T1/train.parquet o entrada/T1/test.parquet")

    X_train, y_train, cols = cargar_parquet(RUTA_TRAIN)
    X_test, y_test, _ = cargar_parquet(RUTA_TEST, columnas_ref=cols)
    n_train_total = int(len(y_train))
    n_test = int(len(y_test))

    curvas = {
        nombre: {"n_train": [], "accuracy": [], "f1_macro": []}
        for nombre, _ in nuevos_modelos()
    }
    composiciones = []
    registros = []

    for frac in FRACCIONES:
        # Submuestra estratificada del entrenamiento (el test nunca interviene aquí)
        if frac >= 1.0:
            X_sub, y_sub = X_train, y_train
        else:
            X_sub, _, y_sub, _ = train_test_split(
                X_train, y_train,
                train_size=frac,
                stratify=y_train,
                random_state=RANDOM_STATE,
            )
        n_sub = int(len(y_sub))
        comp = {}
        for c in np.unique(y_train):
            c = int(c)
            comp[NOMBRE_CLASE[c]] = int((y_sub == c).sum())
        composiciones.append({"fraccion": frac, "n_train": n_sub, "conteo_clases": comp})

        # El escalador se ajusta SOLO con los datos de entrenamiento de esta corrida
        # (la submuestra); el test únicamente se transforma, nunca se ajusta.
        scaler = StandardScaler().fit(X_sub)
        X_sub_s = scaler.transform(X_sub)
        X_test_s = scaler.transform(X_test)

        for nombre, modelo in nuevos_modelos():
            modelo.fit(X_sub_s, y_sub)
            pred = modelo.predict(X_test_s)
            acc = float(accuracy_score(y_test, pred))
            f1m = float(f1_score(y_test, pred, average="macro"))
            curvas[nombre]["n_train"].append(n_sub)
            curvas[nombre]["accuracy"].append(acc)
            curvas[nombre]["f1_macro"].append(f1m)
            registros.append({
                "fraccion": frac,
                "n_train": n_sub,
                "modelo": nombre,
                "accuracy_prueba": acc,
                "f1_macro_prueba": f1m,
            })

    # ---------- Figura: exactitud contra número de ejemplos de entrenamiento ----------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    estilos = {
        "Naive Bayes gaussiano": {"marker": "o", "color": "tab:blue", "dy": 8},
        "Regresión logística": {"marker": "s", "color": "tab:red", "dy": -14},
    }
    for nombre, curva in curvas.items():
        est = estilos[nombre]
        ax.plot(curva["n_train"], curva["accuracy"],
                marker=est["marker"], color=est["color"], label=nombre)
        for n, a in zip(curva["n_train"], curva["accuracy"]):
            ax.annotate(f"{a:.3f}", (n, a), textcoords="offset points",
                        xytext=(0, est["dy"]), ha="center", fontsize=8, color=est["color"])
    ax.set_xticks(curvas["Naive Bayes gaussiano"]["n_train"])
    ax.set_xlabel("Número de ejemplos de entrenamiento")
    ax.set_ylabel("Exactitud (accuracy) en el conjunto de prueba")
    ax.set_title("Curva de aprendizaje: Naive Bayes gaussiano vs regresión logística\n"
                 "(misma división 70/30 de T1; evaluación siempre sobre el mismo test)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RUTA_FIGURA, dpi=120)
    plt.close(fig)

    # ---------- Tabla markdown ----------
    lineas = [
        "| fracción | n_train | accuracy Naive Bayes gaussiano | accuracy Regresión logística |",
        "|---|---|---|---|",
    ]
    for i, frac in enumerate(FRACCIONES):
        n = curvas["Naive Bayes gaussiano"]["n_train"][i]
        nb = curvas["Naive Bayes gaussiano"]["accuracy"][i]
        lr = curvas["Regresión logística"]["accuracy"][i]
        lineas.append(f"| {int(round(frac * 100))} % | {n} | {nb:.6f} | {lr:.6f} |")
    tabla_md = "\n".join(lineas)

    resultados = {
        "subtarea": "T3",
        "descripcion": (
            "Curva de aprendizaje sobre la misma división 70/30 estratificada de T1 "
            "(random_state=42): Naive Bayes gaussiano y regresión logística (los dos modelos "
            "de T2) entrenados con submuestras estratificadas del 5 %, 10 %, 25 %, 50 % y 100 % "
            "del entrenamiento (random_state=42) y evaluados siempre sobre el mismo conjunto "
            "de prueba."
        ),
        "division_usada": {
            "origen": "entrada/T1 (train.parquet, test.parquet)",
            "n_train": n_train_total,
            "n_test": n_test,
            "estratificada_por_clase": True,
            "random_state": 42,
        },
        "submuestreo": {
            "metodo": "train_test_split con stratify=y_train y random_state=42 "
                      "(el 100 % usa el entrenamiento completo)",
            "random_state": 42,
            "fracciones": FRACCIONES,
            "composicion_por_fraccion": composiciones,
        },
        "tamanos_entrenamiento": curvas["Naive Bayes gaussiano"]["n_train"],
        "curva_aprendizaje": {
            nombre: {
                "n_train": curva["n_train"],
                "accuracy_prueba": curva["accuracy"],
                "f1_macro_prueba": curva["f1_macro"],
            }
            for nombre, curva in curvas.items()
        },
        "accuracy_por_modelo_y_tamano": {
            nombre: {str(n): a for n, a in zip(curva["n_train"], curva["accuracy"])}
            for nombre, curva in curvas.items()
        },
        "tabla_registros": registros,
        "tabla_markdown": tabla_md,
        "figura": RUTA_FIGURA,
        "archivos_generados": [RUTA_FIGURA],
    }
    with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print("T3 — Curva de aprendizaje (evaluación siempre sobre el mismo test de T1)")
    print(f"n_train total: {n_train_total} | n_test: {n_test}")
    print(tabla_md)
    print(f"Figura guardada en: {RUTA_FIGURA}")
    print(f"Resultados escritos en: {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
