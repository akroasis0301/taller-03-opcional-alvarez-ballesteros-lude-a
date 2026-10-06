#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T3 — Curva de aprendizaje: Naive Bayes gaussiano vs. regresión logística.

Reutiliza la MISMA división 70/30 estratificada (random_state=42) de la Parte 1,
entrena los dos modelos de la Parte 2 con el 5 %, 10 %, 25 %, 50 % y 100 % del
conjunto de entrenamiento (submuestras estratificadas, random_state=42), evalúa
cada uno siempre sobre el mismo conjunto de prueba y dibuja la exactitud contra
el número de ejemplos de entrenamiento con una curva por modelo (PNG).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

SEMILLA = 42
FRACCIONES = [0.05, 0.10, 0.25, 0.50, 1.00]
RUTA_TRAIN = Path("entrada/T1/t1_train.csv")
RUTA_TEST = Path("entrada/T1/t1_test.csv")
FIG_CURVAS = "t3_curvas_aprendizaje.png"
FIG_TABLA = "t3_tabla_accuracy.png"


# ----------------------------------------------------------------- datos ----
def _col_objetivo(df, feature_names):
    for c in ("target", "diagnosis", "clase", "class", "y", "label", "objetivo"):
        if c in df.columns:
            return c
    conocidos = set(map(str, feature_names))
    no_atributos = [c for c in df.columns if str(c) not in conocidos]
    return no_atributos[-1] if no_atributos else df.columns[-1]


def _y_binaria(serie):
    if serie.dtype == object:
        serie = serie.map({"malignant": 0, "benign": 1})
    return serie.astype(int)


def _separar_xy(df, col_obj, feature_names):
    cols = [c for c in df.columns
            if c != col_obj and not str(c).startswith("Unnamed") and str(c).strip() != ""]
    canon = [c for c in feature_names if c in cols]
    if len(canon) == len(feature_names):
        cols = canon
    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isnan(X).any():
        raise ValueError("atributos no numéricos en el CSV")
    y = _y_binaria(df[col_obj]).to_numpy()
    return X, y


def cargar_desde_csv():
    ds = load_breast_cancer()
    df_tr = pd.read_csv(RUTA_TRAIN)
    df_te = pd.read_csv(RUTA_TEST)
    col = _col_objetivo(df_tr, ds.feature_names)
    X_tr, y_tr = _separar_xy(df_tr, col, ds.feature_names)
    X_te, y_te = _separar_xy(df_te, col, ds.feature_names)
    origen = (f"archivos de la Parte 1: {RUTA_TRAIN.as_posix()} y {RUTA_TEST.as_posix()} "
              f"(columna objetivo: '{col}')")
    return X_tr, y_tr, X_te, y_te, origen


def cargar_regenerando():
    ds = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.data, ds.target, test_size=0.3, stratify=ds.target, random_state=SEMILLA)
    origen = ("división regenerada de forma determinista (load_breast_cancer + "
              "train_test_split(test_size=0.3, stratify=y, random_state=42)), "
              "idéntica a la de la Parte 1")
    return X_tr, y_tr, X_te, y_te, origen


def cargar_split():
    if RUTA_TRAIN.exists() and RUTA_TEST.exists():
        try:
            X_tr, y_tr, X_te, y_te, origen = cargar_desde_csv()
            if (X_tr.shape == (398, 30) and X_te.shape == (171, 30)
                    and set(np.unique(y_tr)).issubset({0, 1})
                    and set(np.unique(y_te)).issubset({0, 1})):
                return X_tr, y_tr, X_te, y_te, origen
        except Exception:
            pass
    return cargar_regenerando()


# ------------------------------------------------------------------ main ----
def main():
    X_tr, y_tr, X_te, y_te, origen = cargar_split()
    n_train_total, n_test = int(X_tr.shape[0]), int(X_te.shape[0])

    filas = []
    curvas = {
        "naive_bayes_gaussiano": {"n_train": [], "accuracy_prueba": [], "f1_macro_prueba": []},
        "regresion_logistica": {"n_train": [], "accuracy_prueba": [], "f1_macro_prueba": []},
    }
    conteos_sub = []

    for frac in FRACCIONES:
        # Submuestra estratificada del entrenamiento (random_state=42);
        # con 100 % se usa el conjunto de entrenamiento completo.
        if frac >= 1.0:
            X_sub, y_sub = X_tr, y_tr
        else:
            X_sub, _xr, y_sub, _yr = train_test_split(
                X_tr, y_tr, train_size=frac, stratify=y_tr, random_state=SEMILLA)

        n_sub = int(X_sub.shape[0])
        conteos_sub.append({
            "fraccion": frac,
            "n_train": n_sub,
            "malignant": int(np.sum(y_sub == 0)),
            "benign": int(np.sum(y_sub == 1)),
        })

        # Modelo generativo: Naive Bayes gaussiano (atributos crudos)
        nb = GaussianNB()
        nb.fit(X_sub, y_sub)
        pred_nb = nb.predict(X_te)

        # Modelo discriminativo: regresión logística con atributos estandarizados;
        # el escalador se ajusta SOLO con la submuestra de entrenamiento.
        lr = Pipeline([
            ("escalador", StandardScaler()),
            ("regresion_logistica", LogisticRegression(max_iter=1000)),
        ])
        lr.fit(X_sub, y_sub)
        pred_lr = lr.predict(X_te)

        acc_nb = float(accuracy_score(y_te, pred_nb))
        f1_nb = float(f1_score(y_te, pred_nb, average="macro"))
        acc_lr = float(accuracy_score(y_te, pred_lr))
        f1_lr = float(f1_score(y_te, pred_lr, average="macro"))

        curvas["naive_bayes_gaussiano"]["n_train"].append(n_sub)
        curvas["naive_bayes_gaussiano"]["accuracy_prueba"].append(acc_nb)
        curvas["naive_bayes_gaussiano"]["f1_macro_prueba"].append(f1_nb)
        curvas["regresion_logistica"]["n_train"].append(n_sub)
        curvas["regresion_logistica"]["accuracy_prueba"].append(acc_lr)
        curvas["regresion_logistica"]["f1_macro_prueba"].append(f1_lr)

        filas.append({
            "fraccion": frac,
            "n_train": n_sub,
            "accuracy_naive_bayes_gaussiano": acc_nb,
            "accuracy_regresion_logistica": acc_lr,
            "f1_macro_naive_bayes_gaussiano": f1_nb,
            "f1_macro_regresion_logistica": f1_lr,
        })

    tamanos = [f["n_train"] for f in filas]

    # ¿Qué modelo gana en cada tamaño?
    mejor_por_tamano = []
    for f in filas:
        if f["accuracy_naive_bayes_gaussiano"] >= f["accuracy_regresion_logistica"]:
            mejor, acc = "naive_bayes_gaussiano", f["accuracy_naive_bayes_gaussiano"]
        else:
            mejor, acc = "regresion_logistica", f["accuracy_regresion_logistica"]
        mejor_por_tamano.append({"n_train": f["n_train"], "mejor_modelo": mejor, "accuracy": acc})

    # Cruce de curvas (insumo cualitativo para la Parte 4)
    dif = [f["accuracy_regresion_logistica"] - f["accuracy_naive_bayes_gaussiano"] for f in filas]
    tam_nb = [f["n_train"] for f, d in zip(filas, dif) if d < 0]
    tam_lr = [f["n_train"] for f, d in zip(filas, dif) if d > 0]
    tam_emp = [f["n_train"] for f, d in zip(filas, dif) if d == 0]
    if tam_nb and tam_lr and max(tam_nb) < min(tam_lr):
        cruce = (f"Se observa el cruce de curvas: el Naive Bayes gaussiano supera a la regresión "
                 f"logística con n_train={tam_nb} y la regresión logística supera al Naive Bayes "
                 f"con n_train={tam_lr} (patrón consistente con Ng y Jordan, 2002).")
    elif not tam_nb:
        base = ("La regresión logística iguala o supera al Naive Bayes gaussiano en todos los "
                f"tamaños evaluados (empates en n_train={tam_emp})." if tam_emp else
                "La regresión logística supera al Naive Bayes gaussiano en todos los tamaños evaluados.")
        cruce = base + " No se observa cruce de curvas en el rango probado."
    else:
        cruce = (f"El Naive Bayes gaussiano supera a la regresión logística con n_train={tam_nb} "
                 f"(empates: {tam_emp}); no se observa cruce de curvas en el rango probado.")

    # ------------------------------------------------------ figura curvas ----
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(curvas["naive_bayes_gaussiano"]["n_train"],
            curvas["naive_bayes_gaussiano"]["accuracy_prueba"],
            marker="o", color="#1f77b4", linewidth=2,
            label="Naive Bayes gaussiano (generativo)")
    ax.plot(curvas["regresion_logistica"]["n_train"],
            curvas["regresion_logistica"]["accuracy_prueba"],
            marker="s", color="#d62728", linewidth=2,
            label="Regresión logística (discriminativo)")
    for n, a in zip(curvas["naive_bayes_gaussiano"]["n_train"],
                    curvas["naive_bayes_gaussiano"]["accuracy_prueba"]):
        ax.annotate(f"{a:.3f}", (n, a), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=8, color="#1f77b4")
    for n, a in zip(curvas["regresion_logistica"]["n_train"],
                    curvas["regresion_logistica"]["accuracy_prueba"]):
        ax.annotate(f"{a:.3f}", (n, a), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#d62728")
    todos = [a for m in curvas.values() for a in m["accuracy_prueba"]]
    ax.set_ylim(max(0.5, min(todos) - 0.07), 1.015)
    ax.set_xticks(tamanos)
    ax.set_xlabel("Número de ejemplos de entrenamiento")
    ax.set_ylabel("Exactitud (accuracy) en el conjunto de prueba")
    ax.set_title("Curva de aprendizaje: Naive Bayes gaussiano vs. regresión logística\n"
                 "Breast Cancer Wisconsin — división 70/30 estratificada, random_state=42")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig(FIG_CURVAS, dpi=120)
    plt.close(fig)

    # ------------------------------------------------------- figura tabla ----
    fig2, ax2 = plt.subplots(figsize=(8.5, 3.6))
    ax2.axis("off")
    columnas = ["% entrenamiento", "n entrenamiento",
                "Accuracy NB gaussiano", "Accuracy reg. logística"]
    celdas = [[f"{f['fraccion'] * 100:.0f}%", f["n_train"],
               f"{f['accuracy_naive_bayes_gaussiano']:.4f}",
               f"{f['accuracy_regresion_logistica']:.4f}"] for f in filas]
    tabla = ax2.table(cellText=celdas, colLabels=columnas, loc="center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 1.7)
    fig2.suptitle("Accuracy en prueba por tamaño de entrenamiento y por modelo "
                  "(5 tamaños × 2 modelos)", y=0.98)
    plt.savefig(FIG_TABLA, dpi=120, bbox_inches="tight")
    plt.close(fig2)

    # ------------------------------------------------------ resultados ----
    resultados = {
        "subtarea": "T3",
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "division_usada": {
            "origen": origen,
            "test_size": 0.3,
            "estratificada": True,
            "random_state": SEMILLA,
            "n_train": n_train_total,
            "n_test": n_test,
        },
        "protocolo": {
            "fracciones_evaluadas": FRACCIONES,
            "submuestreo": ("train_test_split(X_train, y_train, train_size=fraccion, "
                            "stratify=y_train, random_state=42); con 100 % se usa el "
                            "conjunto de entrenamiento completo"),
            "modelos": {
                "naive_bayes_gaussiano": "GaussianNB sobre los atributos crudos",
                "regresion_logistica": ("Pipeline(StandardScaler -> LogisticRegression(max_iter=1000)); "
                                        "el escalador se ajusta únicamente con cada submuestra "
                                        "de entrenamiento"),
            },
            "evaluacion": (f"accuracy (y F1 macro) sobre el mismo conjunto de prueba "
                           f"({n_test} ejemplos) para todos los tamaños de entrenamiento"),
        },
        "tamanos_train": tamanos,
        "tabla_accuracy": filas,
        "curvas": curvas,
        "conteo_clases_submuestras": conteos_sub,
        "mejor_modelo_por_tamano": mejor_por_tamano,
        "cruce_curvas": cruce,
        "archivos_generados": {
            "figura_curvas_aprendizaje": FIG_CURVAS,
            "figura_tabla_accuracy": FIG_TABLA,
        },
    }
    with open("resultados.json", "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, ensure_ascii=False, indent=2)

    # --------------------------------------------------------- resumen ----
    print("=" * 72)
    print("T3 — Curva de aprendizaje (generativo vs. discriminativo)")
    print(f"División usada: {origen}")
    print(f"n_train={n_train_total} | n_test={n_test} (el mismo test para todos los tamaños)")
    print("-" * 72)
    df_print = pd.DataFrame(filas)
    df_print["fraccion"] = (df_print["fraccion"] * 100).map(lambda v: f"{v:.0f}%")
    for c in df_print.columns:
        if c not in ("fraccion", "n_train"):
            df_print[c] = df_print[c].map(lambda v: f"{v:.4f}")
    print(df_print.to_string(index=False))
    print("-" * 72)
    print("Mejor modelo por tamaño de entrenamiento:")
    for m in mejor_por_tamano:
        print(f"  n_train={m['n_train']:>3}: {m['mejor_modelo']} (accuracy={m['accuracy']:.4f})")
    print("Cruce de curvas:", cruce)
    print(f"Figura de curvas guardada en: {FIG_CURVAS}")
    print(f"Figura de tabla guardada en: {FIG_TABLA}")
    print("Resultados escritos en: resultados.json")


if __name__ == "__main__":
    main()
