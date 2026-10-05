#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 — Curva de aprendizaje: Naive Bayes gaussiano vs. regresión logística.

Sobre la única división 70/30 estratificada de T1 (random_state=42), entrena los
dos modelos de T2 con submuestras estratificadas del 5 %, 10 %, 25 %, 50 % y 100 %
del conjunto de entrenamiento (random_state=42), evalúa cada tamaño siempre sobre
el mismo conjunto de prueba, grafica exactitud vs. número de ejemplos de
entrenamiento (una curva por modelo) y guarda la figura como PNG.
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
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42
FRACCIONES = [0.05, 0.10, 0.25, 0.50, 1.00]
NOMBRE_CLASE = {0: "malignant", 1: "benign"}
FIG_PNG = "curva_aprendizaje_T3.png"
TABLA_CSV = "tabla_curva_aprendizaje_T3.csv"


def cargar_json(ruta):
    p = Path(ruta)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def nuevos_modelos():
    """Los dos modelos de T2, con los mismos hiperparámetros."""
    return {
        "naive_bayes_gaussiano": GaussianNB(),  # var_smoothing=1e-9 (defecto), como en T2
        "regresion_logistica": LogisticRegression(
            solver="lbfgs", C=1.0, max_iter=5000, random_state=RANDOM_STATE
        ),
    }


def conteo_clases(y):
    vals, cnts = np.unique(y, return_counts=True)
    return {NOMBRE_CLASE[int(v)]: int(c) for v, c in zip(vals, cnts)}


def main():
    # ------------------------------------------------------------------ #
    # 1) Datos y única división de T1 (70/30 estratificada, random_state=42)
    # ------------------------------------------------------------------ #
    data = load_breast_cancer()
    X, y = data.data, data.target
    n_total = X.shape[0]

    # Regeneración determinista de la división de T1
    idx_tr_reg, idx_te_reg = train_test_split(
        np.arange(n_total), test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )

    # La división autoritativa es la de T1; se usa si está disponible y es válida
    t1 = cargar_json(Path("entrada/T1/resultados.json"))
    idx_tr_t1 = t1.get("indices_train") if isinstance(t1, dict) else None
    idx_te_t1 = t1.get("indices_test") if isinstance(t1, dict) else None

    usa_indices_t1 = (
        idx_tr_t1 is not None
        and idx_te_t1 is not None
        and len(idx_tr_t1) == len(idx_tr_reg)
        and len(idx_te_t1) == len(idx_te_reg)
    )
    if usa_indices_t1:
        idx_tr = np.asarray(idx_tr_t1, dtype=int)
        idx_te = np.asarray(idx_te_t1, dtype=int)
    else:
        idx_tr, idx_te = idx_tr_reg, idx_te_reg

    coincide_con_regenerada = bool(
        np.array_equal(np.sort(idx_tr), np.sort(idx_tr_reg))
        and np.array_equal(np.sort(idx_te), np.sort(idx_te_reg))
    )

    X_train, y_train = X[idx_tr], y[idx_tr]
    X_test, y_test = X[idx_te], y[idx_te]
    n_train, n_test = int(len(y_train)), int(len(y_test))

    # ------------------------------------------------------------------ #
    # 2) Bucle sobre fracciones del entrenamiento (submuestreo estratificado)
    # ------------------------------------------------------------------ #
    filas = []           # tabla por tamaño: fracción, n, accuracy y F1 por modelo
    submuestras = []     # detalle de cada submuestra estratificada
    curva_acc = {m: [] for m in nuevos_modelos()}
    curva_f1 = {m: [] for m in nuevos_modelos()}
    n_por_fraccion = []

    for frac in FRACCIONES:
        if frac >= 1.0:
            sub_local = np.arange(n_train)  # 100 %: entrenamiento completo
        else:
            sub_local, _ = train_test_split(
                np.arange(n_train),
                train_size=frac,
                stratify=y_train,
                random_state=RANDOM_STATE,
            )

        X_sub, y_sub = X_train[sub_local], y_train[sub_local]
        n_sub = int(len(y_sub))
        n_por_fraccion.append(n_sub)

        # Escalador ajustado SOLO con la submuestra de entrenamiento de esta corrida;
        # el conjunto de prueba jamás se usa para ajustar nada.
        scaler = StandardScaler().fit(X_sub)
        X_sub_z = scaler.transform(X_sub)
        X_test_z = scaler.transform(X_test)

        submuestras.append(
            {
                "fraccion": frac,
                "n_train_usado": n_sub,
                "conteo_por_clase_submuestra": conteo_clases(y_sub),
                "indices_locales_en_train": [int(i) for i in np.sort(sub_local)],
            }
        )

        fila = {"fraccion": frac, "n_train_usado": n_sub}
        for nombre, modelo in nuevos_modelos().items():
            modelo.fit(X_sub_z, y_sub)
            pred = modelo.predict(X_test_z)
            acc = float(accuracy_score(y_test, pred))
            f1m = float(f1_score(y_test, pred, average="macro"))
            curva_acc[nombre].append(acc)
            curva_f1[nombre].append(f1m)
            fila[f"accuracy_prueba_{nombre}"] = acc
            fila[f"f1_macro_prueba_{nombre}"] = f1m
        filas.append(fila)

    # ------------------------------------------------------------------ #
    # 3) Verificación: la corrida al 100 % debe reproducir los resultados de T2
    # ------------------------------------------------------------------ #
    t2 = cargar_json(Path("entrada/T2/resultados.json"))
    verif_t2 = {"T2_disponible": isinstance(t2, dict)}
    if isinstance(t2, dict):
        try:
            acc_gnb_t2 = float(t2["metricas_prueba"]["naive_bayes_gaussiano"]["accuracy_prueba"])
            acc_lr_t2 = float(t2["metricas_prueba"]["regresion_logistica"]["accuracy_prueba"])
            verif_t2.update(
                {
                    "accuracy_gnb_en_T2": acc_gnb_t2,
                    "accuracy_lr_en_T2": acc_lr_t2,
                    "accuracy_gnb_100pct_T3": curva_acc["naive_bayes_gaussiano"][-1],
                    "accuracy_lr_100pct_T3": curva_acc["regresion_logistica"][-1],
                    "coincide_gnb": bool(np.isclose(curva_acc["naive_bayes_gaussiano"][-1], acc_gnb_t2)),
                    "coincide_lr": bool(np.isclose(curva_acc["regresion_logistica"][-1], acc_lr_t2)),
                }
            )
        except (KeyError, TypeError, ValueError):
            verif_t2["T2_disponible"] = False

    # ------------------------------------------------------------------ #
    # 4) Figura: curva de aprendizaje (una curva por modelo)
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(
        n_por_fraccion, curva_acc["naive_bayes_gaussiano"], "o-",
        color="#1f77b4", lw=2, ms=6, label="Naive Bayes gaussiano (generativo)",
    )
    ax.plot(
        n_por_fraccion, curva_acc["regresion_logistica"], "s-",
        color="#d62728", lw=2, ms=6, label="Regresión logística (discriminativo)",
    )
    for x, v in zip(n_por_fraccion, curva_acc["naive_bayes_gaussiano"]):
        ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=8, color="#1f77b4")
    for x, v in zip(n_por_fraccion, curva_acc["regresion_logistica"]):
        ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#d62728")
    ax.set_xticks(n_por_fraccion)
    ax.set_xlabel("Número de ejemplos de entrenamiento")
    ax.set_ylabel("Exactitud en prueba (accuracy)")
    ax.set_title(
        "Curva de aprendizaje — Breast Cancer (división 70/30, random_state=42)\n"
        f"Evaluación siempre sobre el mismo conjunto de prueba (n={n_test})"
    )
    y_min = min(min(curva_acc["naive_bayes_gaussiano"]), min(curva_acc["regresion_logistica"]))
    ax.set_ylim(max(0.0, y_min - 0.09), 1.03)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig(FIG_PNG, dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 5) Tabla CSV de apoyo para el reporte
    # ------------------------------------------------------------------ #
    pd.DataFrame(filas).to_csv(TABLA_CSV, index=False, encoding="utf-8")

    # ------------------------------------------------------------------ #
    # 6) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------ #
    resultados = {
        "subtarea": "T3",
        "descripcion": (
            "Curva de aprendizaje: Naive Bayes gaussiano y regresión logística (los dos modelos "
            "de T2) entrenados con submuestras estratificadas del 5 %, 10 %, 25 %, 50 % y 100 % "
            "del entrenamiento de la división de T1 (random_state=42) y evaluados siempre sobre "
            "el mismo conjunto de prueba."
        ),
        "division_usada": {
            "fuente": (
                "entrada/T1/resultados.json" if usa_indices_t1
                else "regenerada (test_size=0.3, stratify=y, random_state=42)"
            ),
            "random_state": RANDOM_STATE,
            "test_size": 0.3,
            "n_train": n_train,
            "n_test": n_test,
            "conteo_por_clase_train": conteo_clases(y_train),
            "conteo_por_clase_test": conteo_clases(y_test),
            "coincide_con_division_regenerada": coincide_con_regenerada,
        },
        "protocolo": {
            "fracciones_entrenamiento": FRACCIONES,
            "random_state_submuestreo": RANDOM_STATE,
            "submuestreo": "train_test_split estratificado por clase sobre el entrenamiento de T1",
            "escalador": (
                "StandardScaler ajustado solo con cada submuestra de entrenamiento (nunca con "
                "prueba); al 100 % coincide con el escalador de T2"
            ),
            "conjunto_de_evaluacion": (
                f"mismo conjunto de prueba (n={n_test}) para todos los tamaños y ambos modelos"
            ),
            "hiperparametros": {
                "GaussianNB": "valores por defecto (var_smoothing=1e-9), igual que en T2",
                "LogisticRegression": "solver=lbfgs, C=1.0, max_iter=5000, random_state=42, igual que en T2",
            },
        },
        "curva_aprendizaje": {
            "n_ejemplos_entrenamiento": n_por_fraccion,
            "accuracy_prueba": {
                "naive_bayes_gaussiano": curva_acc["naive_bayes_gaussiano"],
                "regresion_logistica": curva_acc["regresion_logistica"],
            },
            "f1_macro_prueba": {
                "naive_bayes_gaussiano": curva_f1["naive_bayes_gaussiano"],
                "regresion_logistica": curva_f1["regresion_logistica"],
            },
        },
        "tabla_por_tamano": filas,
        "submuestras_estratificadas": submuestras,
        "verificacion_vs_T2_al_100pct": verif_t2,
        "archivos_generados": [FIG_PNG, TABLA_CSV],
    }
    Path("resultados.json").write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------------------ #
    # 7) Resumen
    # ------------------------------------------------------------------ #
    print("=" * 72)
    print("T3 — Curva de aprendizaje: GNB vs. regresión logística")
    print("=" * 72)
    print(f"División de T1: {n_train} entrenamiento / {n_test} prueba (random_state={RANDOM_STATE})")
    print(f"{'fracción':>9} | {'n_train':>7} | {'acc GNB':>8} | {'acc LR':>8}")
    print("-" * 44)
    for f in filas:
        print(
            f"{f['fraccion']:>8.0%} | {f['n_train_usado']:>7d} | "
            f"{f['accuracy_prueba_naive_bayes_gaussiano']:>8.4f} | "
            f"{f['accuracy_prueba_regresion_logistica']:>8.4f}"
        )
    if verif_t2.get("T2_disponible"):
        print(
            f"Verificación 100 % vs T2 -> GNB coincide: {verif_t2['coincide_gnb']} | "
            f"LR coincide: {verif_t2['coincide_lr']}"
        )
    print(f"Figura PNG: {FIG_PNG}")
    print(f"Tabla CSV:  {TABLA_CSV}")
    print("Cifras completas escritas en resultados.json")


if __name__ == "__main__":
    main()
