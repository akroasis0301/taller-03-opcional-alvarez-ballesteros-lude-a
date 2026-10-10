#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 — Parte 3: La curva de aprendizaje.

Sobre la MISMA división 70/30 estratificada (random_state=42) de la Parte 1,
entrena los dos modelos de la Parte 2 (Naive Bayes gaussiano = generativo y
regresión logística con atributos estandarizados = discriminativo) con el
5 %, 10 %, 25 %, 50 % y 100 % del conjunto de entrenamiento (submuestras
estratificadas, random_state=42) y evalúa cada uno SIEMPRE sobre el mismo
conjunto de prueba. Dibuja la exactitud contra el número de ejemplos de
entrenamiento, con una curva por modelo, y guarda la figura como PNG.

Salidas (carpeta actual):
  * resultados.json                -> tabla de accuracy en prueba
                                      (5 tamaños x 2 modelos = 10 cifras)
  * curvas_aprendizaje_parte3.png  -> figura con las dos curvas de aprendizaje
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
from sklearn.metrics import accuracy_score

SEMILLA = 42
PORCENTAJES = [0.05, 0.10, 0.25, 0.50, 1.00]
ETIQUETAS_PCT = ["5 %", "10 %", "25 %", "50 %", "100 %"]
FIGURA_PNG = "curvas_aprendizaje_parte3.png"
ARCHIVO_RESULTADOS = "resultados.json"


def cargar_division():
    """Devuelve (X_train, y_train, X_test, y_test, fuente).

    Intenta usar la división única 70/30 de la Parte 1 guardada en entrada/T1/;
    si no está disponible o no es consistente, la recrea de forma determinista
    con los parámetros exactos del enunciado (test_size=0.3, estratificada,
    random_state=42), lo que reproduce la misma partición.
    """
    base = Path("entrada/T1")
    nombres_target = {"target", "y", "label", "clase", "class", "diagnosis",
                      "diagnostico", "objetivo"}
    columnas_indice = {"__index_level_0__", "index", "unnamed: 0"}

    def limpiar(df):
        cols = [c for c in df.columns if str(c).strip().lower() in columnas_indice]
        return df.drop(columns=cols)

    for nombre in ("train.parquet", "train.csv"):
        f_train = base / nombre
        if not f_train.exists():
            continue
        f_test = base / ("test" + f_train.suffix)
        if not f_test.exists():
            continue
        try:
            if f_train.suffix == ".parquet":
                df_tr, df_te = pd.read_parquet(f_train), pd.read_parquet(f_test)
            else:
                df_tr, df_te = pd.read_csv(f_train), pd.read_csv(f_test)
        except Exception:
            continue

        df_tr, df_te = limpiar(df_tr), limpiar(df_te)
        col = next((c for c in df_tr.columns if str(c).strip().lower() in nombres_target), None)
        if col is None or col not in df_te.columns:
            continue
        try:
            Xtr = df_tr.drop(columns=[col]).to_numpy(dtype=float)
            Xte = df_te.drop(columns=[col]).to_numpy(dtype=float)
        except Exception:
            continue
        ytr = df_tr[col].to_numpy()
        yte = df_te[col].to_numpy()
        if Xtr.shape == (398, 30) and Xte.shape == (171, 30):
            return Xtr, ytr, Xte, yte, f"entrada/T1/{f_train.name} (división única de la Parte 1)"

    datos = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(datos.data, datos.target,
                                          test_size=0.3, stratify=datos.target,
                                          random_state=SEMILLA)
    return Xtr, ytr, Xte, yte, ("recreada de forma determinista: load_breast_cancer + "
                                "train_test_split(test_size=0.3, stratify=y, random_state=42)")


def main():
    Xtr, ytr, Xte, yte, fuente = cargar_division()
    n_train_total = int(Xtr.shape[0])
    n_test = int(Xte.shape[0])

    filas, tamanos, curva_nb, curva_lr, composicion = [], [], [], [], []

    for etiqueta, fraccion in zip(ETIQUETAS_PCT, PORCENTAJES):
        # Submuestra ESTRATIFICADA del entrenamiento (la parte descartada no se usa)
        if fraccion >= 1.0:
            Xs, ys = Xtr, ytr
        else:
            Xs, _, ys, _ = train_test_split(Xtr, ytr, train_size=fraccion,
                                            stratify=ytr, random_state=SEMILLA)
        n_sub = int(Xs.shape[0])

        # Generativo: Naive Bayes gaussiano (atributos crudos; invariante a
        # transformaciones afines por atributo, como en la Parte 2)
        nb = GaussianNB().fit(Xs, ys)
        acc_nb = float(accuracy_score(yte, nb.predict(Xte)))

        # Discriminativo: regresión logística; el escalador se ajusta SOLO con
        # la submuestra de entrenamiento (el conjunto de prueba jamás se usa
        # para ajustar nada)
        escalador = StandardScaler().fit(Xs)
        lr = LogisticRegression(max_iter=1000).fit(escalador.transform(Xs), ys)
        acc_lr = float(accuracy_score(yte, lr.predict(escalador.transform(Xte))))

        clases, conteos = np.unique(ys, return_counts=True)
        composicion.append({"porcentaje": etiqueta, "n_train": n_sub,
                            "n_por_clase": {str(c): int(k) for c, k in zip(clases, conteos)}})

        tamanos.append(n_sub)
        curva_nb.append(acc_nb)
        curva_lr.append(acc_lr)
        filas.append({"porcentaje_entrenamiento": etiqueta, "n_train": n_sub,
                      "accuracy_prueba_naive_bayes_gaussiano": acc_nb,
                      "accuracy_prueba_regresion_logistica": acc_lr})

    # Verificación opcional de consistencia con la Parte 2 en el punto 100 %
    comparacion_t2 = None
    ruta_t2 = Path("entrada/T2/resultados.json")
    if ruta_t2.exists():
        try:
            t2 = json.loads(ruta_t2.read_text(encoding="utf-8"))
            nb_t2 = float(t2["modelos"]["naive_bayes_gaussiano"]["accuracy_prueba"])
            lr_t2 = float(t2["modelos"]["regresion_logistica"]["accuracy_prueba"])
            comparacion_t2 = {
                "nota": "El punto 100 % de esta parte debe reproducir los resultados de la Parte 2.",
                "accuracy_100pct_naive_bayes_T3": curva_nb[-1],
                "accuracy_naive_bayes_T2": nb_t2,
                "coincide_naive_bayes": bool(abs(curva_nb[-1] - nb_t2) < 1e-12),
                "accuracy_100pct_regresion_logistica_T3": curva_lr[-1],
                "accuracy_regresion_logistica_T2": lr_t2,
                "coincide_regresion_logistica": bool(abs(curva_lr[-1] - lr_t2) < 1e-12),
            }
        except Exception:
            comparacion_t2 = None

    # ---------- Figura: curva de aprendizaje (una curva por modelo) ----------
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    ax.plot(tamanos, curva_nb, marker="o", linewidth=2, color="tab:blue",
            label="Naive Bayes gaussiano (generativo)")
    ax.plot(tamanos, curva_lr, marker="s", linewidth=2, color="tab:red",
            label="Regresión logística estandarizada (discriminativo)")
    for x, a, b in zip(tamanos, curva_nb, curva_lr):
        alto, bajo = (a, b) if a >= b else (b, a)
        ax.annotate(f"{alto:.3f}", (x, alto), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
        ax.annotate(f"{bajo:.3f}", (x, bajo), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8)
    ax.set_xticks(tamanos)
    ax.set_xticklabels([f"{n}\n({p})" for n, p in zip(tamanos, ETIQUETAS_PCT)])
    ax.set_xlim(min(tamanos) * 0.8, max(tamanos) * 1.12)
    ax.set_ylim(max(0.5, min(curva_nb + curva_lr) - 0.05), 1.005)
    ax.set_xlabel("Número de ejemplos de entrenamiento (submuestras estratificadas)")
    ax.set_ylabel(f"Accuracy en el conjunto de prueba (fijo, n = {n_test})")
    ax.set_title("Parte 3 — Curva de aprendizaje: generativo vs. discriminativo\n"
                 "División 70/30 estratificada (random_state=42); misma prueba para todos los puntos")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURA_PNG, dpi=120)
    plt.close(fig)

    # ---------- Tabla markdown ----------
    lineas = ["| % del entrenamiento | n_train | Accuracy prueba — NB gaussiano | Accuracy prueba — Reg. logística |",
              "|---|---|---|---|"]
    for f in filas:
        lineas.append(f"| {f['porcentaje_entrenamiento']} | {f['n_train']} | "
                      f"{f['accuracy_prueba_naive_bayes_gaussiano']:.4f} | "
                      f"{f['accuracy_prueba_regresion_logistica']:.4f} |")
    tabla_md = "\n".join(lineas)

    ganadores = []
    for n, a, b in zip(tamanos, curva_nb, curva_lr):
        if a > b:
            g = "naive_bayes_gaussiano"
        elif b > a:
            g = "regresion_logistica"
        else:
            g = "empate"
        ganadores.append({"n_train": n, "ganador": g,
                          "diferencia_accuracy_nb_menos_lr": float(a - b)})

    resultados = {
        "subtarea": "T3",
        "descripcion": ("Curva de aprendizaje sobre la división 70/30 estratificada de la Parte 1: "
                        "Naive Bayes gaussiano (generativo) y regresión logística con atributos "
                        "estandarizados (discriminativo) entrenados con el 5 %, 10 %, 25 %, 50 % y 100 % "
                        "del entrenamiento (submuestras estratificadas, random_state=42) y evaluados "
                        "siempre sobre el mismo conjunto de prueba."),
        "division_usada": {"fuente": fuente, "test_size": 0.3, "estratificada": True,
                           "random_state": SEMILLA, "n_train_total": n_train_total,
                           "n_test": n_test},
        "porcentajes_entrenamiento": ETIQUETAS_PCT,
        "tamanos_entrenamiento": tamanos,
        "tabla_accuracy_prueba": filas,
        "accuracy_por_modelo": {
            "naive_bayes_gaussiano": {str(n): a for n, a in zip(tamanos, curva_nb)},
            "regresion_logistica": {str(n): a for n, a in zip(tamanos, curva_lr)},
        },
        "n_cifras_accuracy": len(filas) * 2,
        "modelo_ganador_por_tamano": ganadores,
        "composicion_submuestras": composicion,
        "comparacion_con_T2_punto_100pct": comparacion_t2,
        "figura_png": FIGURA_PNG,
        "tabla_markdown": tabla_md,
        "notas": [
            "El conjunto de prueba (n=171) es el de la Parte 1 y no se usa para ajustar nada.",
            "El StandardScaler de la regresión logística se ajusta únicamente con cada submuestra de entrenamiento.",
            "Naive Bayes gaussiano se entrena con los atributos crudos (invariante a transformaciones afines por atributo).",
            ("Submuestras estratificadas: train_test_split(X_train, y_train, train_size=fraccion, "
             "stratify=y_train, random_state=42); el tamaño resultante trunca hacia abajo "
             "(5 % de 398 -> 19, 10 % -> 39, 25 % -> 99, 50 % -> 199, 100 % -> 398)."),
        ],
    }
    Path(ARCHIVO_RESULTADOS).write_text(json.dumps(resultados, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

    # ---------- Resumen ----------
    print("=" * 78)
    print("T3 — Parte 3: curva de aprendizaje (generativo vs. discriminativo)")
    print("=" * 78)
    print(f"División: {fuente}")
    print(f"Entrenamiento total: {n_train_total} | Prueba fija: {n_test} ejemplos\n")
    print(tabla_md)
    print()
    for g in ganadores:
        print(f"n_train={g['n_train']:>3}: mejor -> {g['ganador']} "
              f"(Δ accuracy NB-LR = {g['diferencia_accuracy_nb_menos_lr']:+.4f})")
    if comparacion_t2:
        print("\nConsistencia con la Parte 2 (punto 100 %): "
              f"NB coincide={comparacion_t2['coincide_naive_bayes']}, "
              f"RL coincide={comparacion_t2['coincide_regresion_logistica']}")
    print(f"\nFigura guardada: {FIGURA_PNG}")
    print(f"Resultados escritos: {ARCHIVO_RESULTADOS}")


if __name__ == "__main__":
    main()
