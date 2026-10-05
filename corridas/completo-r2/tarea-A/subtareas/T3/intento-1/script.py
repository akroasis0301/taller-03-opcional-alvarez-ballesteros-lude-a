# -*- coding: utf-8 -*-
"""
Subtarea T3 — Curva de aprendizaje: generativo (GaussianNB) vs. discriminativo
(regresión logística) con pocos datos.

Sobre la MISMA división 70/30 estratificada de T1 (random_state=42), se entrenan
los dos modelos de T2 con submuestras estratificadas del 5 %, 10 %, 25 %, 50 % y
100 % del conjunto de entrenamiento (random_state=42). Cada tamaño se evalúa
SIEMPRE sobre el mismo conjunto de prueba (30 %), que no se usa para ajustar
nada (el escalador tampoco: se ajusta solo con datos de entrenamiento).

Salidas:
  - curva_aprendizaje.png : exactitud vs. número de ejemplos de entrenamiento,
                            una curva por modelo.
  - resultados.json       : todas las cifras que pide la subtarea.
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

SEMILLA = 42
FRACCIONES = [0.05, 0.10, 0.25, 0.50, 1.00]
PORCENTAJES = [5, 10, 25, 50, 100]
FIGURA_PNG = "curva_aprendizaje.png"
RUTA_T1 = os.path.join("entrada", "T1", "resultados.json")


def main():
    # ------------------------------------------------------------------
    # 1) Misma división única de T1: 70/30 estratificada, random_state=42
    #    (se recrea de forma determinista y se verifica contra T1).
    # ------------------------------------------------------------------
    datos = load_breast_cancer()
    X, y = datos.data, datos.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEMILLA, stratify=y
    )
    n_train, n_test = int(X_train.shape[0]), int(X_test.shape[0])

    verificacion = {
        "fuente_division": (
            "recreada de forma determinista con train_test_split(test_size=0.3, "
            "random_state=42, stratify=y): los mismos parámetros con los que T1 "
            "creó la división única"
        ),
        "archivo_T1_leido": False,
    }
    if os.path.exists(RUTA_T1):
        try:
            with open(RUTA_T1, "r", encoding="utf-8") as f:
                t1 = json.load(f)
            verificacion["archivo_T1_leido"] = True
            verificacion["tamano_entrenamiento_coincide"] = (
                int(t1.get("tamano_entrenamiento", -1)) == n_train
            )
            verificacion["tamano_prueba_coincide"] = (
                int(t1.get("tamano_prueba", -1)) == n_test
            )
            c_ent = t1.get("conteo_clases_entrenamiento", {})
            c_pru = t1.get("conteo_clases_prueba", {})
            verificacion["conteo_clases_entrenamiento_coincide"] = (
                int(c_ent.get("malignant", -1)) == int(np.sum(y_train == 0))
                and int(c_ent.get("benign", -1)) == int(np.sum(y_train == 1))
            )
            verificacion["conteo_clases_prueba_coincide"] = (
                int(c_pru.get("malignant", -1)) == int(np.sum(y_test == 0))
                and int(c_pru.get("benign", -1)) == int(np.sum(y_test == 1))
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # ------------------------------------------------------------------
    # 2) Curva de aprendizaje: submuestras estratificadas del entrenamiento
    # ------------------------------------------------------------------
    curvas = {
        "gaussian_nb": {"n_ejemplos": [], "accuracy_prueba": [], "f1_macro_prueba": []},
        "regresion_logistica": {"n_ejemplos": [], "accuracy_prueba": [], "f1_macro_prueba": []},
    }
    composicion = {}
    filas_tabla = []

    for frac, pct in zip(FRACCIONES, PORCENTAJES):
        if frac >= 1.0:
            Xs, ys = X_train, y_train
        else:
            Xs, _, ys, _ = train_test_split(
                X_train,
                y_train,
                train_size=frac,
                random_state=SEMILLA,
                stratify=y_train,
            )
        n_sub = int(Xs.shape[0])
        composicion[f"{pct}%"] = {
            "n_ejemplos": n_sub,
            "malignant": int(np.sum(ys == 0)),
            "benign": int(np.sum(ys == 1)),
        }

        # Escalador ajustado SOLO con la submuestra de entrenamiento de esta
        # corrida (nunca con el conjunto de prueba).
        escalador = StandardScaler().fit(Xs)
        Xs_esc = escalador.transform(Xs)
        X_test_esc = escalador.transform(X_test)  # solo transformar, nunca fit

        # --- Modelo generativo: GaussianNB ---
        nb = GaussianNB()
        nb.fit(Xs_esc, ys)
        pred_nb = nb.predict(X_test_esc)
        acc_nb = float(accuracy_score(y_test, pred_nb))
        f1_nb = float(f1_score(y_test, pred_nb, average="macro"))

        # --- Modelo discriminativo: regresión logística ---
        rl = LogisticRegression(max_iter=5000, random_state=SEMILLA)
        rl.fit(Xs_esc, ys)
        pred_rl = rl.predict(X_test_esc)
        acc_rl = float(accuracy_score(y_test, pred_rl))
        f1_rl = float(f1_score(y_test, pred_rl, average="macro"))

        curvas["gaussian_nb"]["n_ejemplos"].append(n_sub)
        curvas["gaussian_nb"]["accuracy_prueba"].append(acc_nb)
        curvas["gaussian_nb"]["f1_macro_prueba"].append(f1_nb)
        curvas["regresion_logistica"]["n_ejemplos"].append(n_sub)
        curvas["regresion_logistica"]["accuracy_prueba"].append(acc_rl)
        curvas["regresion_logistica"]["f1_macro_prueba"].append(f1_rl)

        filas_tabla.append(
            {
                "porcentaje_entrenamiento": pct,
                "n_ejemplos_entrenamiento": n_sub,
                "accuracy_prueba_gaussian_nb": acc_nb,
                "accuracy_prueba_regresion_logistica": acc_rl,
                "f1_macro_prueba_gaussian_nb": f1_nb,
                "f1_macro_prueba_regresion_logistica": f1_rl,
            }
        )

    n_vals = curvas["gaussian_nb"]["n_ejemplos"]
    acc_nb_vals = curvas["gaussian_nb"]["accuracy_prueba"]
    acc_rl_vals = curvas["regresion_logistica"]["accuracy_prueba"]

    # ------------------------------------------------------------------
    # 3) Figura: curva de aprendizaje (una curva por modelo)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(
        n_vals, acc_nb_vals, marker="o", color="tab:blue", linewidth=2,
        label="Naive Bayes gaussiano (generativo)",
    )
    ax.plot(
        n_vals, acc_rl_vals, marker="s", color="tab:red", linewidth=2,
        label="Regresión logística (discriminativo)",
    )
    for xv, yv in zip(n_vals, acc_nb_vals):
        ax.annotate(
            f"{yv:.3f}", (xv, yv), textcoords="offset points", xytext=(0, -16),
            ha="center", fontsize=8, color="tab:blue",
        )
    for xv, yv in zip(n_vals, acc_rl_vals):
        dy = 9 if yv <= 0.97 else -16  # evita salirse del eje si está muy arriba
        ax.annotate(
            f"{yv:.3f}", (xv, yv), textcoords="offset points", xytext=(0, dy),
            ha="center", fontsize=8, color="tab:red",
        )
    ax.set_xticks(n_vals)
    ax.set_xticklabels([f"{n}\n({p} %)" for n, p in zip(n_vals, PORCENTAJES)])
    ax.set_xlabel("Número de ejemplos de entrenamiento (porcentaje del entrenamiento de T1)")
    ax.set_ylabel("Exactitud (accuracy) en el conjunto de prueba")
    ax.set_title(
        "Curva de aprendizaje: generativo vs. discriminativo\n"
        f"Breast Cancer Wisconsin — división 70/30 estratificada de T1, "
        f"prueba fija de {n_test} ejemplos"
    )
    y_min = min(min(acc_nb_vals), min(acc_rl_vals))
    ax.set_ylim(max(0.0, y_min - 0.08), 1.0)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig(FIGURA_PNG, dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    tabla_md = (
        "| % del entrenamiento | n ejemplos | Accuracy prueba (GaussianNB) | "
        "Accuracy prueba (Reg. logística) |\n"
        "|---|---|---|---|\n"
        + "\n".join(
            f"| {r['porcentaje_entrenamiento']} % | {r['n_ejemplos_entrenamiento']} | "
            f"{r['accuracy_prueba_gaussian_nb']:.6f} | "
            f"{r['accuracy_prueba_regresion_logistica']:.6f} |"
            for r in filas_tabla
        )
    )

    resultados = {
        "subtarea": "T3",
        "descripcion": (
            "Curva de aprendizaje sobre la división única de T1: GaussianNB y "
            "regresión logística (atributos estandarizados) entrenados con "
            "submuestras estratificadas del 5 %, 10 %, 25 %, 50 % y 100 % del "
            "entrenamiento (random_state=42) y evaluados siempre sobre el mismo "
            "conjunto de prueba."
        ),
        "dataset": "breast_cancer_wisconsin_diagnostic (sklearn.datasets.load_breast_cancer)",
        "division_usada": {
            "fuente": verificacion["fuente_division"],
            "test_size": 0.3,
            "random_state": SEMILLA,
            "estratificada_por_clase": True,
            "tamano_entrenamiento": n_train,
            "tamano_prueba": n_test,
        },
        "verificacion_contra_T1": verificacion,
        "conjunto_prueba": {
            "fijo_para_todos_los_tamanos": True,
            "n_ejemplos": n_test,
            "usado_para_ajustar": False,
        },
        "submuestras_estratificadas": {
            "metodo": (
                "train_test_split(train_size=fraccion, random_state=42, "
                "stratify=y_train) sobre el entrenamiento de T1; para el 100 % "
                "se usa el entrenamiento completo"
            ),
            "random_state": SEMILLA,
            "fracciones": FRACCIONES,
            "porcentajes": PORCENTAJES,
            "composicion_por_porcentaje": composicion,
        },
        "preprocesamiento": {
            "escalador": "StandardScaler",
            "ajustado_solo_con_datos_de_entrenamiento": True,
            "nota": (
                "Para cada tamaño, el escalador se ajusta únicamente con la "
                "submuestra de entrenamiento de esa corrida (nunca con el "
                "conjunto de prueba); con la submuestra del 100 % coincide con "
                "el ajuste de T2. Ambos modelos usan los atributos "
                "estandarizados por consistencia con T2 (GaussianNB es "
                "invariante a transformaciones afines por atributo)."
            ),
        },
        "modelos": {
            "gaussian_nb": {
                "clase": "sklearn.naive_bayes.GaussianNB",
                "parametros": "por defecto",
            },
            "regresion_logistica": {
                "clase": "sklearn.linear_model.LogisticRegression",
                "parametros": {
                    "max_iter": 5000,
                    "random_state": SEMILLA,
                    "resto": "por defecto",
                },
            },
        },
        "n_ejemplos_entrenamiento": n_vals,
        "curva_aprendizaje": {
            "eje_x": "número de ejemplos de entrenamiento",
            "eje_y": "accuracy en el conjunto de prueba (fijo)",
            "n_ejemplos_entrenamiento": n_vals,
            "gaussian_nb": curvas["gaussian_nb"],
            "regresion_logistica": curvas["regresion_logistica"],
        },
        "accuracy_prueba_gaussian_nb_por_n": {
            str(n): a for n, a in zip(n_vals, acc_nb_vals)
        },
        "accuracy_prueba_regresion_logistica_por_n": {
            str(n): a for n, a in zip(n_vals, acc_rl_vals)
        },
        "f1_macro_prueba_gaussian_nb_por_n": {
            str(n): a for n, a in zip(n_vals, curvas["gaussian_nb"]["f1_macro_prueba"])
        },
        "f1_macro_prueba_regresion_logistica_por_n": {
            str(n): a
            for n, a in zip(n_vals, curvas["regresion_logistica"]["f1_macro_prueba"])
        },
        "tabla_por_punto": filas_tabla,
        "tabla_markdown": tabla_md,
        "figura_png": FIGURA_PNG,
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 5) Resumen breve
    # ------------------------------------------------------------------
    print("=" * 74)
    print(f"T3 — Curva de aprendizaje (prueba fija de {n_test} ejemplos)")
    print("=" * 74)
    print(f"{'% entrenamiento':>16} {'n':>5} {'Acc GaussianNB':>16} {'Acc Reg.Log.':>16}")
    for r in filas_tabla:
        print(
            f"{r['porcentaje_entrenamiento']:>15}% {r['n_ejemplos_entrenamiento']:>5} "
            f"{r['accuracy_prueba_gaussian_nb']:>16.6f} "
            f"{r['accuracy_prueba_regresion_logistica']:>16.6f}"
        )
    print(f"\nVerificación contra T1: {verificacion}")
    print(f"Figura guardada en: {FIGURA_PNG}")
    print("Cifras completas escritas en: resultados.json")


if __name__ == "__main__":
    main()
