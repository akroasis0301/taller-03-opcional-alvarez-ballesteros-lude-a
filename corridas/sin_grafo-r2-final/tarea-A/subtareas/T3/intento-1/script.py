# -*- coding: utf-8 -*-
"""
Subtarea T3 — Curva de aprendizaje (Breast Cancer Wisconsin (Diagnostic)).

Sobre la MISMA división 70/30 estratificada de la Parte 1 (random_state=42) se
entrenan los dos modelos de la Parte 2 (Naive Bayes gaussiano y regresión
logística con atributos estandarizados) con el 5 %, 10 %, 25 %, 50 % y 100 %
del conjunto de entrenamiento (submuestras estratificadas, random_state=42).
Cada modelo se evalúa SIEMPRE sobre el mismo conjunto de prueba.

Salidas:
  * resultados.json           -> tabla de exactitudes por modelo y tamaño
  * curva_aprendizaje_T3.png  -> figura con las dos curvas de aprendizaje
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
FRACCIONES = [0.05, 0.10, 0.25, 0.50, 1.00]
ETIQUETAS = ["5%", "10%", "25%", "50%", "100%"]
FIGURA_PNG = "curva_aprendizaje_T3.png"


def main():
    # ------------------------------------------------------------------
    # 1) Datos y MISMA división de la Parte 1 (70/30 estratificada, rs=42)
    # ------------------------------------------------------------------
    datos = load_breast_cancer()
    X, y = datos.data, datos.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    n_train_total, n_test = int(X_train.shape[0]), int(X_test.shape[0])

    # Verificación (informativa) de que la división coincide con la de T1
    coincide_t1 = None
    try:
        with open(Path("entrada") / "T1" / "resultados.json", encoding="utf-8") as f:
            div = json.load(f).get("division", {})
        coincide_t1 = (div.get("n_train") == n_train_total
                       and div.get("n_test") == n_test)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 2) Entrenamiento con 5/10/25/50/100 % del entrenamiento
    # ------------------------------------------------------------------
    filas = []
    curvas = {
        "naive_bayes_gaussiano": {"n_ejemplos": [], "accuracy_test": [], "f1_macro_test": []},
        "regresion_logistica": {"n_ejemplos": [], "accuracy_test": [], "f1_macro_test": []},
    }

    for frac, etiqueta in zip(FRACCIONES, ETIQUETAS):
        # Submuestra ESTRATIFICADA del entrenamiento (random_state=42);
        # con el 100 % se usa el entrenamiento completo.
        if frac >= 1.0:
            X_tr, y_tr = X_train, y_train
        else:
            X_tr, _, y_tr, _ = train_test_split(
                X_train, y_train, train_size=frac,
                stratify=y_train, random_state=RANDOM_STATE,
            )
        n_tr = int(X_tr.shape[0])

        # --- Modelo 1: Naive Bayes gaussiano (atributos originales) ---
        nb = GaussianNB().fit(X_tr, y_tr)
        pred_nb = nb.predict(X_test)
        acc_nb = float(accuracy_score(y_test, pred_nb))
        f1_nb = float(f1_score(y_test, pred_nb, average="macro"))

        # --- Modelo 2: Regresión logística ---
        # El escalador se ajusta SOLO con los datos de entrenamiento usados
        # en este punto (la submuestra); el conjunto de prueba nunca participa.
        escalador = StandardScaler().fit(X_tr)
        lr = LogisticRegression(max_iter=1000)
        lr.fit(escalador.transform(X_tr), y_tr)
        pred_lr = lr.predict(escalador.transform(X_test))
        acc_lr = float(accuracy_score(y_test, pred_lr))
        f1_lr = float(f1_score(y_test, pred_lr, average="macro"))

        filas.append({
            "fraccion_entrenamiento": etiqueta,
            "n_ejemplos_entrenamiento": n_tr,
            "accuracy_naive_bayes_gaussiano": acc_nb,
            "accuracy_regresion_logistica": acc_lr,
            "f1_macro_naive_bayes_gaussiano": f1_nb,
            "f1_macro_regresion_logistica": f1_lr,
        })
        for modelo, acc, f1m in (("naive_bayes_gaussiano", acc_nb, f1_nb),
                                 ("regresion_logistica", acc_lr, f1_lr)):
            curvas[modelo]["n_ejemplos"].append(n_tr)
            curvas[modelo]["accuracy_test"].append(acc)
            curvas[modelo]["f1_macro_test"].append(f1m)

    # Verificación (informativa) del punto 100 % contra los resultados de T2
    coincide_t2 = None
    try:
        with open(Path("entrada") / "T2" / "resultados.json", encoding="utf-8") as f:
            t2 = json.load(f)["accuracy_f1_macro_prueba"]
        coincide_t2 = {
            "naive_bayes_gaussiano": bool(
                abs(curvas["naive_bayes_gaussiano"]["accuracy_test"][-1]
                    - t2["naive_bayes_gaussiano"]["accuracy"]) < 1e-12),
            "regresion_logistica": bool(
                abs(curvas["regresion_logistica"]["accuracy_test"][-1]
                    - t2["regresion_logistica"]["accuracy"]) < 1e-12),
        }
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 3) Figura: exactitud frente al número de ejemplos de entrenamiento
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    estilos = {
        "naive_bayes_gaussiano": ("o", "tab:blue", "Naive Bayes gaussiano", 8),
        "regresion_logistica": ("s", "tab:red",
                                "Regresión logística (atributos estandarizados)", -12),
    }
    for modelo, (marcador, color, etiqueta, despl) in estilos.items():
        ax.plot(curvas[modelo]["n_ejemplos"], curvas[modelo]["accuracy_test"],
                marker=marcador, color=color, label=etiqueta)
        for n_x, acc in zip(curvas[modelo]["n_ejemplos"],
                            curvas[modelo]["accuracy_test"]):
            ax.annotate(f"{acc:.3f}", (n_x, acc), textcoords="offset points",
                        xytext=(0, despl), ha="center", fontsize=8, color=color)

    todo_acc = (curvas["naive_bayes_gaussiano"]["accuracy_test"]
                + curvas["regresion_logistica"]["accuracy_test"])
    ax.set_ylim(max(0.0, min(todo_acc) - 0.07), 1.02)
    ax.set_xticks(curvas["naive_bayes_gaussiano"]["n_ejemplos"])
    ax.set_xlabel("Número de ejemplos de entrenamiento")
    ax.set_ylabel("Exactitud (accuracy) en el conjunto de prueba")
    ax.set_title("Curva de aprendizaje — Breast Cancer Wisconsin (Diagnostic)\n"
                 f"Evaluación fija sobre el mismo conjunto de prueba (n = {n_test})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURA_PNG, dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    tamanos = [f_["n_ejemplos_entrenamiento"] for f_ in filas]
    resultados = {
        "subtarea": "T3",
        "descripcion": (
            "Curva de aprendizaje sobre la división 70/30 estratificada de la Parte 1: "
            "Naive Bayes gaussiano y regresión logística (atributos estandarizados, "
            "escalador ajustado solo con el entrenamiento) entrenados con el 5 %, 10 %, "
            "25 %, 50 % y 100 % del entrenamiento (submuestras estratificadas, "
            "random_state=42) y evaluados siempre sobre el mismo conjunto de prueba."
        ),
        "division_usada": {
            "test_size": 0.3,
            "estratificada": True,
            "random_state": RANDOM_STATE,
            "n_train": n_train_total,
            "n_test": n_test,
            "coincide_con_T1": coincide_t1,
        },
        "submuestreo_entrenamiento": {
            "estratificado": True,
            "random_state": RANDOM_STATE,
            "fracciones": FRACCIONES,
            "n_ejemplos_por_fraccion": dict(zip(ETIQUETAS, tamanos)),
        },
        "preprocesado": {
            "naive_bayes_gaussiano": "atributos originales",
            "regresion_logistica": (
                "StandardScaler ajustado únicamente con la submuestra de "
                "entrenamiento correspondiente a cada tamaño"
            ),
        },
        "tabla_exactitud_prueba": filas,
        "curva_aprendizaje": curvas,
        "verificacion_punto_100pct_vs_T2": coincide_t2,
        "figura_png": FIGURA_PNG,
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 5) Resumen
    # ------------------------------------------------------------------
    print("Subtarea T3 — Curva de aprendizaje")
    print(f"División 70/30 estratificada (random_state={RANDOM_STATE}): "
          f"n_train={n_train_total}, n_test={n_test} (coincide con T1: {coincide_t1})")
    enc = f"{'% entren':>9} {'n_train':>8} {'acc NB':>9} {'acc LR':>9}"
    print(enc)
    print("-" * len(enc))
    for f_ in filas:
        print(f"{f_['fraccion_entrenamiento']:>9} "
              f"{f_['n_ejemplos_entrenamiento']:>8} "
              f"{f_['accuracy_naive_bayes_gaussiano']:>9.4f} "
              f"{f_['accuracy_regresion_logistica']:>9.4f}")
    print(f"Verificación del punto 100 % frente a T2: {coincide_t2}")
    print(f"Figura guardada en: {FIGURA_PNG}")
    print("Resultados escritos en: resultados.json")


if __name__ == "__main__":
    main()
