# -*- coding: utf-8 -*-
"""
T2 (Parte 2 de la Tarea A, MMIA 6013): generativo contra discriminativo.

Entrena sobre el conjunto de ENTRENAMIENTO de la división única de T1
(70/30 estratificada por clase, random_state=42):
  - GaussianNB (Naive Bayes gaussiano, generativo)
  - Regresión logística (discriminativo) con atributos estandarizados,
    usando un StandardScaler ajustado SOLO con el conjunto de entrenamiento.
Evalúa ambos modelos sobre el conjunto de PRUEBA y reporta accuracy y F1
macro en una tabla (resultados.json + figuras PNG).
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

RANDOM_STATE = 42
TEST_SIZE = 0.3
DIR_T1 = Path("entrada/T1")


def _primero_existente(base, nombres):
    for n in nombres:
        p = base / n
        if p.exists():
            return p
    return None


def intentar_cargar_split_T1(X, y):
    """Intenta recuperar la división exacta de T1 desde entrada/T1/.

    Opción A: arrays X_train/y_train/X_test/y_test guardados como .npy.
    Opción B: índices train/test guardados como .npy.
    Devuelve (dict_con_el_split, descripcion_fuente) o (None, None).
    """
    # Opción A: arrays completos
    claves = {
        "X_train": ["X_train.npy", "X_entrenamiento.npy"],
        "y_train": ["y_train.npy", "y_entrenamiento.npy"],
        "X_test": ["X_test.npy", "X_prueba.npy"],
        "y_test": ["y_test.npy", "y_prueba.npy"],
    }
    datos, completo = {}, True
    for clave, nombres in claves.items():
        p = _primero_existente(DIR_T1, nombres)
        if p is None:
            completo = False
            break
        try:
            datos[clave] = np.load(p)
        except Exception:
            return None, None
    if completo:
        try:
            ok = (
                datos["X_train"].ndim == 2
                and datos["X_test"].ndim == 2
                and datos["X_train"].shape[1] == datos["X_test"].shape[1]
                and datos["X_train"].shape[0] == datos["y_train"].shape[0]
                and datos["X_test"].shape[0] == datos["y_test"].shape[0]
                and set(np.unique(datos["y_train"]).tolist()).issubset({0, 1})
                and set(np.unique(datos["y_test"]).tolist()).issubset({0, 1})
            )
        except Exception:
            ok = False
        if ok:
            return datos, "arrays de la división guardados por T1 en entrada/T1/"

    # Opción B: índices de la división
    p_ient = _primero_existente(
        DIR_T1,
        ["train_indices.npy", "indices_entrenamiento.npy", "indices_train.npy", "train_idx.npy"],
    )
    p_ipru = _primero_existente(
        DIR_T1,
        ["test_indices.npy", "indices_prueba.npy", "indices_test.npy", "test_idx.npy"],
    )
    if p_ient is not None and p_ipru is not None:
        try:
            idx_tr = np.load(p_ient).ravel().astype(int)
            idx_te = np.load(p_ipru).ravel().astype(int)
            ok = (
                idx_tr.size > 0
                and idx_te.size > 0
                and idx_tr.max() < len(y)
                and idx_te.max() < len(y)
                and len(set(idx_tr.tolist()) & set(idx_te.tolist())) == 0
            )
        except Exception:
            ok = False
        if ok:
            return (
                {
                    "X_train": X[idx_tr],
                    "y_train": y[idx_tr],
                    "X_test": X[idx_te],
                    "y_test": y[idx_te],
                },
                "índices de la división guardados por T1 en entrada/T1/",
            )
    return None, None


def main():
    # ------------------------------------------------------------------
    # 1) Datos y división única de T1 (70/30 estratificada, random_state=42)
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X, y = data.data, data.target

    split_previo, fuente_division = intentar_cargar_split_T1(X, y)
    if split_previo is not None:
        X_train = np.asarray(split_previo["X_train"], dtype=float)
        X_test = np.asarray(split_previo["X_test"], dtype=float)
        y_train = np.asarray(split_previo["y_train"]).ravel()
        y_test = np.asarray(split_previo["y_test"]).ravel()
    else:
        # train_test_split es determinista con la misma semilla, la misma
        # fracción de prueba y la misma estratificación: esto reproduce
        # exactamente la división única creada en T1.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        fuente_division = (
            "recreada de forma determinista con train_test_split(test_size=0.3, "
            "random_state=42, stratify=y): los mismos parámetros con los que T1 "
            "creó la división única"
        )

    # ------------------------------------------------------------------
    # 2) Verificación contra los resultados reportados por T1
    # ------------------------------------------------------------------
    verificacion = {"fuente_division": fuente_division}
    ruta_t1 = DIR_T1 / "resultados.json"
    if ruta_t1.exists():
        try:
            t1 = json.loads(ruta_t1.read_text(encoding="utf-8"))
            verificacion["archivo_T1_leido"] = True
            verificacion["tamano_entrenamiento_coincide"] = bool(
                len(y_train) == t1.get("tamano_entrenamiento")
            )
            verificacion["tamano_prueba_coincide"] = bool(
                len(y_test) == t1.get("tamano_prueba")
            )
            c_ent = t1.get("conteo_clases_entrenamiento") or {}
            c_pru = t1.get("conteo_clases_prueba") or {}
            if c_ent:
                verificacion["conteo_clases_entrenamiento_coincide"] = bool(
                    int((y_train == 0).sum()) == c_ent.get("malignant")
                    and int((y_train == 1).sum()) == c_ent.get("benign")
                )
            if c_pru:
                verificacion["conteo_clases_prueba_coincide"] = bool(
                    int((y_test == 0).sum()) == c_pru.get("malignant")
                    and int((y_test == 1).sum()) == c_pru.get("benign")
                )
        except Exception as exc:
            verificacion["archivo_T1_leido"] = False
            verificacion["error_leyendo_T1"] = str(exc)
    else:
        verificacion["archivo_T1_encontrado"] = False

    # ------------------------------------------------------------------
    # 3) Estandarización: StandardScaler ajustado SOLO con el entrenamiento
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)  # fit SOLO con entrenamiento
    X_test_std = scaler.transform(X_test)        # prueba: solo se transforma

    # ------------------------------------------------------------------
    # 4) Modelos: generativo (GaussianNB) y discriminativo (reg. logística)
    # ------------------------------------------------------------------
    # GaussianNB modela cada atributo con una gaussiana independiente, por lo
    # que es invariante a transformaciones afines por atributo: estandarizar
    # no altera sus decisiones de forma apreciable. Se usan los atributos
    # estandarizados para ambos modelos por consistencia con el enunciado.
    gnb = GaussianNB()
    gnb.fit(X_train_std, y_train)
    y_pred_gnb = gnb.predict(X_test_std)

    logreg = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    logreg.fit(X_train_std, y_train)
    y_pred_lr = logreg.predict(X_test_std)

    # ------------------------------------------------------------------
    # 5) Métricas sobre el conjunto de prueba
    # ------------------------------------------------------------------
    acc_gnb = float(accuracy_score(y_test, y_pred_gnb))
    f1_gnb = float(f1_score(y_test, y_pred_gnb, average="macro"))
    acc_lr = float(accuracy_score(y_test, y_pred_lr))
    f1_lr = float(f1_score(y_test, y_pred_lr, average="macro"))

    cm_gnb = confusion_matrix(y_test, y_pred_gnb, labels=[0, 1]).tolist()
    cm_lr = confusion_matrix(y_test, y_pred_lr, labels=[0, 1]).tolist()

    tabla_metricas = [
        {
            "modelo": "Naive Bayes gaussiano (GaussianNB)",
            "accuracy_prueba": acc_gnb,
            "f1_macro_prueba": f1_gnb,
        },
        {
            "modelo": "Regresión logística (atributos estandarizados)",
            "accuracy_prueba": acc_lr,
            "f1_macro_prueba": f1_lr,
        },
    ]

    tabla_md = (
        "| Modelo | Accuracy (prueba) | F1 macro (prueba) |\n"
        "|---|---|---|\n"
        f"| Naive Bayes gaussiano (GaussianNB) | {acc_gnb:.6f} | {f1_gnb:.6f} |\n"
        f"| Regresión logística (atributos estandarizados) | {acc_lr:.6f} | {f1_lr:.6f} |"
    )

    # ------------------------------------------------------------------
    # 6) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T2",
        "descripcion": (
            "GaussianNB y regresión logística (atributos estandarizados con "
            "StandardScaler ajustado solo con el entrenamiento), entrenados con la "
            "división 70/30 estratificada de T1 y evaluados sobre el conjunto de prueba."
        ),
        "dataset": "breast_cancer_wisconsin_diagnostic (sklearn.datasets.load_breast_cancer)",
        "division_usada": {
            "fuente": fuente_division,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "estratificada_por_clase": True,
            "tamano_entrenamiento": int(len(y_train)),
            "tamano_prueba": int(len(y_test)),
        },
        "verificacion_contra_T1": verificacion,
        "preprocesamiento": {
            "escalador": "StandardScaler",
            "ajustado_solo_con_entrenamiento": True,
            "atributos_estandarizados_para_regresion_logistica": True,
            "nota_gaussian_nb": (
                "GaussianNB es invariante a transformaciones afines por atributo, "
                "de modo que estandarizar no altera sus decisiones de forma "
                "apreciable; se usan los atributos estandarizados para ambos "
                "modelos por consistencia."
            ),
            "medias_scaler_entrenamiento": [float(v) for v in scaler.mean_],
            "desviaciones_scaler_entrenamiento": [float(v) for v in scaler.scale_],
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
                    "random_state": RANDOM_STATE,
                    "resto": "por defecto",
                },
                "iteraciones_usadas": int(logreg.n_iter_[0]),
            },
        },
        "tabla_metricas_prueba": tabla_metricas,
        "accuracy_prueba_gaussian_nb": acc_gnb,
        "f1_macro_prueba_gaussian_nb": f1_gnb,
        "accuracy_prueba_regresion_logistica": acc_lr,
        "f1_macro_prueba_regresion_logistica": f1_lr,
        "matrices_de_confusion_prueba": {
            "orden_etiquetas": ["0 = malignant", "1 = benign"],
            "gaussian_nb": cm_gnb,
            "regresion_logistica": cm_lr,
        },
        "tabla_markdown": tabla_md,
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 7) Figuras PNG
    # ------------------------------------------------------------------
    # 7a) Tabla de métricas
    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    ax.axis("off")
    tbl = ax.table(
        cellText=[
            [f"{acc_gnb:.4f}", f"{f1_gnb:.4f}"],
            [f"{acc_lr:.4f}", f"{f1_lr:.4f}"],
        ],
        rowLabels=["GaussianNB", "Regresión logística"],
        colLabels=["Accuracy (prueba)", "F1 macro (prueba)"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.8)
    ax.set_title(
        "Parte 2 — Métricas en el conjunto de prueba\n"
        "(división 70/30 estratificada de T1; escalador ajustado solo con entrenamiento)",
        fontsize=10,
    )
    plt.savefig("tabla_parte2_metricas_prueba.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 7b) Barras comparativas
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    modelos = ["GaussianNB", "Regresión logística"]
    xpos = np.arange(len(modelos))
    ancho = 0.35
    ax.bar(xpos - ancho / 2, [acc_gnb, acc_lr], ancho, label="Accuracy", color="#4C72B0")
    ax.bar(xpos + ancho / 2, [f1_gnb, f1_lr], ancho, label="F1 macro", color="#DD8452")
    ax.set_xticks(xpos)
    ax.set_xticklabels(modelos)
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("Valor en el conjunto de prueba")
    ax.set_title("Parte 2 — Generativo vs discriminativo (conjunto de prueba)")
    for i, (a, f1v) in enumerate([(acc_gnb, f1_gnb), (acc_lr, f1_lr)]):
        ax.text(i - ancho / 2, a + 0.015, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + ancho / 2, f1v + 0.015, f"{f1v:.3f}", ha="center", fontsize=9)
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig("comparacion_modelos_parte2.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 8) Resumen
    # ------------------------------------------------------------------
    print("=" * 66)
    print("T2 — Parte 2: GaussianNB vs regresión logística (conjunto de prueba)")
    print("=" * 66)
    print(f"División: {fuente_division}")
    print(f"Tamaños: entrenamiento={len(y_train)} | prueba={len(y_test)}")
    print("-" * 66)
    df_tabla = pd.DataFrame(tabla_metricas)
    print(df_tabla.to_string(index=False))
    print("-" * 66)
    print("Salidas: resultados.json, tabla_parte2_metricas_prueba.png,")
    print("         comparacion_modelos_parte2.png")


if __name__ == "__main__":
    main()
