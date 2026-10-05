# -*- coding: utf-8 -*-
"""
T1 — Carga de Breast Cancer Wisconsin (Diagnostic) y división única 70/30 estratificada.

Produce:
  - resultados.json               (contrato de la subtarea: cifras del dataset y del split)
  - split_T1.npz                  (X/y de train y test + índices, reutilizable en T2, T3, ...)
  - split_T1.json                 (índices de la división, reutilizable)
  - T1_distribucion_clases.png    (verificación visual de la estratificación)
  - Copias de los artefactos en entrada/T1/ para las subtareas siguientes.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.30
TRAIN_SIZE = 0.70


def conteo_proporcion(y, target_names):
    """Conteos y proporciones por clase (claves = nombre de la clase)."""
    n = len(y)
    conteo, proporcion = {}, {}
    for c in np.unique(y):
        c_int = int(c)
        nombre = str(target_names[c_int])
        n_c = int(np.sum(y == c_int))
        conteo[nombre] = n_c
        proporcion[nombre] = n_c / n
    return conteo, proporcion


def escribir_json(objeto, rutas):
    for ruta in rutas:
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(objeto, f, indent=2, ensure_ascii=False)


def main():
    # ---------- Carga del dataset ----------
    data = load_breast_cancer()
    X, y = data.data, data.target
    target_names = [str(t) for t in data.target_names]
    n_ejemplos, n_atributos = X.shape

    conteo_total, prop_total = conteo_proporcion(y, data.target_names)

    # ---------- División única: 70/30 estratificada por clase, random_state=42 ----------
    indices = np.arange(n_ejemplos)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices,
        test_size=TEST_SIZE,
        train_size=TRAIN_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    conteo_train, prop_train = conteo_proporcion(y_train, data.target_names)
    conteo_test, prop_test = conteo_proporcion(y_test, data.target_names)

    # ---------- Verificación de la estratificación y de la integridad del split ----------
    dif_train = {c: abs(prop_train[c] - prop_total[c]) for c in target_names}
    dif_test = {c: abs(prop_test[c] - prop_total[c]) for c in target_names}
    max_dif = float(max(list(dif_train.values()) + list(dif_test.values())))

    set_train, set_test = set(idx_train.tolist()), set(idx_test.tolist())
    sin_solape = len(set_train & set_test) == 0
    cubre_todo = sorted(set_train | set_test) == list(range(n_ejemplos))
    estratificacion_ok = bool(max_dif <= 0.01 and sin_solape and cubre_todo)

    # ---------- Guardar la división (única y reutilizable) ----------
    Path("entrada/T1").mkdir(parents=True, exist_ok=True)
    np.savez(
        "split_T1.npz",
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        idx_train=idx_train, idx_test=idx_test,
    )
    np.savez(
        "entrada/T1/split_T1.npz",
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        idx_train=idx_train, idx_test=idx_test,
    )
    split_indices = {
        "descripcion": "División única de la tarea: 70/30 estratificada por clase, random_state=42",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_size": TRAIN_SIZE,
        "estratificado_por_clase": True,
        "indices_train": idx_train.tolist(),
        "indices_test": idx_test.tolist(),
    }
    escribir_json(split_indices, ["split_T1.json", "entrada/T1/split_T1.json"])

    # ---------- Contrato de resultados ----------
    resultados = {
        "subtarea": "T1",
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "fuente": "sklearn.datasets.load_breast_cancer",
        "n_ejemplos": int(n_ejemplos),
        "n_atributos": int(n_atributos),
        "clases": {str(i): nombre for i, nombre in enumerate(target_names)},
        "conteo_por_clase_total": conteo_total,
        "proporcion_por_clase_total": prop_total,
        "division": {
            "descripcion": "Única división de la tarea: 70% entrenamiento / 30% prueba, estratificada por clase",
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "train_size": TRAIN_SIZE,
            "estratificada_por_clase": True,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "fraccion_train": len(y_train) / n_ejemplos,
            "fraccion_test": len(y_test) / n_ejemplos,
            "conteo_por_clase_train": conteo_train,
            "conteo_por_clase_test": conteo_test,
            "proporcion_por_clase_train": prop_train,
            "proporcion_por_clase_test": prop_test,
        },
        "verificacion_estratificacion": {
            "diferencia_proporciones_train_vs_total": dif_train,
            "diferencia_proporciones_test_vs_total": dif_test,
            "max_diferencia_proporciones": max_dif,
            "sin_solape_entre_train_y_test": bool(sin_solape),
            "union_train_test_cubre_todos_los_ejemplos": bool(cubre_todo),
            "estratificacion_correcta": estratificacion_ok,
        },
        "indices_train": idx_train.tolist(),
        "indices_test": idx_test.tolist(),
        "archivos_generados": [
            "resultados.json",
            "split_T1.npz",
            "split_T1.json",
            "T1_distribucion_clases.png",
            "entrada/T1/resultados.json",
            "entrada/T1/split_T1.npz",
            "entrada/T1/split_T1.json",
        ],
    }
    escribir_json(resultados, ["resultados.json", "entrada/T1/resultados.json"])

    # ---------- Figura de verificación ----------
    x_pos = np.arange(len(target_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x_pos - width, [conteo_total[c] for c in target_names], width,
           label=f"Total (n={n_ejemplos})")
    ax.bar(x_pos, [conteo_train[c] for c in target_names], width,
           label=f"Entrenamiento (n={len(y_train)})")
    ax.bar(x_pos + width, [conteo_test[c] for c in target_names], width,
           label=f"Prueba (n={len(y_test)})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(target_names)
    ax.set_ylabel("Número de casos")
    ax.set_title("Breast Cancer Wisconsin (Diagnostic): casos por clase y partición")
    ax.legend()
    fig.tight_layout()
    fig.savefig("T1_distribucion_clases.png", dpi=120)
    plt.close(fig)

    # ---------- Resumen ----------
    print("=" * 62)
    print("T1 — Breast Cancer Wisconsin (Diagnostic)")
    print(f"  Ejemplos: {n_ejemplos} | Atributos: {n_atributos}")
    for c in target_names:
        print(f"  Clase '{c}': {conteo_total[c]} casos "
              f"({prop_total[c] * 100:.2f} % del total)")
    print(f"  División 70/30 estratificada (random_state={RANDOM_STATE}):")
    print(f"    Train: {len(y_train)} ejemplos "
          f"({len(y_train) / n_ejemplos * 100:.2f} %) -> {conteo_train}")
    print(f"    Test : {len(y_test)} ejemplos "
          f"({len(y_test) / n_ejemplos * 100:.2f} %) -> {conteo_test}")
    print(f"  Estratificación correcta: {estratificacion_ok} "
          f"(máx. desviación de proporciones = {max_dif:.6f})")
    print("  División guardada en split_T1.npz / split_T1.json (y en entrada/T1/).")
    print("=" * 62)


if __name__ == "__main__":
    main()
