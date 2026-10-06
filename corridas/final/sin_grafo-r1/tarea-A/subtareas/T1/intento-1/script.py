# T1: Carga de Breast Cancer Wisconsin (Diagnostic), reporte de cifras
# y creación de la división única entrenamiento/prueba 70/30 estratificada (random_state=42).

import json
import numpy as np
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def main():
    # ------------------------------------------------------------------
    # 1) Carga del dataset
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = [str(f) for f in data.feature_names]
    target_names = [str(t) for t in data.target_names]  # 0 = malignant, 1 = benign

    n_ejemplos, n_atributos = X.shape

    # Conteo por clase en el dataset completo
    vals, cnts = np.unique(y, return_counts=True)
    conteo_clase = {target_names[int(v)]: int(c) for v, c in zip(vals, cnts)}
    conteo_clase_numerico = {int(v): int(c) for v, c in zip(vals, cnts)}
    proporcion_clase = {
        target_names[int(v)]: float(c / n_ejemplos) for v, c in zip(vals, cnts)
    }

    # ------------------------------------------------------------------
    # 2) División única 70/30 estratificada por clase, random_state=42
    #    (se guardan los índices para reutilizarlos en las demás subtareas)
    # ------------------------------------------------------------------
    indices = np.arange(n_ejemplos)
    train_idx, test_idx = train_test_split(
        indices, test_size=0.30, stratify=y, random_state=42
    )

    y_train, y_test = y[train_idx], y[test_idx]

    tv, tc = np.unique(y_train, return_counts=True)
    sv, sc = np.unique(y_test, return_counts=True)
    conteo_train = {target_names[int(v)]: int(c) for v, c in zip(tv, tc)}
    conteo_test = {target_names[int(v)]: int(c) for v, c in zip(sv, sc)}
    proporcion_train = {k: float(v / len(train_idx)) for k, v in conteo_train.items()}
    proporcion_test = {k: float(v / len(test_idx)) for k, v in conteo_test.items()}

    # Verificación de la estratificación: proporciones train/test ~ proporciones globales
    estrat_ok = all(
        abs(proporcion_train[k] - proporcion_clase[k]) < 0.01
        and abs(proporcion_test[k] - proporcion_clase[k]) < 0.01
        for k in proporcion_clase
    )

    # ------------------------------------------------------------------
    # 3) Guardar la división para reutilizarla en las demás subtareas
    # ------------------------------------------------------------------
    np.save("train_indices.npy", train_idx.astype(int))
    np.save("test_indices.npy", test_idx.astype(int))
    np.save("X_train.npy", X[train_idx])
    np.save("X_test.npy", X[test_idx])
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    with open("feature_names.json", "w", encoding="utf-8") as f:
        json.dump({"feature_names": feature_names, "target_names": target_names}, f, indent=2)

    # ------------------------------------------------------------------
    # 4) Figura de verificación de la estratificación
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clases = target_names
    x_pos = np.arange(len(clases))
    width = 0.25
    tot = [conteo_clase[c] for c in clases]
    tr = [conteo_train.get(c, 0) for c in clases]
    te = [conteo_test.get(c, 0) for c in clases]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x_pos - width, tot, width, label="Completo", color="gray")
    ax.bar(x_pos, tr, width, label=f"Train (n={len(train_idx)})", color="tab:blue")
    ax.bar(x_pos + width, te, width, label=f"Test (n={len(test_idx)})", color="tab:orange")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(clases)
    ax.set_ylabel("Número de casos")
    ax.set_title("Breast Cancer Wisconsin: distribución de clases y estratificación 70/30")
    ax.legend()
    for i, (a, b, c) in enumerate(zip(tot, tr, te)):
        ax.text(i - width, a + 3, str(a), ha="center", fontsize=9)
        ax.text(i, b + 3, str(b), ha="center", fontsize=9)
        ax.text(i + width, c + 3, str(c), ha="center", fontsize=9)
    fig.tight_layout()
    plt.savefig("t1_distribucion_clases.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T1",
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "n_ejemplos": int(n_ejemplos),
        "n_atributos": int(n_atributos),
        "clases": target_names,
        "conteo_por_clase": conteo_clase,
        "conteo_por_clase_numerico": conteo_clase_numerico,
        "proporcion_por_clase": proporcion_clase,
        "split": {
            "test_size": 0.30,
            "train_size": 0.70,
            "estratificado": True,
            "random_state": 42,
            "unica_division_de_la_tarea": True,
        },
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "fraccion_train": float(len(train_idx) / n_ejemplos),
        "fraccion_test": float(len(test_idx) / n_ejemplos),
        "conteo_train_por_clase": conteo_train,
        "conteo_test_por_clase": conteo_test,
        "proporcion_train_por_clase": proporcion_train,
        "proporcion_test_por_clase": proporcion_test,
        "estratificacion_verificada": bool(estrat_ok),
        "archivos_division": [
            "train_indices.npy",
            "test_indices.npy",
            "X_train.npy",
            "X_test.npy",
            "y_train.npy",
            "y_test.npy",
            "feature_names.json",
        ],
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 6) Resumen
    # ------------------------------------------------------------------
    print("=== T1: Breast Cancer Wisconsin (Diagnostic) ===")
    print(f"Ejemplos: {n_ejemplos} | Atributos: {n_atributos}")
    print(f"Clases: {conteo_clase}")
    print(f"Train: {len(train_idx)} ({len(train_idx)/n_ejemplos:.1%}) -> {conteo_train}")
    print(f"Test : {len(test_idx)} ({len(test_idx)/n_ejemplos:.1%}) -> {conteo_test}")
    print(f"Estratificación verificada: {estrat_ok}")
    print("División guardada en train_indices.npy / test_indices.npy (reutilizable).")
    print("Resultados escritos en resultados.json; figura en t1_distribucion_clases.png")


if __name__ == "__main__":
    main()
