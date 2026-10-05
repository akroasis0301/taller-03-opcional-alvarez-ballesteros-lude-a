# T1: Cargar Breast Cancer Wisconsin (Diagnostic), reportar cifras básicas y crear
# la única división train/test 70/30 estratificada con random_state=42.

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def main():
    # ---------- 1) Carga del dataset ----------
    data = load_breast_cancer()
    X = data.data
    y = data.target
    target_names = [str(t) for t in data.target_names]
    feature_names = [str(f) for f in data.feature_names]

    total_ejemplos = int(X.shape[0])
    n_atributos = int(X.shape[1])

    clases_idx, conteos = np.unique(y, return_counts=True)
    casos_por_clase = {
        target_names[int(c)]: int(n) for c, n in zip(clases_idx, conteos)
    }

    # ---------- 2) División única 70/30 estratificada, random_state=42 ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    tam_train = int(X_train.shape[0])
    tam_test = int(X_test.shape[0])
    frac_train = tam_train / total_ejemplos
    frac_test = tam_test / total_ejemplos

    def conteos_y_proporciones(y_vec):
        cl, ct = np.unique(y_vec, return_counts=True)
        conteo = {target_names[int(c)]: int(n) for c, n in zip(cl, ct)}
        proporcion = {k: v / len(y_vec) for k, v in conteo.items()}
        return conteo, proporcion

    conteos_train, prop_train = conteos_y_proporciones(y_train)
    conteos_test, prop_test = conteos_y_proporciones(y_test)

    # Verificaciones del criterio de éxito
    verif_tamanos = (abs(frac_train - 0.70) < 1e-9) and (abs(frac_test - 0.30) < 1e-9)
    difs_prop = [
        abs(prop_train[k] - prop_test[k]) for k in casos_por_clase
    ]
    difs_prop += [
        abs(prop_train[k] - casos_por_clase[k] / total_ejemplos)
        for k in casos_por_clase
    ]
    max_dif_prop = float(max(difs_prop))
    verif_estratificacion = max_dif_prop < 0.01  # proporciones de clase preservadas

    # ---------- 3) Guardar la división para reutilizarla en subtareas posteriores ----------
    np.savez_compressed(
        "t1_split_breast_cancer.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=np.array(feature_names),
        target_names=np.array(target_names),
    )
    np.save("t1_X_train.npy", X_train)
    np.save("t1_X_test.npy", X_test)
    np.save("t1_y_train.npy", y_train)
    np.save("t1_y_test.npy", y_test)

    # ---------- 4) Figura: distribución de clases ----------
    etiquetas = list(casos_por_clase.keys())
    x_pos = np.arange(len(etiquetas))
    ancho = 0.27
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x_pos - ancho, [casos_por_clase[k] for k in etiquetas], ancho,
           label="Total", color="#4C72B0")
    ax.bar(x_pos, [conteos_train[k] for k in etiquetas], ancho,
           label="Train (70%)", color="#DD8452")
    ax.bar(x_pos + ancho, [conteos_test[k] for k in etiquetas], ancho,
           label="Test (30%)", color="#55A868")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(etiquetas)
    ax.set_ylabel("Número de casos")
    ax.set_title("Breast Cancer Wisconsin: casos por clase (split 70/30, seed=42)")
    ax.legend()
    for i, k in enumerate(etiquetas):
        ax.text(i - ancho, casos_por_clase[k] + 3, str(casos_por_clase[k]),
                ha="center", fontsize=9)
        ax.text(i, conteos_train[k] + 3, str(conteos_train[k]), ha="center", fontsize=9)
        ax.text(i + ancho, conteos_test[k] + 3, str(conteos_test[k]), ha="center", fontsize=9)
    fig.tight_layout()
    plt.savefig("t1_distribucion_clases.png", dpi=120)
    plt.close(fig)

    # ---------- 5) resultados.json (contrato) ----------
    resultados = {
        "subtarea": "T1",
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "total_ejemplos": total_ejemplos,
        "numero_atributos": n_atributos,
        "clases": etiquetas,
        "casos_por_clase": casos_por_clase,
        "split": {
            "test_size": 0.30,
            "train_size": 0.70,
            "estratificado": True,
            "random_state": 42,
            "tam_train": tam_train,
            "tam_test": tam_test,
            "fraccion_train": frac_train,
            "fraccion_test": frac_test,
            "conteos_train": conteos_train,
            "conteos_test": conteos_test,
            "proporciones_train": prop_train,
            "proporciones_test": prop_test,
        },
        "verificacion": {
            "tamanos_70_30": bool(verif_tamanos),
            "proporciones_clase_preservadas": bool(verif_estratificacion),
            "max_diferencia_proporciones": max_dif_prop,
        },
        "archivos_guardados": [
            "t1_split_breast_cancer.npz",
            "t1_X_train.npy",
            "t1_X_test.npy",
            "t1_y_train.npy",
            "t1_y_test.npy",
            "t1_distribucion_clases.png",
        ],
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # ---------- 6) Resumen ----------
    print("=== T1: Carga y división 70/30 estratificada ===")
    print(f"Dataset: Breast Cancer Wisconsin (Diagnostic)")
    print(f"Total de ejemplos: {total_ejemplos}")
    print(f"Número de atributos: {n_atributos}")
    print(f"Casos por clase: {casos_por_clase}")
    print(f"Train: {tam_train} ejemplos ({frac_train:.4f}) | Test: {tam_test} ejemplos ({frac_test:.4f})")
    print(f"Conteos train: {conteos_train}")
    print(f"Conteos test:  {conteos_test}")
    print(f"Proporciones train: {prop_train}")
    print(f"Proporciones test:  {prop_test}")
    print(f"Verificación 70/30: {verif_tamanos} | Estratificación preservada: {verif_estratificacion} "
          f"(max dif. proporciones = {max_dif_prop:.6f})")
    print("Archivos guardados:", ", ".join(resultados["archivos_guardados"]))


if __name__ == "__main__":
    main()
