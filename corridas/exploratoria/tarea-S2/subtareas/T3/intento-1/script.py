#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 — Hacer visible la evidencia por tienda.

Figura con UNA observación por tienda (16 tiendas) que muestra el signo y la
magnitud de difference = error_a − error_b, con una línea de referencia
visible en cero. Se guarda en output/store_differences.png y se identifican
las diferencias de mayor magnitud en cada dirección (positiva y negativa).

Entradas:
  - data/model_errors.csv        (datos de la tarea)
  - entrada/T2/resultados.json   (resultado previo, solo para verificación)
Salidas:
  - output/store_differences.png (figura requerida por la subtarea)
  - t3_store_differences.png     (copia en la carpeta actual)
  - resultados.json              (contrato de la subtarea)
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RUTA_CSV = "data/model_errors.csv"
RUTA_T2 = os.path.join("entrada", "T2", "resultados.json")
RUTA_FIGURA = os.path.join("output", "store_differences.png")
RUTA_FIGURA_COPIA = "t3_store_differences.png"
RUTA_RESULTADOS = "resultados.json"
COLUMNAS_REQUERIDAS = ["store_id", "error_a", "error_b"]


def cargar_y_validar_datos():
    """Carga data/model_errors.csv con las mismas validaciones que T2:
    columnas exactas, store_id único y no nulo, errores numéricos, finitos y
    no negativos, y exactamente 16 tiendas (una observación por tienda)."""
    if not os.path.exists(RUTA_CSV):
        raise FileNotFoundError(f"No se encontró el archivo de datos: {RUTA_CSV}")
    df = pd.read_csv(RUTA_CSV)

    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en {RUTA_CSV}: {faltantes}")

    if df["store_id"].isna().any():
        raise ValueError("store_id (identificador de tienda) contiene valores nulos.")
    if df["store_id"].duplicated().any():
        duplicados = sorted(
            df.loc[df["store_id"].duplicated(), "store_id"].astype(str).unique().tolist()
        )
        raise ValueError(f"store_id (identificador de tienda) duplicado: {duplicados}")

    for col in ("error_a", "error_b"):
        num = pd.to_numeric(df[col], errors="coerce")
        if num.isna().any():
            raise ValueError(f"La columna '{col}' contiene valores no numéricos.")
        valores = num.to_numpy(dtype=float)
        if not np.isfinite(valores).all():
            raise ValueError(f"La columna '{col}' contiene valores no finitos (NaN/inf).")
        if (valores < 0).any():
            raise ValueError(f"La columna '{col}' contiene valores negativos.")
        df[col] = valores

    if len(df) != 16:
        raise ValueError(f"Se esperaban 16 tiendas y se encontraron {len(df)}.")

    df = df.sort_values("store_id", kind="mergesort").reset_index(drop=True)
    df["difference"] = df["error_a"] - df["error_b"]
    df["favors"] = np.where(
        df["difference"] > 0, "B", np.where(df["difference"] < 0, "A", "empate")
    )
    return df


def verificar_coherencia_con_t2(df):
    """Si existe el resultado de T2, comprueba que las difference recalculadas
    coincidan con su tabla pareada (emparejamiento por store_id)."""
    if not os.path.exists(RUTA_T2):
        return None
    with open(RUTA_T2, "r", encoding="utf-8") as f:
        t2 = json.load(f)
    tabla_t2 = t2.get("tabla_pareada")
    if not tabla_t2:
        return None
    mapa_t2 = {str(r["store_id"]): float(r["difference"]) for r in tabla_t2}
    ok = all(
        sid in mapa_t2 and np.isclose(mapa_t2[sid], float(d), atol=1e-9, rtol=0.0)
        for sid, d in zip(df["store_id"], df["difference"])
    )
    if not ok:
        raise ValueError(
            "Inconsistencia: las difference recalculadas no coinciden con la "
            "tabla pareada de T2 (entrada/T2/resultados.json)."
        )
    return True


def crear_figura(df):
    """Barra por tienda (16 en total): signo y magnitud de difference,
    colores por dirección del favor y línea de referencia en cero."""
    os.makedirs("output", exist_ok=True)

    ids = df["store_id"].tolist()
    diffs = df["difference"].to_numpy(dtype=float)
    favors = df["favors"].tolist()

    color_pos, color_neg, color_empate = "#2b8cbe", "#e34a33", "#9e9e9e"
    colores = [
        color_pos if f == "B" else (color_neg if f == "A" else color_empate)
        for f in favors
    ]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    barras = ax.bar(ids, diffs, color=colores, edgecolor="black",
                    linewidth=0.6, zorder=3)

    # Referencia visible en cero
    ax.axhline(0, color="black", linewidth=1.8, zorder=4)

    lo, hi = float(diffs.min()), float(diffs.max())
    rango = max(hi - lo, 1e-9)
    ax.set_ylim(lo - 0.32 * rango, hi + 0.34 * rango)

    i_max = int(np.argmax(diffs))   # mayor diferencia positiva
    i_min = int(np.argmin(diffs))   # mayor diferencia negativa (en magnitud)

    # Signo y magnitud de cada diferencia: una etiqueta por tienda
    for i, (barra, d) in enumerate(zip(barras, diffs)):
        x = barra.get_x() + barra.get_width() / 2.0
        if i == i_max and d > 0:
            ax.text(x, d + 0.13 * rango, f"máx + (favorece a B)\n{d:+.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        elif i == i_min and d < 0:
            ax.text(x, d - 0.13 * rango, f"máx − (favorece a A)\n{d:+.2f}",
                    ha="center", va="top", fontsize=9, fontweight="bold")
        elif d >= 0:
            ax.text(x, d + 0.045 * rango, f"{d:+.2f}",
                    ha="center", va="bottom", fontsize=9)
        else:
            ax.text(x, d - 0.045 * rango, f"{d:+.2f}",
                    ha="center", va="top", fontsize=9)

    # Resaltar las barras de diferencias extremas
    for i in (i_max, i_min):
        barras[i].set_edgecolor("black")
        barras[i].set_linewidth(2.2)

    n_a = int((df["favors"] == "A").sum())
    n_b = int((df["favors"] == "B").sum())
    n_e = int((df["favors"] == "empate").sum())

    ax.set_title(
        "T3 · Diferencia de error por tienda: difference = error_a − error_b\n"
        f"(una observación por tienda · {n_a} favorecen a A · "
        f"{n_b} favorecen a B · {n_e} empates)",
        fontsize=11,
    )
    ax.set_xlabel("Tienda (store_id)")
    ax.set_ylabel("difference = error_a − error_b")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    leyenda = [
        Patch(facecolor=color_pos, edgecolor="black",
              label="difference > 0 → favorece a B"),
        Patch(facecolor=color_neg, edgecolor="black",
              label="difference < 0 → favorece a A"),
    ]
    if n_e:
        leyenda.append(Patch(facecolor=color_empate, edgecolor="black",
                             label="difference = 0 → empate"))
    ax.legend(handles=leyenda, loc="upper left", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(RUTA_FIGURA, dpi=120)
    fig.savefig(RUTA_FIGURA_COPIA, dpi=120)
    plt.close(fig)


def main():
    df = cargar_y_validar_datos()
    verificacion_t2 = verificar_coherencia_con_t2(df)

    crear_figura(df)

    if not (os.path.exists(RUTA_FIGURA) and os.path.getsize(RUTA_FIGURA) > 0):
        raise RuntimeError(f"No se pudo guardar la figura en {RUTA_FIGURA}")

    ids = df["store_id"].tolist()
    diffs = df["difference"].to_numpy(dtype=float)
    i_max = int(np.argmax(diffs))
    i_min = int(np.argmin(diffs))
    i_abs = int(np.argmax(np.abs(diffs)))

    n_a = int((df["favors"] == "A").sum())
    n_b = int((df["favors"] == "B").sum())
    n_e = int((df["favors"] == "empate").sum())

    resultados = {
        "subtarea": "T3",
        "descripcion": (
            "Figura con una observación por tienda (16 tiendas) que muestra el "
            "signo y la magnitud de difference = error_a − error_b, con línea "
            "de referencia visible en cero."
        ),
        "fuente_datos": RUTA_CSV,
        "figura_guardada_en": RUTA_FIGURA,
        "figura_existe": os.path.exists(RUTA_FIGURA),
        "dpi_figura": 120,
        "copia_figura_carpeta_actual": RUTA_FIGURA_COPIA,
        "n_tiendas_mostradas": int(len(df)),
        "una_observacion_por_tienda": bool(df["store_id"].is_unique and len(df) == 16),
        "tiendas_mostradas": ids,
        "referencia_cero_incluida": True,
        "diferencias_por_tienda": [
            {
                "store_id": str(r.store_id),
                "error_a": float(r.error_a),
                "error_b": float(r.error_b),
                "difference": float(r.difference),
                "favors": str(r.favors),
            }
            for r in df.itertuples(index=False)
        ],
        "diferencia_maxima_positiva": {
            "store_id": ids[i_max],
            "difference": float(diffs[i_max]),
            "favors": str(df["favors"].iloc[i_max]),
        },
        "diferencia_maxima_negativa": {
            "store_id": ids[i_min],
            "difference": float(diffs[i_min]),
            "favors": str(df["favors"].iloc[i_min]),
        },
        "tienda_mayor_diferencia_positiva": ids[i_max],
        "valor_mayor_diferencia_positiva": float(diffs[i_max]),
        "tienda_mayor_diferencia_negativa": ids[i_min],
        "valor_mayor_diferencia_negativa": float(diffs[i_min]),
        "mayor_magnitud_absoluta": {
            "store_id": ids[i_abs],
            "difference": float(diffs[i_abs]),
            "abs_difference": float(abs(diffs[i_abs])),
        },
        "n_tiendas_favorecen_A": n_a,
        "n_tiendas_favorecen_B": n_b,
        "n_empates": n_e,
        "ids_favor_A": df.loc[df["favors"] == "A", "store_id"].tolist(),
        "ids_favor_B": df.loc[df["favors"] == "B", "store_id"].tolist(),
        "ids_empate": df.loc[df["favors"] == "empate", "store_id"].tolist(),
        "min_difference": float(diffs.min()),
        "max_difference": float(diffs.max()),
        "media_diferencias": float(diffs.mean()),
        "suma_diferencias": float(diffs.sum()),
        "coherente_con_T2": verificacion_t2,
    }

    with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print("=" * 64)
    print("T3 · Figura de diferencias por tienda")
    print("=" * 64)
    print(f"Figura guardada en : {RUTA_FIGURA} "
          f"(existe: {resultados['figura_existe']})")
    print(f"Tiendas mostradas  : {len(df)} (una observación por tienda)")
    print(f"Favorecen a A      : {n_a} -> {resultados['ids_favor_A']}")
    print(f"Favorecen a B      : {n_b} -> {resultados['ids_favor_B']}")
    print(f"Empates            : {n_e}")
    print(f"Mayor diferencia positiva : {ids[i_max]} = {diffs[i_max]:+.4f} "
          f"(favorece a B)")
    print(f"Mayor diferencia negativa : {ids[i_min]} = {diffs[i_min]:+.4f} "
          f"(favorece a A)")
    print(f"Mayor |difference|        : {ids[i_abs]} = {diffs[i_abs]:+.4f}")
    if verificacion_t2 is not None:
        print(f"Coherencia con T2  : {'OK' if verificacion_t2 else 'FALLÓ'}")
    print(f"Contrato escrito en: {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
