#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUBTAREA T2: Auditar data/model_errors.csv y construir la tabla pareada.

- Valida que el CSV tenga EXACTAMENTE las columnas store_id, error_a, error_b.
- Valida que cada store_id sea único y no nulo (ValueError que menciona el identificador).
- Valida que los errores sean numéricos, finitos y no negativos (ValueError en caso contrario).
- Construye la tabla pareada de 16 filas: columnas originales + difference + favors.
- Verifica el emparejamiento mediante store_id (no basta con la longitud de los vectores).
- Cuenta cuántas tiendas favorecen a A, cuántas a B y si hay empates.
- NO modifica el CSV original (se comprueba con SHA-256 antes y después).
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RUTA_CSV = "data/model_errors.csv"
RUTA_TABLA_SALIDA = "output/store_differences.csv"
RUTA_RESULTADOS = "resultados.json"
RUTA_FIGURA = "t2_diferencias_por_tienda.png"

COLUMNAS_ESPERADAS = ["store_id", "error_a", "error_b"]
N_FILAS_ESPERADAS = 16


def sha256_archivo(ruta: str) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(65536), b""):
            h.update(bloque)
    return h.hexdigest()


def validar_columnas(df: pd.DataFrame) -> None:
    """El CSV debe tener exactamente las columnas store_id, error_a, error_b."""
    columnas = list(df.columns)
    if len(columnas) != len(COLUMNAS_ESPERADAS) or sorted(columnas) != sorted(COLUMNAS_ESPERADAS):
        raise ValueError(
            f"El CSV debe tener exactamente las columnas {COLUMNAS_ESPERADAS}; "
            f"se encontraron {columnas}."
        )


def validar_store_id(df: pd.DataFrame) -> None:
    """Cada identificador de tienda debe ser único y no nulo."""
    nulos = df["store_id"].isna()
    if nulos.any():
        filas = df.index[nulos].tolist()
        raise ValueError(
            "El identificador 'store_id' no puede ser nulo; "
            f"filas con store_id nulo (índice): {filas}."
        )
    dup_mask = df["store_id"].duplicated(keep=False)
    if dup_mask.any():
        ids_dup = sorted({str(v) for v in df.loc[dup_mask, "store_id"]})
        filas = df.index[dup_mask].tolist()
        raise ValueError(
            f"El identificador 'store_id' debe ser único; identificadores repetidos: "
            f"{ids_dup} (filas índice: {filas})."
        )


def validar_y_coercionar_errores(df: pd.DataFrame) -> dict:
    """Los errores deben ser numéricos, finitos y no negativos."""
    coerciones = {}
    for col in ("error_a", "error_b"):
        serie = df[col]
        numerico = pd.to_numeric(serie, errors="coerce")
        no_numericos = numerico.isna() & serie.notna()
        if no_numericos.any():
            detalles = {int(i): str(v) for i, v in serie[no_numericos].items()}
            raise ValueError(
                f"La columna '{col}' debe ser numérica; valores no numéricos: {detalles}."
            )
        arr = numerico.to_numpy(dtype=float)
        no_finitos = ~np.isfinite(arr)
        if no_finitos.any():
            filas = df.index[no_finitos].tolist()
            raise ValueError(
                f"La columna '{col}' debe contener valores finitos (sin NaN/inf); "
                f"filas con valores no finitos: {filas}."
            )
        negativos = arr < 0
        if negativos.any():
            detalles = {int(i): float(v) for i, v in zip(df.index[negativos], arr[negativos])}
            raise ValueError(
                f"La columna '{col}' debe contener errores no negativos; "
                f"valores negativos: {detalles}."
            )
        coerciones[col] = numerico
    return coerciones


def validar_n_filas(df: pd.DataFrame) -> None:
    if len(df) != N_FILAS_ESPERADAS:
        raise ValueError(
            f"Se esperaban {N_FILAS_ESPERADAS} tiendas en el CSV y se encontraron {len(df)}."
        )


def construir_tabla_pareada(df_work: pd.DataFrame) -> pd.DataFrame:
    """Conserva las tres columnas originales y añade difference y favors."""
    tabla = df_work[COLUMNAS_ESPERADAS].copy()
    tabla["difference"] = tabla["error_a"] - tabla["error_b"]
    tabla["favors"] = np.select(
        [tabla["difference"] > 0, tabla["difference"] < 0],
        ["B", "A"],
        default="EMPATE",
    )
    return tabla


def verificar_emparejamiento(tabla: pd.DataFrame, fuente: pd.DataFrame) -> None:
    """Verifica los pares mediante store_id: cada fila de la tabla debe corresponder
    a la misma tienda del CSV, y difference debe recalcularse sobre esa misma tienda."""
    if tabla["store_id"].duplicated().any():
        raise ValueError("La tabla pareada contiene store_id repetidos.")
    if set(tabla["store_id"]) != set(fuente["store_id"]):
        raise ValueError("El conjunto de store_id de la tabla no coincide con el del CSV fuente.")
    mapa = fuente.set_index("store_id")[["error_a", "error_b"]]
    for _, fila in tabla.iterrows():
        sid = fila["store_id"]
        if sid not in mapa.index:
            raise ValueError(f"El store_id '{sid}' de la tabla no existe en el CSV fuente.")
        ea, eb = mapa.at[sid, "error_a"], mapa.at[sid, "error_b"]
        if float(fila["error_a"]) != float(ea) or float(fila["error_b"]) != float(eb):
            raise ValueError(
                f"Emparejamiento inválido para el store_id '{sid}': los errores de la tabla "
                "no corresponden a esa misma tienda en el CSV."
            )
        if float(fila["difference"]) != float(ea) - float(eb):
            raise ValueError(
                f"Emparejamiento inválido para el store_id '{sid}': 'difference' no coincide "
                "con error_a - error_b calculado sobre esa misma tienda."
            )


def main():
    # Integridad del CSV original: hash antes de tocar nada
    hash_antes = sha256_archivo(RUTA_CSV)

    df = pd.read_csv(RUTA_CSV)

    # ---- Validaciones (ValueError descriptivos) ----
    validar_columnas(df)
    validar_store_id(df)
    coerciones = validar_y_coercionar_errores(df)
    validar_n_filas(df)

    # Copia de trabajo con errores ya validados como numéricos (el CSV no se toca)
    df_work = df.copy()
    for col, serie in coerciones.items():
        df_work[col] = serie

    # ---- Tabla pareada ----
    tabla = construir_tabla_pareada(df_work)
    verificar_emparejamiento(tabla, df_work)

    ids_a = [str(x) for x in tabla.loc[tabla["favors"] == "A", "store_id"]]
    ids_b = [str(x) for x in tabla.loc[tabla["favors"] == "B", "store_id"]]
    ids_empate = [str(x) for x in tabla.loc[tabla["favors"] == "EMPATE", "store_id"]]
    n_a, n_b, n_emp = len(ids_a), len(ids_b), len(ids_empate)

    # ---- Guardar tabla pareada (CSV entregable) ----
    os.makedirs(os.path.dirname(RUTA_TABLA_SALIDA), exist_ok=True)
    tabla.to_csv(RUTA_TABLA_SALIDA, index=False, encoding="utf-8")

    # Control de lectura de la tabla guardada: 16 filas x 5 columnas
    control = pd.read_csv(RUTA_TABLA_SALIDA)
    if control.shape != (N_FILAS_ESPERADAS, 5):
        raise RuntimeError(
            f"La tabla guardada en {RUTA_TABLA_SALIDA} tiene forma {control.shape}; "
            f"se esperaba ({N_FILAS_ESPERADAS}, 5)."
        )

    # ---- Figura: diferencia por tienda ----
    fig, ax = plt.subplots(figsize=(9.5, 5))
    fav_arr = tabla["favors"].to_numpy()
    colores = np.select(
        [fav_arr == "A", fav_arr == "B"], ["#1f77b4", "#d62728"], default="#7f7f7f"
    )
    ax.bar(range(len(tabla)), tabla["difference"].to_numpy(), color=colores,
           edgecolor="black", linewidth=0.4)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(range(len(tabla)))
    ax.set_xticklabels([str(s) for s in tabla["store_id"]], rotation=45, ha="right")
    ax.set_xlabel("store_id")
    ax.set_ylabel("difference = error_a - error_b")
    ax.set_title("T2 · Diferencia de errores por tienda (pareada por store_id)")
    ax.legend(
        handles=[Patch(color="#1f77b4", label="favors = A"),
                 Patch(color="#d62728", label="favors = B"),
                 Patch(color="#7f7f7f", label="empate")],
        title="favors",
    )
    for i, d in enumerate(tabla["difference"].to_numpy()):
        ax.annotate(f"{d:+.2f}", (i, d), ha="center",
                    va="bottom" if d >= 0 else "top", fontsize=7)
    fig.tight_layout()
    plt.savefig(RUTA_FIGURA, dpi=120)
    plt.close(fig)

    # ---- Integridad del CSV original: hash después ----
    hash_despues = sha256_archivo(RUTA_CSV)
    if hash_despues != hash_antes:
        raise RuntimeError("El CSV original fue modificado durante la ejecución.")
    csv_intacto = True

    # ---- resultados.json (contrato de la subtarea) ----
    registros = [
        {
            "store_id": str(r.store_id),
            "error_a": float(r.error_a),
            "error_b": float(r.error_b),
            "difference": float(r.difference),
            "favors": str(r.favors),
        }
        for r in tabla.itertuples(index=False)
    ]

    resultados = {
        "subtarea": "T2",
        "archivo_entrada": RUTA_CSV,
        "validaciones_aplicadas": [
            "Columnas exactas store_id, error_a, error_b (ValueError en caso contrario)",
            "store_id único y no nulo (ValueError que menciona el identificador)",
            "error_a y error_b numéricos, finitos y no negativos (ValueError en caso contrario)",
            "Número de tiendas igual a 16",
            "Emparejamiento verificado por store_id (difference recalculada por tienda)",
            "CSV original intacto (SHA-256 idéntico antes y después)",
        ],
        "n_filas_tabla_pareada": int(tabla.shape[0]),
        "n_columnas_tabla_pareada": int(tabla.shape[1]),
        "columnas_tabla_pareada": list(tabla.columns),
        "tabla_pareada": registros,
        "tiendas_favorecen_A": n_a,
        "tiendas_favorecen_B": n_b,
        "empates": n_emp,
        "hay_empates": bool(n_emp > 0),
        "ids_favor_A": ids_a,
        "ids_favor_B": ids_b,
        "ids_empate": ids_empate,
        "emparejamiento_verificado_por_store_id": True,
        "sha256_csv_antes": hash_antes,
        "sha256_csv_despues": hash_despues,
        "csv_original_intacto": csv_intacto,
        "tabla_pareada_guardada_en": RUTA_TABLA_SALIDA,
        "figura_guardada_en": RUTA_FIGURA,
    }

    with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ---- Resumen ----
    print("=== Resumen T2 ===")
    print("CSV validado: columnas exactas, store_id único/no nulo, "
          "errores numéricos/finitos/no negativos.")
    print(f"Tabla pareada: {tabla.shape[0]} filas x {tabla.shape[1]} columnas; "
          "emparejamiento verificado por store_id.")
    print(f"Tiendas que favorecen a A: {n_a} -> {ids_a}")
    print(f"Tiendas que favorecen a B: {n_b} -> {ids_b}")
    print(f"Empates: {n_emp} (hay_empates = {n_emp > 0})")
    print(f"Salidas: {RUTA_TABLA_SALIDA}, {RUTA_RESULTADOS}, {RUTA_FIGURA}")
    print(f"data/model_errors.csv intacto: {csv_intacto} (SHA-256 {hash_antes[:12]}...)")


if __name__ == "__main__":
    main()
