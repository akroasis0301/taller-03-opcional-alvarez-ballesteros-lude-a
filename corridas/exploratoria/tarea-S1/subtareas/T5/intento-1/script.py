#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5 - Guardar y verificar el resultado (round trip Parquet con PyArrow).

1) Reconstruye el DataFrame analizado de la Tarea 3 (columnas originales de
   data/server_measurements.csv + load_score y requires_review registrados por T3).
2) Lo guarda en output/server_analysis.parquet con PyArrow, SIN el índice.
3) Vuelve a leerlo y verifica: filas, columnas, esquema y valores iguales
   antes y después del round trip.
4) Escribe resultados.json e imprime el veredicto en el formato del enunciado.
"""

import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RUTA_CSV = "data/server_measurements.csv"
RUTA_T3 = "entrada/T3/resultados.json"
RUTA_PARQUET = "output/server_analysis.parquet"
RUTA_RESULTADOS = "resultados.json"

PESOS = {"gpu_utilization": 0.5, "cpu_utilization": 0.3, "memory_gb": 0.2}
UMBRAL_LOAD = 1.5
UMBRAL_TEMP = 80.0
COLUMNAS_ESPERADAS = [
    "server", "timestamp", "gpu_utilization", "cpu_utilization",
    "memory_gb", "power_w", "temperature_c", "load_score", "requires_review",
]


def calcular_load_score(df_base: pd.DataFrame, ddof: int = 0) -> np.ndarray:
    """load_score = 0.5*z(gpu_utilization) + 0.3*z(cpu_utilization) + 0.2*z(memory_gb)."""
    total = np.zeros(len(df_base), dtype=np.float64)
    for col, peso in PESOS.items():
        x = df_base[col].to_numpy(dtype=np.float64)
        total += peso * (x - x.mean()) / x.std(ddof=ddof)
    return total


def esquema_como_lista(schema: pa.Schema):
    return [{"nombre": fl.name, "tipo": str(fl.type)} for fl in schema]


def main():
    # ------------------------------------------------------------------
    # 1) Resultados de T3 (definen el DataFrame analizado)
    # ------------------------------------------------------------------
    t3 = {}
    if os.path.exists(RUTA_T3):
        with open(RUTA_T3, "r", encoding="utf-8") as f:
            t3 = json.load(f)

    # ------------------------------------------------------------------
    # 2) DataFrame original + columnas de T3 (copia; el original no se modifica)
    # ------------------------------------------------------------------
    df_orig = pd.read_csv(RUTA_CSV, parse_dates=["timestamp"])
    n = len(df_orig)

    ls_t3 = t3.get("load_score")
    if isinstance(ls_t3, list) and len(ls_t3) == n:
        load_score = np.asarray(ls_t3, dtype=np.float64)
        fuente_load = "entrada/T3/resultados.json:load_score"
    else:  # respaldo: recalcular con la definición de T2/T3
        load_score = calcular_load_score(df_orig, ddof=0)
        fuente_load = "recomputado (z-scores ddof=0, pesos 0.5/0.3/0.2)"

    idx_true = t3.get("indices_requires_review_true")
    if isinstance(idx_true, list) and isinstance(ls_t3, list) and len(ls_t3) == n:
        requires_review = np.zeros(n, dtype=bool)
        requires_review[np.asarray(idx_true, dtype=int)] = True
        fuente_rr = "entrada/T3/resultados.json:indices_requires_review_true"
    else:
        requires_review = (load_score > UMBRAL_LOAD) | (
            df_orig["temperature_c"].to_numpy(dtype=np.float64) > UMBRAL_TEMP
        )
        fuente_rr = "recomputado ((load_score > 1.5) | (temperature_c > 80))"

    df_analizado = df_orig.copy()
    df_analizado["load_score"] = load_score
    df_analizado["requires_review"] = np.asarray(requires_review, dtype=bool)
    columnas_t3 = t3.get("columnas_finales", COLUMNAS_ESPERADAS)
    df_analizado = df_analizado[[c for c in columnas_t3 if c in df_analizado.columns]]

    # Coherencia con la definición de T3 (solo verificación, no ajuste)
    rr_criterio = (load_score > UMBRAL_LOAD) | (
        df_orig["temperature_c"].to_numpy(dtype=np.float64) > UMBRAL_TEMP
    )
    mismatches_rr = int(np.sum(rr_criterio != np.asarray(requires_review, dtype=bool)))
    maxdiff_ddof0 = float(np.max(np.abs(calcular_load_score(df_orig, 0) - load_score)))
    maxdiff_ddof1 = float(np.max(np.abs(calcular_load_score(df_orig, 1) - load_score)))
    n_rr_true = int(np.sum(requires_review))
    n_rr_true_t3 = t3.get("n_requires_review_true")
    rr_count_coincide = None if n_rr_true_t3 is None else bool(n_rr_true == int(n_rr_true_t3))

    fila23_t3 = t3.get("fila_23") or {}
    fila23_ok = None
    if fila23_t3 and n > 23:
        f = df_analizado.iloc[23]
        try:
            fila23_ok = bool(
                str(f["server"]) == str(fila23_t3.get("server"))
                and pd.Timestamp(f["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                == str(fila23_t3.get("timestamp"))
                and abs(float(f["load_score"]) - float(fila23_t3.get("load_score"))) <= 1e-9
                and abs(float(f["temperature_c"]) - float(fila23_t3.get("temperature_c"))) <= 1e-9
                and bool(f["requires_review"]) == bool(fila23_t3.get("requires_review"))
            )
        except Exception:
            fila23_ok = False

    # ------------------------------------------------------------------
    # 3) Guardar con PyArrow, sin el índice
    # ------------------------------------------------------------------
    os.makedirs("output", exist_ok=True)
    df_antes = df_analizado.copy()
    tabla_antes = pa.Table.from_pandas(df_antes, preserve_index=False)
    pq.write_table(tabla_antes, RUTA_PARQUET)

    # ------------------------------------------------------------------
    # 4) Volver a leer el archivo Parquet
    # ------------------------------------------------------------------
    tabla_despues = pq.read_table(RUTA_PARQUET)
    df_despues = tabla_despues.to_pandas()
    pf_meta = pq.ParquetFile(RUTA_PARQUET).metadata

    # ------------------------------------------------------------------
    # 5) Verificación del round trip
    # ------------------------------------------------------------------
    filas_ok = bool(
        tabla_antes.num_rows == tabla_despues.num_rows
        and len(df_antes) == len(df_despues)
    )

    cols_antes = list(df_antes.columns)
    cols_despues = list(df_despues.columns)
    sin_col_indice = "__index_level_0__" not in tabla_despues.column_names
    columnas_ok = bool(
        cols_antes == cols_despues
        and tabla_antes.num_columns == tabla_despues.num_columns
        and sin_col_indice
    )

    esquema_ok = bool(tabla_antes.schema.equals(tabla_despues.schema))
    esquema_ok_meta = bool(
        tabla_antes.schema.equals(tabla_despues.schema, check_metadata=True)
    )

    valores_por_columna = {}
    maxdiff_por_columna = {}
    for col in cols_antes:
        a, b = df_antes[col], df_despues[col]
        if pd.api.types.is_bool_dtype(a) or pd.api.types.is_datetime64_any_dtype(a):
            valores_por_columna[col] = bool(np.all(a.to_numpy() == b.to_numpy()))
        elif pd.api.types.is_numeric_dtype(a):
            an = a.to_numpy(dtype=np.float64)
            bn = b.to_numpy(dtype=np.float64)
            valores_por_columna[col] = bool(np.all((an == bn) | (np.isnan(an) & np.isnan(bn))))
            d = np.abs(an - bn)
            d = d[~np.isnan(d)]
            maxdiff_por_columna[col] = float(d.max()) if d.size else 0.0
        else:
            valores_por_columna[col] = bool(
                np.all(a.astype(str).to_numpy() == b.astype(str).to_numpy())
            )
    valores_ok = bool(all(valores_por_columna.values()))

    dtypes_antes = {c: str(t) for c, t in df_antes.dtypes.items()}
    dtypes_despues = {c: str(t) for c, t in df_despues.dtypes.items()}
    dtypes_ok = bool(dtypes_antes == dtypes_despues)
    tablas_iguales = bool(tabla_antes.equals(tabla_despues))
    dfs_iguales = bool(df_antes.equals(df_despues))

    # ------------------------------------------------------------------
    # 6) Veredicto (formato del enunciado) + resumen breve
    # ------------------------------------------------------------------
    print(f"Round trip verificado: {RUTA_PARQUET}")
    print(f"Filas: {filas_ok}")
    print(f"Columnas: {columnas_ok}")
    print(f"Esquema: {esquema_ok}")
    print(f"Valores: {valores_ok}")

    print("-" * 70)
    print(f"Archivo: {RUTA_PARQUET} ({os.path.getsize(RUTA_PARQUET)} bytes, "
          f"PyArrow {pa.__version__}, pandas {pd.__version__})")
    print(f"Filas antes/después: {len(df_antes)}/{len(df_despues)} | "
          f"Columnas antes/después: {len(cols_antes)}/{len(cols_despues)}")
    print(f"Guardado sin índice (sin '__index_level_0__'): {sin_col_indice}")
    print(f"Esquema parquet: {[(fl.name, str(fl.type)) for fl in tabla_despues.schema]}")
    print(f"Esquema idéntico incluso con metadatos de pandas: {esquema_ok_meta}")
    print(f"Tablas Arrow iguales: {tablas_iguales} | DataFrames pandas iguales: {dfs_iguales} "
          f"| dtypes iguales: {dtypes_ok}")
    print(f"Valores iguales por columna: {valores_por_columna}")
    print(f"Máx. diferencia absoluta por columna numérica: {maxdiff_por_columna}")
    print(f"Reconstrucción de T3 -> load_score: {fuente_load} | requires_review: {fuente_rr}")
    print(f"Coherencia con T3: requires_review True={n_rr_true} (coincide con T3: "
          f"{rr_count_coincide}), mismatches vs criterio={mismatches_rr}, "
          f"fila 23 coincide={fila23_ok}, "
          f"max|load_score-recomputado| ddof0={maxdiff_ddof0:.3e}, ddof1={maxdiff_ddof1:.3e}")

    # ------------------------------------------------------------------
    # 7) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T5_guardar_y_verificar_parquet",
        "archivo_parquet": RUTA_PARQUET,
        "libreria": "pyarrow",
        "version_pyarrow": str(pa.__version__),
        "version_pandas": str(pd.__version__),
        "guardado_sin_indice": bool(sin_col_indice),
        "reconstruccion_T3": {
            "fuente_load_score": fuente_load,
            "fuente_requires_review": fuente_rr,
            "n_filas": int(n),
            "columnas": cols_antes,
            "n_requires_review_true": n_rr_true,
            "n_requires_review_true_coincide_con_T3": rr_count_coincide,
            "mismatches_requires_review_vs_criterio": mismatches_rr,
            "max_abs_diff_load_score_vs_recomputado_ddof0": maxdiff_ddof0,
            "max_abs_diff_load_score_vs_recomputado_ddof1": maxdiff_ddof1,
            "fila_23_coincide_con_T3": fila23_ok,
        },
        "round_trip": {
            "filas_antes": int(len(df_antes)),
            "filas_despues": int(len(df_despues)),
            "filas_ok": filas_ok,
            "columnas_antes": cols_antes,
            "columnas_despues": cols_despues,
            "n_columnas_antes": int(len(cols_antes)),
            "n_columnas_despues": int(len(cols_despues)),
            "columnas_ok": columnas_ok,
            "esquema_antes": esquema_como_lista(tabla_antes.schema),
            "esquema_despues": esquema_como_lista(tabla_despues.schema),
            "esquema_ok": esquema_ok,
            "esquema_ok_con_metadatos": esquema_ok_meta,
            "valores_ok": valores_ok,
            "valores_iguales_por_columna": valores_por_columna,
            "max_abs_diff_por_columna_numerica": maxdiff_por_columna,
            "dtypes_antes": dtypes_antes,
            "dtypes_despues": dtypes_despues,
            "dtypes_ok": dtypes_ok,
            "tablas_arrow_iguales": tablas_iguales,
            "dataframes_pandas_iguales": dfs_iguales,
            "num_row_groups_parquet": int(pf_meta.num_row_groups),
            "formato_parquet": str(pf_meta.format_version),
            "tamano_archivo_bytes": int(os.path.getsize(RUTA_PARQUET)),
        },
        "verificacion_final": {
            "Filas": filas_ok,
            "Columnas": columnas_ok,
            "Esquema": esquema_ok,
            "Valores": valores_ok,
        },
        "exito": bool(filas_ok and columnas_ok and esquema_ok and valores_ok),
    }
    with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False, default=str)
    print(f"Resultados escritos en {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
