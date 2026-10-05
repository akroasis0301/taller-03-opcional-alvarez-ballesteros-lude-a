#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5 - Guardar y verificar el resultado (round trip Parquet con PyArrow).

1) Reconstruye el DataFrame analizado de T3 (copia del original + load_score
   + requires_review) usando los parámetros registrados en entrada/T3/resultados.json.
2) Lo guarda en output/server_analysis.parquet con PyArrow, SIN el índice.
3) Lo vuelve a leer y verifica: filas, columnas, esquema y valores.
4) Escribe resultados.json con todas las cifras y verificaciones.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RUTA_PARQUET = Path("output") / "server_analysis.parquet"
RUTA_RESULTADOS = Path("resultados.json")
RUTA_T3_JSON = Path("entrada") / "T3" / "resultados.json"
RUTA_CSV = Path("data") / "server_measurements.csv"

COLUMNAS_ESPERADAS = [
    "server", "timestamp", "gpu_utilization", "cpu_utilization",
    "memory_gb", "power_w", "temperature_c", "load_score", "requires_review",
]
UMBRAL_LOAD_SCORE = 1.5
UMBRAL_TEMP = 80.0


def cargar_t3():
    """Carga los resultados registrados por la subtarea T3 (si existen)."""
    if RUTA_T3_JSON.exists():
        try:
            with open(RUTA_T3_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def construir_dataframe_analizado():
    """Obtiene el DataFrame analizado de la T3.

    Camino principal: reconstrucción desde data/server_measurements.csv usando
    las medias, desviaciones y pesos registrados por T3 (mismas fórmulas).
    Camino alternativo: archivo que T3 haya dejado en entrada/T3/.
    """
    params = cargar_t3()

    if RUTA_CSV.exists():
        df = pd.read_csv(RUTA_CSV, parse_dates=["timestamp"])
        if params is not None:
            medias = [float(m) for m in params["means_usados"]]
            desvs = [float(s) for s in params["stds_usados"]]
            pesos = [float(p) for p in params["pesos_usados"]]
        else:
            cols_z = ("gpu_utilization", "cpu_utilization", "memory_gb")
            medias = [float(df[c].mean()) for c in cols_z]
            desvs = [float(df[c].std(ddof=0)) for c in cols_z]
            pesos = [0.50, 0.30, 0.20]

        z_gpu = (df["gpu_utilization"] - medias[0]) / desvs[0]
        z_cpu = (df["cpu_utilization"] - medias[1]) / desvs[1]
        z_mem = (df["memory_gb"] - medias[2]) / desvs[2]
        df["load_score"] = (pesos[0] * z_gpu + pesos[1] * z_cpu + pesos[2] * z_mem).astype("float64")
        df["requires_review"] = (df["load_score"] > UMBRAL_LOAD_SCORE) | (df["temperature_c"] > UMBRAL_TEMP)
        df["requires_review"] = df["requires_review"].astype(bool)

        orden = params["columnas_finales"] if (params and "columnas_finales" in params) else COLUMNAS_ESPERADAS
        df = df[orden]
        fuente = {
            "modo": "reconstruido_desde_data/server_measurements.csv_con_parametros_de_T3",
            "medias_usadas": medias,
            "desviaciones_usadas": desvs,
            "pesos_usados": pesos,
        }
        return df.reset_index(drop=True), fuente, params

    t3_dir = Path("entrada") / "T3"
    if t3_dir.is_dir():
        for p in sorted(list(t3_dir.glob("*.parquet")) + list(t3_dir.glob("*.csv"))):
            try:
                if p.suffix == ".parquet":
                    d = pd.read_parquet(p)
                else:
                    d = pd.read_csv(p, parse_dates=["timestamp"])
            except Exception:
                continue
            d = d.drop(columns=[c for c in d.columns if str(c).startswith("__index_level")], errors="ignore")
            if set(COLUMNAS_ESPERADAS).issubset(d.columns):
                d = d[COLUMNAS_ESPERADAS].copy()
                d["requires_review"] = d["requires_review"].astype(bool)
                if not pd.api.types.is_datetime64_any_dtype(d["timestamp"]):
                    d["timestamp"] = pd.to_datetime(d["timestamp"])
                fuente = {"modo": "cargado_desde_" + p.as_posix()}
                return d.reset_index(drop=True), fuente, params

    raise FileNotFoundError("No se pudo obtener el DataFrame analizado de la T3")


def columnas_iguales(sa, sb):
    """Compara dos Series: mismo dtype y mismos valores (NaN tratados como iguales)."""
    if sa.dtype != sb.dtype:
        return False
    va, vb = sa.to_numpy(), sb.to_numpy()
    if pd.api.types.is_float_dtype(sa.dtype):
        return bool(np.array_equal(va.astype("float64"), vb.astype("float64"), equal_nan=True))
    return bool(np.array_equal(va, vb))


def fila_a_dict(fila):
    """Convierte una fila (Series) en un dict con tipos nativos de Python."""
    out = {}
    for c, v in fila.items():
        if isinstance(v, pd.Timestamp):
            out[str(c)] = v.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(v, (bool, np.bool_)):
            out[str(c)] = bool(v)
        elif isinstance(v, (int, np.integer)):
            out[str(c)] = int(v)
        elif isinstance(v, (float, np.floating)):
            out[str(c)] = float(v)
        else:
            out[str(c)] = str(v)
    return out


def jsonable(x):
    """Convierte tipos de numpy/pandas a tipos nativos para JSON."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    return x


def main():
    # ------------------------------------------------------------------
    # 1) DataFrame analizado de T3 (ANTES del round trip)
    # ------------------------------------------------------------------
    df_antes, fuente, params = construir_dataframe_analizado()
    n_filas_antes, n_cols_antes = int(df_antes.shape[0]), int(df_antes.shape[1])
    cols_antes = list(df_antes.columns)

    # ------------------------------------------------------------------
    # 2) Guardar con PyArrow, SIN el índice
    # ------------------------------------------------------------------
    RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    tabla_antes = pa.Table.from_pandas(df_antes, preserve_index=False)
    pq.write_table(tabla_antes, RUTA_PARQUET)

    # ------------------------------------------------------------------
    # 3) Volver a leer y verificar el round trip (DESPUÉS)
    # ------------------------------------------------------------------
    tabla_despues = pq.read_table(RUTA_PARQUET)
    df_despues = tabla_despues.to_pandas()
    n_filas_despues, n_cols_despues = int(df_despues.shape[0]), int(df_despues.shape[1])
    cols_despues = list(df_despues.columns)

    # Filas
    filas_ok = n_filas_antes == n_filas_despues

    # Columnas
    columnas_ok = cols_antes == cols_despues

    # Esquema (tipos PyArrow y dtypes de pandas)
    esquema_arrow_igual = bool(tabla_antes.schema.equals(tabla_despues.schema))
    dtypes_antes = {c: str(t) for c, t in df_antes.dtypes.items()}
    dtypes_despues = {c: str(t) for c, t in df_despues.dtypes.items()}
    dtypes_iguales = dtypes_antes == dtypes_despues
    esquema_ok = bool(esquema_arrow_igual and dtypes_iguales)

    # Valores (columna por columna)
    a = df_antes.reset_index(drop=True)
    b = df_despues.reset_index(drop=True)
    valores_ok = True
    col_divergente = None
    max_diff = 0.0
    for col in a.columns:
        if not columnas_iguales(a[col], b[col]):
            valores_ok, col_divergente = False, col
            break
        if pd.api.types.is_float_dtype(a[col].dtype):
            va = a[col].to_numpy(dtype="float64")
            vb = b[col].to_numpy(dtype="float64")
            if va.size:
                max_diff = max(max_diff, float(np.nanmax(np.abs(va - vb))))
    valores_pandas_equals = bool(a.equals(b))
    tablas_pyarrow_iguales = bool(tabla_antes.equals(tabla_despues))

    # El índice no debe haberse guardado
    campos_schema = [str(f.name) for f in tabla_despues.schema]
    indice_no_guardado = not any(n.startswith("__index_level") for n in campos_schema)

    # Extra: lectura con pd.read_parquet
    try:
        df_pd = pd.read_parquet(RUTA_PARQUET).reset_index(drop=True)
        lectura_pd_ok = bool(
            df_pd.shape == a.shape
            and list(df_pd.columns) == cols_antes
            and all(columnas_iguales(a[c], df_pd[c]) for c in cols_antes)
        )
    except Exception:
        lectura_pd_ok = None

    # ------------------------------------------------------------------
    # 4) Comprobaciones cruzadas con T3
    # ------------------------------------------------------------------
    fila23_antes = fila_a_dict(a.loc[23]) if 23 in a.index else None
    fila23_despues = fila_a_dict(b.loc[23]) if 23 in b.index else None

    coincide_fila23_t3 = None
    if params and isinstance(params.get("fila_23"), dict) and fila23_antes is not None:
        ref = params["fila_23"]
        try:
            coincide_fila23_t3 = bool(
                fila23_antes["server"] == ref["server"]
                and pd.Timestamp(fila23_antes["timestamp"]) == pd.Timestamp(ref["timestamp"])
                and abs(fila23_antes["load_score"] - float(ref["load_score"])) <= 1e-9
                and abs(fila23_antes["temperature_c"] - float(ref["temperature_c"])) <= 1e-9
                and bool(fila23_antes["requires_review"]) == bool(ref["requires_review"])
            )
        except Exception:
            coincide_fila23_t3 = None

    n_review = int(a["requires_review"].sum())
    coincide_conteo_t3 = None
    if params and isinstance(params.get("conteos"), dict):
        try:
            coincide_conteo_t3 = bool(n_review == int(params["conteos"]["n_requires_review_true"]))
        except Exception:
            coincide_conteo_t3 = None

    coincide_forma_t3 = None
    if params and "forma_copia" in params:
        coincide_forma_t3 = bool([n_filas_antes, n_cols_antes] == list(params["forma_copia"]))

    # ------------------------------------------------------------------
    # 5) resultados.json
    # ------------------------------------------------------------------
    verificaciones = {
        "Filas": bool(filas_ok),
        "Columnas": bool(columnas_ok),
        "Esquema": bool(esquema_ok),
        "Valores": bool(valores_ok),
    }
    round_trip_ok = all(verificaciones.values())
    criterio = bool(RUTA_PARQUET.exists() and round_trip_ok)

    resultados = {
        "subtarea": "T5 - Guardar y verificar el resultado (round trip Parquet con PyArrow)",
        "parquet_escrito": RUTA_PARQUET.as_posix(),
        "parquet_existe": bool(RUTA_PARQUET.exists()),
        "tamano_bytes": int(RUTA_PARQUET.stat().st_size) if RUTA_PARQUET.exists() else None,
        "metodo_guardado": "pyarrow: pa.Table.from_pandas(df, preserve_index=False) + pq.write_table",
        "indice_no_guardado": bool(indice_no_guardado),
        "fuente_dataframe": fuente,
        "parametros_T3_cargados": bool(params),
        "forma_antes": [n_filas_antes, n_cols_antes],
        "forma_despues": [n_filas_despues, n_cols_despues],
        "n_filas_antes": n_filas_antes,
        "n_filas_despues": n_filas_despues,
        "n_columnas_antes": n_cols_antes,
        "n_columnas_despues": n_cols_despues,
        "columnas_antes": cols_antes,
        "columnas_despues": cols_despues,
        "dtypes_antes": dtypes_antes,
        "dtypes_despues": dtypes_despues,
        "esquema_antes": str(tabla_antes.schema),
        "esquema_despues": str(tabla_despues.schema),
        "esquema_arrow_igual": esquema_arrow_igual,
        "dtypes_iguales": bool(dtypes_iguales),
        "campos_schema_despues": campos_schema,
        "max_diferencia_absoluta_floats": max_diff,
        "valores_iguales_pandas_equals": valores_pandas_equals,
        "tablas_pyarrow_iguales": tablas_pyarrow_iguales,
        "columna_con_valores_divergentes": col_divergente,
        "lectura_pd_read_parquet_coincide": lectura_pd_ok,
        "fila_23_antes": fila23_antes,
        "fila_23_despues": fila23_despues,
        "coincide_con_T3_fila_23": coincide_fila23_t3,
        "n_requires_review_true": n_review,
        "coincide_con_T3_conteo_requires_review": coincide_conteo_t3,
        "coincide_forma_con_T3": coincide_forma_t3,
        "verificaciones_round_trip": verificaciones,
        "filas_iguales": bool(filas_ok),
        "columnas_iguales": bool(columnas_ok),
        "esquema_igual": bool(esquema_ok),
        "valores_iguales": bool(valores_ok),
        "round_trip_ok": bool(round_trip_ok),
        "criterio_exito_cumplido": criterio,
    }

    with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(jsonable(resultados), f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Resumen
    # ------------------------------------------------------------------
    print("Round trip verificado:", RUTA_PARQUET.as_posix())
    print("Filas:", verificaciones["Filas"])
    print("Columnas:", verificaciones["Columnas"])
    print("Esquema:", verificaciones["Esquema"])
    print("Valores:", verificaciones["Valores"])
    print("-" * 55)
    print(f"DataFrame analizado: {n_filas_antes} filas x {n_cols_antes} columnas")
    print(f"Parquet: {RUTA_PARQUET.as_posix()} ({resultados['tamano_bytes']} bytes) | indice guardado: {not indice_no_guardado}")
    print(f"Max. diferencia absoluta en floats tras el round trip: {max_diff}")
    print(f"Filas con requires_review=True: {n_review}")
    print(f"Criterio de exito cumplido: {criterio}")


if __name__ == "__main__":
    main()
