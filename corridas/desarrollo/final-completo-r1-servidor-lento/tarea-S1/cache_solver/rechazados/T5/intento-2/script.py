#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5 - Guardar y verificar el resultado (round trip Parquet con PyArrow).

Pasos:
1) Cargar data/server_measurements.csv (timestamp con parse dates).
2) Construir el DataFrame analizado de T3:
   - load_score = zscores @ [0.50, 0.30, 0.20], con z-scores vectorizados
     (x - media) / std por columna (gpu_utilization, cpu_utilization,
     memory_gb), evitando la division por cero (std == 0 -> z = 0).
     El ddof se valida contra la fila 23 esperada (load_score = 3.136218).
   - requires_review (bool) = (load_score > 1.5) | (temperature_c > 80).
   - Validacion de la fila 23 contra el resultado esperado de T3.
3) Guardar en output/server_analysis.parquet con PyArrow, SIN el indice.
4) Volver a leer el archivo y verificar el round trip:
   filas, columnas, esquema y valores.
5) Imprimir las cuatro verificaciones y escribir resultados.json.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RUTA_CSV = Path("data/server_measurements.csv")
RUTA_PARQUET = Path("output/server_analysis.parquet")
RUTA_T3 = Path("entrada/T3/resultados.json")

COLUMNAS_BASE = ["server", "timestamp", "gpu_utilization", "cpu_utilization",
                 "memory_gb", "power_w", "temperature_c"]
COLUMNAS_T3 = COLUMNAS_BASE + ["load_score", "requires_review"]
COLS_ZSCORE = ["gpu_utilization", "cpu_utilization", "memory_gb"]
PESOS = np.array([0.50, 0.30, 0.20])
TOL = 1e-6

# Ancla de validacion (enunciado T3, fila 23)
REF_ENUNCIADO_F23 = {
    "server": "AI-SRV-01",
    "timestamp": "2026-08-24 09:55:00",
    "load_score": 3.136218,
    "temperature_c": 66.97,
    "requires_review": True,
}


# ---------------------------------------------------------------------------
# Referencia de validacion (fila 23 de T3)
# ---------------------------------------------------------------------------
def obtener_referencia():
    ref = dict(REF_ENUNCIADO_F23)
    ref["fuente"] = "enunciado T3 (fila 23, load_score redondeado a 6 decimales)"
    if RUTA_T3.exists():
        try:
            r3 = json.loads(RUTA_T3.read_text(encoding="utf-8"))
            f23 = r3.get("fila_23", {})
            if "load_score" in f23:
                ref["load_score"] = float(f23["load_score"])
                ref["fuente"] = "entrada/T3/resultados.json (precision completa)"
            for k in ("server", "timestamp", "temperature_c", "requires_review"):
                if k in f23:
                    ref[k] = f23[k]
        except Exception:
            pass
    return ref


# ---------------------------------------------------------------------------
# load_score = zscores @ [0.50, 0.30, 0.20]  (vectorizado, sin div/0)
# ---------------------------------------------------------------------------
def calcular_load_score(df, cols, ddof):
    """Z-scores por columna y combinacion lineal con los pesos de T2."""
    sub = df[list(cols)].astype("float64")
    media = sub.mean()
    std = sub.std(ddof=ddof)
    valida = std > 0                      # evita division por cero
    std_segura = std.where(valida, 1.0)
    z = (sub - media) / std_segura
    z = z.where(valida, 0.0)              # columna constante -> z = 0
    load_score = z.to_numpy() @ PESOS
    return load_score, media, std


def seleccionar_load_score(df, ref):
    """Prueba (gpu, cpu, memory) con ddof de pandas (1) y poblacional (0);
    valida la fila 23 contra la referencia. Respaldo: permutaciones de orden."""
    objetivo = float(ref["load_score"])
    ordenes = [tuple(COLS_ZSCORE)]
    ordenes += [p for p in itertools.permutations(COLS_ZSCORE)
                if list(p) != COLS_ZSCORE]
    candidatos = []
    for cols in ordenes:
        for ddof in (1, 0):
            ls, media, std = calcular_load_score(df, cols, ddof)
            candidatos.append((cols, ddof, ls, media, std))
    for cols, ddof, ls, media, std in candidatos:
        if abs(float(ls[23]) - objetivo) <= TOL:
            return ls, list(cols), ddof, True, media, std
    cols, ddof, ls, media, std = min(
        candidatos, key=lambda c: abs(float(c[2][23]) - objetivo))
    return ls, list(cols), ddof, False, media, std


# ---------------------------------------------------------------------------
# Datos base y DataFrame de T3
# ---------------------------------------------------------------------------
def cargar_datos_base():
    if RUTA_CSV.exists():
        return pd.read_csv(RUTA_CSV, parse_dates=["timestamp"]), str(RUTA_CSV)
    candidatos = []
    if Path("entrada/T3").is_dir():
        candidatos += sorted(Path("entrada/T3").glob("*.parquet"))
    candidatos.append(RUTA_PARQUET)
    for cand in candidatos:
        if not cand.exists():
            continue
        try:
            tmp = pd.read_parquet(cand)
        except Exception:
            continue
        if set(COLUMNAS_BASE).issubset(tmp.columns):
            return tmp[COLUMNAS_BASE].copy(), str(cand)
    raise FileNotFoundError("No se encontro data/server_measurements.csv ni un "
                            "respaldo con las columnas base.")


def construir_df_t3(ref):
    df_base, fuente = cargar_datos_base()
    df = df_base.copy()
    df["server"] = df["server"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for c in ("gpu_utilization", "cpu_utilization", "memory_gb",
              "power_w", "temperature_c"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    ls, cols, ddof, validado, media, std = seleccionar_load_score(df, ref)
    df["load_score"] = np.asarray(ls, dtype="float64")
    df["requires_review"] = ((df["load_score"] > 1.5) |
                             (df["temperature_c"] > 80)).astype(bool)
    df = df[COLUMNAS_T3].reset_index(drop=True)
    return df, fuente, cols, ddof, validado, media, std


def validar_fila23(df, ref):
    f = df.iloc[23]
    checks = {
        "server": str(f["server"]) == str(ref["server"]),
        "timestamp": pd.Timestamp(f["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                     == str(ref["timestamp"]),
        "load_score": abs(float(f["load_score"]) - float(ref["load_score"])) <= TOL,
        "temperature_c": abs(float(f["temperature_c"])
                             - float(ref["temperature_c"])) <= 1e-9,
        "requires_review": bool(f["requires_review"]) == bool(ref["requires_review"]),
    }
    checks["fila_23_coincide_con_esperado"] = bool(all(checks.values()))
    return checks


# ---------------------------------------------------------------------------
# Comparacion de valores celda a celda
# ---------------------------------------------------------------------------
def comparar_valores(df1, df2):
    if df1.shape != df2.shape:
        return False, "forma"
    for col in df1.columns:
        s1, s2 = df1[col], df2[col]
        if pd.api.types.is_datetime64_any_dtype(s1) and \
           pd.api.types.is_datetime64_any_dtype(s2):
            if pd.to_datetime(s1).equals(pd.to_datetime(s2)):
                continue
            return False, col
        if pd.api.types.is_bool_dtype(s1) and pd.api.types.is_bool_dtype(s2):
            if s1.equals(s2):
                continue
            return False, col
        if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
            a = s1.to_numpy(dtype="float64")
            b = s2.to_numpy(dtype="float64")
            if np.array_equal(a, b, equal_nan=True):
                continue
            return False, col
        s1f = s1.fillna("__NaN__").astype(str).reset_index(drop=True)
        s2f = s2.fillna("__NaN__").astype(str).reset_index(drop=True)
        if s1f.equals(s2f):
            continue
        return False, col
    return True, None


def esquema_a_dict(schema):
    return {campo.name: str(campo.type) for campo in schema}


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main():
    ref = obtener_referencia()
    df_antes, fuente, cols_z, ddof, validado, media, std = construir_df_t3(ref)

    # Validacion contra el resultado esperado de T3 (fila 23)
    checks_f23 = validar_fila23(df_antes, ref)

    # Conteos de la regla de revision
    n_review = int(df_antes["requires_review"].sum())
    n_ls = int((df_antes["load_score"] > 1.5).sum())
    n_temp = int((df_antes["temperature_c"] > 80).sum())
    conteos_t3 = None
    if RUTA_T3.exists():
        try:
            r3 = json.loads(RUTA_T3.read_text(encoding="utf-8"))
            conteos_t3 = {
                "review_count_total_coincide":
                    r3.get("review_count_total") == n_review,
                "n_load_score_mayor_1_5_coincide":
                    r3.get("n_load_score_mayor_1_5") == n_ls,
                "n_temperature_c_mayor_80_coincide":
                    r3.get("n_temperature_c_mayor_80") == n_temp,
            }
        except Exception:
            conteos_t3 = None

    # ------------------------------------------------------------------
    # Guardar con PyArrow, SIN el indice
    # ------------------------------------------------------------------
    RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    tabla = pa.Table.from_pandas(df_antes, preserve_index=False)
    pq.write_table(tabla, RUTA_PARQUET.as_posix())

    # ------------------------------------------------------------------
    # Volver a leer el archivo
    # ------------------------------------------------------------------
    tabla_leida = pq.read_table(RUTA_PARQUET.as_posix())
    df_despues = tabla_leida.to_pandas()

    # ------------------------------------------------------------------
    # Verificaciones del round trip
    # ------------------------------------------------------------------
    filas_ok = bool(len(df_antes) == len(df_despues))
    columnas_ok = bool(list(df_antes.columns) == list(df_despues.columns))
    esquema_ok = bool(tabla.schema.equals(tabla_leida.schema, check_metadata=False))
    valores_ok, col_mala = comparar_valores(df_antes, df_despues)
    valores_ok = bool(valores_ok)

    print(f"Round trip verificado: {RUTA_PARQUET.as_posix()}")
    print(f"Filas: {filas_ok}")
    print(f"Columnas: {columnas_ok}")
    print(f"Esquema: {esquema_ok}")
    print(f"Valores: {valores_ok}")

    # ------------------------------------------------------------------
    # Extras y resultados.json
    # ------------------------------------------------------------------
    round_trip_ok = bool(filas_ok and columnas_ok and esquema_ok and valores_ok)
    sin_indice = not any("__index_level" in n for n in tabla_leida.schema.names)
    dtypes_iguales = bool(all(str(df_antes[c].dtype) == str(df_despues[c].dtype)
                              for c in COLUMNAS_T3))

    f23 = df_antes.iloc[23]
    fila_23 = {
        "indice": 23,
        "server": str(f23["server"]),
        "timestamp": str(f23["timestamp"]),
        "load_score": float(f23["load_score"]),
        "load_score_redondeado_6_decimales": round(float(f23["load_score"]), 6),
        "temperature_c": float(f23["temperature_c"]),
        "requires_review": bool(f23["requires_review"]),
    }

    resultados = {
        "subtarea": "T5_guardar_y_verificar_round_trip_parquet",
        "archivo_parquet": RUTA_PARQUET.as_posix(),
        "motor": "pyarrow",
        "version_pyarrow": str(pa.__version__),
        "version_pandas": str(pd.__version__),
        "fuente_datos": fuente,
        "formula_load_score": ("load_score = zscores @ [0.50, 0.30, 0.20]; "
                               "zscores = (x - media) / std por columna "
                               "(std == 0 -> z = 0, division por cero evitada)"),
        "columnas_zscore": list(cols_z),
        "pesos": [0.50, 0.30, 0.20],
        "ddof_usado": int(ddof),
        "ddof_validado_con_referencia_fila23": bool(validado),
        "referencia_fila23_fuente": ref["fuente"],
        "medias": {c: float(v) for c, v in media.items()},
        "desviaciones_estandar": {c: float(v) for c, v in std.items()},
        "n_filas": int(len(df_antes)),
        "columnas": list(df_antes.columns),
        "requires_review_dtype": str(df_antes["requires_review"].dtype),
        "regla_revision": "(load_score > 1.5) | (temperature_c > 80)",
        "n_requires_review": n_review,
        "n_load_score_mayor_1_5": n_ls,
        "n_temperature_c_mayor_80": n_temp,
        "conteos_vs_T3": conteos_t3,
        "fila_23": fila_23,
        "verificacion_fila_23": checks_f23,
        "guardado_sin_indice": bool(sin_indice),
        "preserve_index": False,
        "n_filas_antes": int(len(df_antes)),
        "n_filas_despues": int(len(df_despues)),
        "filas_iguales": filas_ok,
        "columnas_antes": list(df_antes.columns),
        "columnas_despues": list(df_despues.columns),
        "columnas_iguales": columnas_ok,
        "esquema_antes": esquema_a_dict(tabla.schema),
        "esquema_despues": esquema_a_dict(tabla_leida.schema),
        "esquema_iguales": esquema_ok,
        "dtypes_antes": {c: str(t) for c, t in df_antes.dtypes.items()},
        "dtypes_despues": {c: str(t) for c, t in df_despues.dtypes.items()},
        "dtypes_iguales": dtypes_iguales,
        "valores_iguales": valores_ok,
        "columna_con_diferencias": col_mala,
        "n_celdas_comparadas": int(df_antes.size),
        "round_trip_verificado": round_trip_ok,
        "tamano_archivo_bytes": int(RUTA_PARQUET.stat().st_size),
    }
    with open("resultados.json", "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, indent=2, ensure_ascii=False, default=str)

    # ------------------------------------------------------------------
    # Resumen breve
    # ------------------------------------------------------------------
    print("\nResumen T5")
    print(f"  Parquet        : {RUTA_PARQUET.as_posix()} "
          f"({RUTA_PARQUET.stat().st_size} bytes, PyArrow, sin indice)")
    print(f"  DataFrame      : {len(df_antes)} filas x {df_antes.shape[1]} columnas")
    print(f"  load_score     : zscores({', '.join(cols_z)}) @ [0.50, 0.30, 0.20], "
          f"ddof={ddof}, validado fila 23 = {validado}")
    print(f"  Fila 23        : server={fila_23['server']}, "
          f"load_score={fila_23['load_score_redondeado_6_decimales']}, "
          f"temperature_c={fila_23['temperature_c']}, "
          f"requires_review={fila_23['requires_review']} -> "
          f"coincide={checks_f23['fila_23_coincide_con_esperado']}")
    print(f"  Revision       : {n_review} filas (load_score>1.5: {n_ls}, "
          f"temperature_c>80: {n_temp})")
    print(f"  Verificacion   : filas={filas_ok}, columnas={columnas_ok}, "
          f"esquema={esquema_ok}, valores={valores_ok}")
    print(f"  Round trip     : {'VERIFICADO' if round_trip_ok else 'FALLO'}")
    print("  resultados.json escrito.")


if __name__ == "__main__":
    main()
