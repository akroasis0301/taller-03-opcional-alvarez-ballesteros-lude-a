#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5 - Guardar y verificar el resultado (round trip Parquet con PyArrow).

Pasos:
1) Obtener el DataFrame analizado de T3 (se carga desde entrada/T3/ si está disponible;
   si no, se reconstruye a partir de data/server_measurements.csv con los parámetros
   documentados en entrada/T3/resultados.json: pesos, medias y desviaciones de T2).
2) Guardarlo en output/server_analysis.parquet con PyArrow, SIN el índice.
3) Volver a leerlo y verificar, como mínimo:
   filas antes == después, columnas antes == después,
   esquema antes == después y valores iguales.
4) Imprimir las cuatro verificaciones y escribir resultados.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_PATH = Path("output") / "server_analysis.parquet"
T3_DIR = Path("entrada") / "T3"
T3_JSON = T3_DIR / "resultados.json"
DATA_CSV = Path("data") / "server_measurements.csv"

BASE_COLS = ["server", "timestamp", "gpu_utilization", "cpu_utilization",
             "memory_gb", "power_w", "temperature_c"]
REQ_COLS = BASE_COLS + ["load_score", "requires_review"]


def load_t3_meta():
    """Carga los resultados de T3 (parámetros y estadísticas de referencia)."""
    if T3_JSON.exists():
        try:
            return json.loads(T3_JSON.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def t3_consistency(df, meta):
    """Comprueba que el DataFrame coincide con lo documentado por T3."""
    checks = {}
    if "n_rows" in meta:
        checks["n_rows_match"] = bool(len(df) == int(meta["n_rows"]))
    if "load_score_max" in meta:
        checks["load_score_max_match"] = bool(np.isclose(
            float(df["load_score"].max()), float(meta["load_score_max"]),
            rtol=0.0, atol=1e-9))
    if "n_requires_review" in meta:
        checks["n_requires_review_match"] = bool(
            int(df["requires_review"].sum()) == int(meta["n_requires_review"]))
    rule = (df["load_score"] > 1.5) | (df["temperature_c"] > 80)
    checks["review_rule_consistent"] = bool((rule == df["requires_review"]).all())
    checks["requires_review_is_bool"] = bool(df["requires_review"].dtype == bool)
    flagged = meta.get("flagged_rows") or []
    if flagged:
        r0 = flagged[0]
        row = df.iloc[int(r0["index"])]
        checks["flagged_row0_match"] = bool(
            row["server"] == r0["server"]
            and pd.Timestamp(row["timestamp"]) == pd.Timestamp(r0["timestamp"])
            and np.isclose(float(row["load_score"]), float(r0["load_score"]),
                           rtol=0.0, atol=1e-9)
            and float(row["temperature_c"]) == float(r0["temperature_c"])
            and bool(row["requires_review"]) == bool(r0["requires_review"]))
    return checks


def try_load_from_t3(meta):
    """Intenta cargar el DataFrame analizado de T3 desde entrada/T3/."""
    if not T3_DIR.exists():
        return None, None
    for pattern in ("*.parquet", "*.csv"):
        for p in sorted(T3_DIR.glob(pattern)):
            try:
                if p.suffix == ".parquet":
                    cand = pd.read_parquet(p)
                else:
                    cand = pd.read_csv(p, parse_dates=["timestamp"])
            except Exception:
                continue
            cand = cand.reset_index(drop=True)
            drop = [c for c in cand.columns if str(c).startswith("__index_level")]
            if drop:
                cand = cand.drop(columns=drop)
            if not set(REQ_COLS).issubset(cand.columns):
                continue
            cand = cand[REQ_COLS]
            if not all(t3_consistency(cand, meta).values()):
                continue
            return cand, p.as_posix()
    return None, None


def rebuild_from_source(meta):
    """Reconstruye el DataFrame analizado de T3 desde el CSV original, usando los
    pesos, medias y desviaciones (ddof=0) documentados en resultados.json de T3."""
    df = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    feats = list(meta.get("feature_names") or
                 ["gpu_utilization", "cpu_utilization", "memory_gb"])
    weights = list(meta.get("weights") or [0.5, 0.3, 0.2])
    ddof = int(meta.get("std_ddof", 0))
    if meta.get("means_used") and meta.get("stds_used"):
        means = [float(m) for m in meta["means_used"]]
        stds = [float(s) for s in meta["stds_used"]]
    else:
        means = [float(df[f].mean()) for f in feats]
        stds = [float(df[f].std(ddof=ddof)) for f in feats]
    load_score = np.zeros(len(df), dtype=float)
    for f, w, m, s in zip(feats, weights, means, stds):
        load_score = load_score + float(w) * ((df[f].to_numpy(dtype=float) - m) / s)
    df = df.copy()
    df["load_score"] = load_score
    df["requires_review"] = ((df["load_score"] > 1.5) |
                             (df["temperature_c"] > 80)).astype(bool)
    return df


def valores_iguales_leniente(a, b):
    """Comparación valor a valor, independiente de los dtypes."""
    if len(a) != len(b) or list(a.columns) != list(b.columns):
        return False
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if pd.api.types.is_float_dtype(x.dtype) and pd.api.types.is_float_dtype(y.dtype):
            if not np.array_equal(x, y, equal_nan=True):
                return False
        elif not np.array_equal(x, y):
            return False
    return True


def main():
    meta = load_t3_meta()

    # --- DataFrame analizado de T3 ---------------------------------------
    df, src = try_load_from_t3(meta)
    if df is None:
        df = rebuild_from_source(meta)
        src = ("reconstruido desde data/server_measurements.csv "
               "(parámetros de entrada/T3/resultados.json)")
    df = df[REQ_COLS].reset_index(drop=True)
    consistency = t3_consistency(df, meta)

    # --- Guardar con PyArrow, sin el índice -------------------------------
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, PARQUET_PATH, version="2.6")

    # --- Volver a leer -----------------------------------------------------
    table_back = pq.read_table(PARQUET_PATH)
    df_back = table_back.to_pandas().reset_index(drop=True)

    # --- Verificaciones mínimas del round trip ------------------------------
    a = df.reset_index(drop=True)
    b = df_back.reset_index(drop=True)

    filas_ok = bool(len(a) == len(b))
    columnas_ok = bool(list(a.columns) == list(b.columns))
    esquema_ok = bool(table.schema.equals(table_back.schema))
    valores_strict = bool(a.equals(b))
    valores_ok = bool(valores_strict or valores_iguales_leniente(a, b))
    dtypes_ok = bool(all(str(a[c].dtype) == str(b[c].dtype) for c in a.columns))
    sin_indice = bool(not any(str(f.name).startswith("__index_level")
                              for f in table_back.schema))

    # --- Salida exigida por el enunciado ------------------------------------
    print(f"Round trip verificado: {PARQUET_PATH.as_posix()}")
    print(f"Filas: {filas_ok}")
    print(f"Columnas: {columnas_ok}")
    print(f"Esquema: {esquema_ok}")
    print(f"Valores: {valores_ok}")

    # --- resultados.json (contrato de la subtarea) ---------------------------
    resultados = {
        "task": "T5_guardar_y_verificar_resultado",
        "parquet_path": PARQUET_PATH.as_posix(),
        "file_exists": bool(PARQUET_PATH.exists()),
        "engine": "pyarrow",
        "pyarrow_write_options": {"preserve_index": False, "version": "2.6"},
        "saved_without_index": sin_indice,
        "dataframe_source": src,
        "n_rows_before": int(len(a)),
        "n_rows_after": int(len(b)),
        "n_columns_before": int(a.shape[1]),
        "n_columns_after": int(b.shape[1]),
        "columns_before": list(a.columns),
        "columns_after": list(b.columns),
        "schema_before": [f"{f.name}: {f.type}" for f in table.schema],
        "schema_after": [f"{f.name}: {f.type}" for f in table_back.schema],
        "dtypes_before": {c: str(t) for c, t in a.dtypes.items()},
        "dtypes_after": {c: str(t) for c, t in b.dtypes.items()},
        "verificaciones": {
            "Filas": filas_ok,
            "Columnas": columnas_ok,
            "Esquema": esquema_ok,
            "Valores": valores_ok,
        },
        "filas_iguales": filas_ok,
        "columnas_iguales": columnas_ok,
        "esquema_igual": esquema_ok,
        "valores_iguales": valores_ok,
        "valores_iguales_estricto_pandas": valores_strict,
        "dtypes_iguales": dtypes_ok,
        "round_trip_verified": bool(filas_ok and columnas_ok and esquema_ok and valores_ok),
        "n_requires_review": int(a["requires_review"].sum()),
        "load_score_min": float(a["load_score"].min()),
        "load_score_max": float(a["load_score"].max()),
        "t3_consistency_checks": consistency,
        "file_size_bytes": int(PARQUET_PATH.stat().st_size),
    }
    Path("resultados.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Resumen breve -------------------------------------------------------
    print("-" * 60)
    print(f"DataFrame: {len(a)} filas x {a.shape[1]} columnas | origen: {src}")
    print(f"Parquet: {PARQUET_PATH.as_posix()} "
          f"({resultados['file_size_bytes']} bytes, sin índice: {sin_indice})")
    print(f"Filas antes/después: {len(a)}/{len(b)} | "
          f"Columnas antes/después: {a.shape[1]}/{b.shape[1]}")
    print(f"Esquema antes:     {resultados['schema_before']}")
    print(f"Esquema después:   {resultados['schema_after']}")
    print(f"Filas marcadas requires_review: {resultados['n_requires_review']}")
    print(f"Consistencia con T3: "
          f"{all(consistency.values()) if consistency else 'sin datos de referencia'}")
    print("resultados.json escrito.")


if __name__ == "__main__":
    main()
