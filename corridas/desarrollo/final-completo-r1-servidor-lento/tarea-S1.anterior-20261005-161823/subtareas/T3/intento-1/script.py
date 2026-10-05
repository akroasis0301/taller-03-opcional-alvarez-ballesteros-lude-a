#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 - Identificar observaciones para revisión.

Crea una COPIA del DataFrame original (data/server_measurements.csv, con
timestamp parseado como fecha-hora, igual que en T1) y le agrega:
  * load_score      -> indicador de carga calculado en T2
                       (zscores @ [0.50, 0.30, 0.20] sobre gpu_utilization,
                       cpu_utilization, memory_gb; ddof=0, vectorizado)
  * requires_review -> bool; True si load_score > 1.5 o temperature_c > 80
Muestra la fila 23. El DataFrame original permanece sin cambios.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RUTA_CSV = Path("data/server_measurements.csv")
RUTA_T2 = Path("entrada/T2/resultados.json")
RUTA_PARQUET = Path("output/server_analysis.parquet")
RUTA_CARPETA_T3 = Path("entrada/T3")

PESOS = np.array([0.50, 0.30, 0.20], dtype=float)
UMBRAL_LOAD_SCORE = 1.5
UMBRAL_TEMPERATURA = 80.0
FILA_OBJETIVO = 23

# ----------------------------------------------------------------------------
# 1) DataFrame original (no se modifica en ningún momento)
# ----------------------------------------------------------------------------
df_original = pd.read_csv(RUTA_CSV, parse_dates=["timestamp"])
snapshot_original = df_original.copy(deep=True)  # respaldo para verificación

# ----------------------------------------------------------------------------
# 2) load_score: se toma de T2 (entrada/T2/resultados.json); si no está
#    disponible, se recalcula con el mismo procedimiento vectorizado de T2.
# ----------------------------------------------------------------------------
X = df_original[["gpu_utilization", "cpu_utilization", "memory_gb"]].to_numpy(dtype=float)
medias = X.mean(axis=0)
stds = X.std(axis=0, ddof=0)
stds_seguras = np.where(stds == 0.0, 1.0, stds)   # evita división por cero
zscores = (X - medias) / stds_seguras             # forma (300, 3)
load_score_recalculado = zscores @ PESOS          # vectorizado, forma (300,)

load_score_t2 = None
if RUTA_T2.exists():
    try:
        with open(RUTA_T2, "r", encoding="utf-8") as f:
            t2 = json.load(f)
        candidato = t2.get("load_score")
        if isinstance(candidato, list) and len(candidato) == len(df_original):
            load_score_t2 = np.asarray(candidato, dtype=float)
    except (OSError, ValueError):
        load_score_t2 = None

if load_score_t2 is not None:
    load_score = load_score_t2
    fuente_load_score = "entrada/T2/resultados.json"
else:
    load_score = load_score_recalculado
    fuente_load_score = "recalculado con el procedimiento de T2"

coincide_con_recalculo = bool(
    np.allclose(load_score, load_score_recalculado, rtol=0.0, atol=1e-12)
)

# ----------------------------------------------------------------------------
# 3) Copia del original + columnas nuevas (el original no se toca)
# ----------------------------------------------------------------------------
df_nuevo = df_original.copy(deep=True)
df_nuevo["load_score"] = load_score
df_nuevo["requires_review"] = (
    (df_nuevo["load_score"] > UMBRAL_LOAD_SCORE)
    | (df_nuevo["temperature_c"] > UMBRAL_TEMPERATURA)
).astype(bool)

# ----------------------------------------------------------------------------
# 4) Fila 23
# ----------------------------------------------------------------------------
fila23 = df_nuevo.iloc[FILA_OBJETIVO]
fila_23 = {
    "server": str(fila23["server"]),
    "timestamp": pd.Timestamp(fila23["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
    "load_score": float(fila23["load_score"]),
    "load_score_6dec": round(float(fila23["load_score"]), 6),
    "temperature_c": float(fila23["temperature_c"]),
    "requires_review": bool(fila23["requires_review"]),
}

fila_23_completa = {}
for col in df_nuevo.columns:
    val = fila23[col]
    if isinstance(val, pd.Timestamp):
        fila_23_completa[col] = val.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(val, (bool, np.bool_)):
        fila_23_completa[col] = bool(val)
    elif isinstance(val, np.integer):
        fila_23_completa[col] = int(val)
    elif isinstance(val, (float, np.floating)):
        fila_23_completa[col] = float(val)
    else:
        fila_23_completa[col] = str(val)

# ----------------------------------------------------------------------------
# 5) Verificaciones contra el criterio de éxito
# ----------------------------------------------------------------------------
esperado = {
    "server": "AI-SRV-01",
    "timestamp": "2026-08-24 09:55:00",
    "load_score": 3.136218,
    "temperature_c": 66.97,
    "requires_review": True,
}
verificaciones = {
    "server_ok": fila_23["server"] == esperado["server"],
    "timestamp_ok": fila_23["timestamp"] == esperado["timestamp"],
    "load_score_ok": fila_23["load_score_6dec"] == esperado["load_score"],
    "temperature_c_ok": abs(fila_23["temperature_c"] - esperado["temperature_c"]) < 1e-9,
    "requires_review_ok": fila_23["requires_review"] == esperado["requires_review"],
    "requires_review_dtype_bool": df_nuevo["requires_review"].dtype == bool,
    "load_score_dtype_float": bool(pd.api.types.is_float_dtype(df_nuevo["load_score"])),
    "columnas_nuevas_presentes": {"load_score", "requires_review"}.issubset(df_nuevo.columns),
    "original_sin_columnas_nuevas": ("load_score" not in df_original.columns)
    and ("requires_review" not in df_original.columns),
    "original_identico_al_snapshot": bool(df_original.equals(snapshot_original)),
    "load_score_coincide_con_recalculo": coincide_con_recalculo,
}
criterio_cumplido = all(verificaciones.values())

# ----------------------------------------------------------------------------
# 6) Archivos exigidos / de apoyo
# ----------------------------------------------------------------------------
RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df_nuevo.to_parquet(RUTA_PARQUET, index=False)

RUTA_CARPETA_T3.mkdir(parents=True, exist_ok=True)
df_nuevo.to_parquet(RUTA_CARPETA_T3 / "dataframe_enriquecido.parquet", index=False)

mascara_revision = df_nuevo["requires_review"].to_numpy(dtype=bool)
resultados = {
    "subtarea": "T3_identificar_observaciones_para_revision",
    "n_filas": int(len(df_nuevo)),
    "shape_nuevo_df": [int(df_nuevo.shape[0]), int(df_nuevo.shape[1])],
    "columnas_nuevo_df": list(df_nuevo.columns),
    "dtypes_nuevo_df": {c: str(t) for c, t in df_nuevo.dtypes.items()},
    "fuente_load_score": fuente_load_score,
    "fila_23": fila_23,
    "fila_23_esperada": esperado,
    "fila_23_completa": fila_23_completa,
    "verificaciones": verificaciones,
    "criterio_cumplido": bool(criterio_cumplido),
    "original_sin_cambios": bool(
        verificaciones["original_identico_al_snapshot"]
        and verificaciones["original_sin_columnas_nuevas"]
    ),
    "n_requires_review": int(mascara_revision.sum()),
    "n_sin_revision": int((~mascara_revision).sum()),
    "indices_requires_review": [int(i) for i in np.flatnonzero(mascara_revision)],
    "load_score": [float(v) for v in load_score],
    "requires_review": [bool(v) for v in mascara_revision],
    "parquet_enriquecido": str(RUTA_PARQUET),
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)
with open(RUTA_CARPETA_T3 / "resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# 7) Resumen
# ----------------------------------------------------------------------------
print("=== T3: copia del DataFrame con load_score y requires_review ===")
print(f"Nuevo DataFrame: {df_nuevo.shape[0]} filas x {df_nuevo.shape[1]} columnas")
print(f"Columnas: {list(df_nuevo.columns)}")
print(
    f"dtypes nuevos: load_score={df_nuevo['load_score'].dtype} | "
    f"requires_review={df_nuevo['requires_review'].dtype}"
)
print(f"Observaciones a revisar: {int(mascara_revision.sum())} de {len(df_nuevo)}")
print(f"load_score obtenido de: {fuente_load_score} (coincide con recálculo: {coincide_con_recalculo})")
print("\nFila 23 (columnas del resultado esperado):")
print(
    df_nuevo.loc[
        FILA_OBJETIVO, ["server", "timestamp", "load_score", "temperature_c", "requires_review"]
    ].to_string()
)
print(f"\nDataFrame original sin cambios: {resultados['original_sin_cambios']}")
print(f"Criterio de éxito cumplido: {criterio_cumplido}")
print(f"Parquet escrito en: {RUTA_PARQUET}")
