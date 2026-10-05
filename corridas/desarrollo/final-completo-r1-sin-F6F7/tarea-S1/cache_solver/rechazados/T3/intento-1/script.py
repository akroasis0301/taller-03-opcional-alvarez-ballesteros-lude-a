#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 — Identificar observaciones para revisión.

1) Lee el CSV original (sin modificarlo) y conserva una copia de control.
2) Recupera load_score de T2 (entrada/T2/resultados.json) y lo verifica
   recalculándolo de forma vectorizada con las medias/stds de T2.
3) Crea una COPIA del DataFrame original y le agrega:
      - load_score       (float, proveniente de T2)
      - requires_review  (bool, vectorizado: load_score > 1.5  o  temperature_c > 80)
4) Muestra la fila que requiere revisión, verifica el criterio de éxito,
   guarda el DataFrame analizado en output/server_analysis.parquet,
   una figura PNG y todas las cifras en resultados.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUTA_CSV = Path("data/server_measurements.csv")
RUTA_T2 = Path("entrada/T2/resultados.json")
RUTA_PARQUET = Path("output/server_analysis.parquet")
RUTA_FIG = Path("t3_revision.png")
RUTA_RESULT = Path("resultados.json")

UMBRAL_LOAD = 1.5
UMBRAL_TEMP = 80.0
PESOS = np.array([0.50, 0.30, 0.20], dtype=float)

ESPERADO = {
    "indice": 23,
    "server": "AI-SRV-01",
    "timestamp": "2026-08-24 09:55:00",
    "load_score": 3.136218,
    "temperature_c": 66.97,
}


def normalizar_columnas(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").rstrip("_") for c in df.columns]
    return df


def buscar_col(df, clave):
    for c in df.columns:
        if clave in c:
            return c
    raise KeyError(f"Columna con '{clave}' no encontrada en {list(df.columns)}")


# ------------------------------------------------------------------ 1) datos
df_original = normalizar_columnas(pd.read_csv(RUTA_CSV))
col_server = buscar_col(df_original, "server")
col_ts = buscar_col(df_original, "timestamp")
col_gpu = buscar_col(df_original, "gpu")
col_cpu = buscar_col(df_original, "cpu")
col_mem = buscar_col(df_original, "memory")
col_temp = buscar_col(df_original, "temperature")

if not np.issubdtype(df_original[col_ts].dtype, np.datetime64):
    df_original[col_ts] = pd.to_datetime(df_original[col_ts])

n = len(df_original)
columnas_originales = list(df_original.columns)
snapshot_original = df_original.copy()  # control: el original no debe cambiar

# ------------------------------------------------- 2) load_score desde T2
t2 = None
if RUTA_T2.exists():
    with open(RUTA_T2, "r", encoding="utf-8") as fh:
        t2 = json.load(fh)

valores = df_original[[col_gpu, col_cpu, col_mem]].to_numpy(dtype=float)
if t2 is not None:
    medias = np.asarray(t2["means"], dtype=float)
    stds = np.asarray(t2.get("stds_usados_para_dividir", t2["stds"]), dtype=float)
    pesos_t2 = np.asarray(t2.get("pesos", PESOS), dtype=float)
else:
    medias = valores.mean(axis=0)
    stds = valores.std(axis=0)
    pesos_t2 = PESOS

# Recálculo vectorizado de verificación (sin división por cero, sin for)
zscores_rec = (valores - medias) / np.where(stds == 0.0, 1.0, stds)
load_score_rec = zscores_rec @ pesos_t2

load_score = None
fuente = None
if t2 is not None:
    if isinstance(t2.get("load_score"), list) and len(t2["load_score"]) == n:
        load_score = np.asarray(t2["load_score"], dtype=float)
        fuente = "entrada/T2/resultados.json (clave 'load_score')"
    elif isinstance(t2.get("zscores"), list) and len(t2["zscores"]) == n:
        load_score = np.asarray(t2["zscores"], dtype=float) @ pesos_t2
        fuente = "entrada/T2/resultados.json (zscores @ pesos de T2)"

max_diff = None
if load_score is None or load_score.shape != (n,):
    load_score = load_score_rec
    fuente = "recalculado de forma vectorizada con medias/stds de T2"
else:
    max_diff = float(np.max(np.abs(load_score - load_score_rec)))

# --------------------------------- 3) copia del original + columnas nuevas
df_analizado = df_original.copy()           # el original permanece intacto
df_analizado["load_score"] = load_score     # shape (n,), vectorizado
df_analizado["requires_review"] = (
    (df_analizado["load_score"] > UMBRAL_LOAD) | (df_analizado[col_temp] > UMBRAL_TEMP)
)

# ------------------------------------------------- 4) fila(s) a revisar
mask = df_analizado["requires_review"].astype(bool)
filas_revision = df_analizado.loc[mask]
n_review = int(mask.sum())
indices_review = [int(i) for i in filas_revision.index]

registros = [
    {
        "indice": int(idx),
        "server": str(fila[col_server]),
        "timestamp": str(fila[col_ts]),
        "load_score": float(fila["load_score"]),
        "temperature_c": float(fila[col_temp]),
        "requires_review": bool(fila["requires_review"]),
    }
    for idx, fila in filas_revision.iterrows()
]

# ------------------------------------------------- verificación del criterio
verif = {
    "una_sola_fila_ok": n_review == 1,
    "indice_ok": False,
    "server_ok": False,
    "timestamp_ok": False,
    "load_score": None,
    "load_score_ok": False,
    "temperature_c": None,
    "temperature_c_ok": False,
    "requires_review_ok": False,
    "exito": False,
}
if n_review == 1:
    r = registros[0]
    verif.update(
        {
            "indice_ok": r["indice"] == ESPERADO["indice"],
            "server_ok": r["server"] == ESPERADO["server"],
            "timestamp_ok": r["timestamp"] == ESPERADO["timestamp"],
            "load_score": r["load_score"],
            "load_score_ok": abs(r["load_score"] - ESPERADO["load_score"]) < 5e-7,
            "temperature_c": r["temperature_c"],
            "temperature_c_ok": abs(r["temperature_c"] - ESPERADO["temperature_c"]) < 1e-9,
            "requires_review_ok": r["requires_review"] is True,
        }
    )
    verif["exito"] = bool(
        verif["una_sola_fila_ok"]
        and verif["indice_ok"]
        and verif["server_ok"]
        and verif["timestamp_ok"]
        and verif["load_score_ok"]
        and verif["temperature_c_ok"]
        and verif["requires_review_ok"]
    )

# ------------------------------------- el original no fue modificado
original_sin_cambios = bool(
    list(df_original.columns) == columnas_originales
    and "load_score" not in df_original.columns
    and "requires_review" not in df_original.columns
    and df_original.equals(snapshot_original)
)

# ------------------------------------------------- conteos y estadísticas
n_load_gt = int((df_analizado["load_score"] > UMBRAL_LOAD).sum())
n_temp_gt = int((df_analizado[col_temp] > UMBRAL_TEMP).sum())

# ------------------------------------------------- parquet exigido por el enunciado
RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df_analizado.to_parquet(RUTA_PARQUET, engine="pyarrow")

# ------------------------------------------------- figura PNG
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(
    df_analizado.loc[~mask, "load_score"],
    df_analizado.loc[~mask, col_temp],
    s=18, alpha=0.6, label="Sin revisión",
)
ax.scatter(
    filas_revision["load_score"],
    filas_revision[col_temp],
    color="red", s=110, marker="X", label="requires_review = True",
)
ax.axvline(UMBRAL_LOAD, color="gray", ls="--", lw=1, label="load_score = 1.5")
ax.axhline(UMBRAL_TEMP, color="darkorange", ls=":", lw=1, label="temperature_c = 80")
ax.set_xlabel("load_score")
ax.set_ylabel("temperature_c (°C)")
ax.set_title("T3: observaciones que requieren revisión")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(RUTA_FIG, dpi=120)
plt.close(fig)

# ------------------------------------------------- resultados.json (contrato)
resultados = {
    "subtarea": "T3",
    "criterio_revision": "load_score > 1.5 o temperature_c > 80",
    "umbrales": {"load_score": UMBRAL_LOAD, "temperature_c": UMBRAL_TEMP},
    "n_observaciones": int(n),
    "columnas_originales": columnas_originales,
    "columnas_df_analizado": list(df_analizado.columns),
    "requires_review_dtype": str(df_analizado["requires_review"].dtype),
    "load_score_shape": [int(df_analizado["load_score"].shape[0])],
    "load_score_fuente": fuente,
    "max_diff_load_score_recalculado": max_diff,
    "n_requires_review": n_review,
    "indices_requires_review": indices_review,
    "filas_requires_review": registros,
    "n_load_score_mayor_umbral": n_load_gt,
    "n_temperature_mayor_umbral": n_temp_gt,
    "load_score_min": float(df_analizado["load_score"].min()),
    "load_score_max": float(df_analizado["load_score"].max()),
    "load_score_media": float(df_analizado["load_score"].mean()),
    "temperature_c_max": float(df_analizado[col_temp].max()),
    "original_sin_cambios": original_sin_cambios,
    "verificacion_fila_esperada": verif,
    "archivos_generados": [
        str(RUTA_PARQUET),
        str(RUTA_FIG),
        str(RUTA_RESULT),
    ],
}
RUTA_RESULT.write_text(
    json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
)

# ------------------------------------------------- resumen en consola
print("=" * 70)
print("T3 — Identificar observaciones para revisión")
print("=" * 70)
print(f"Observaciones: {n} | Columnas originales: {len(columnas_originales)}")
print(f"load_score fuente: {fuente}")
if max_diff is not None:
    print(f"Verificación (recálculo vectorizado): max|diff| = {max_diff:.3e}")
print(f"Criterio: load_score > {UMBRAL_LOAD} o temperature_c > {UMBRAL_TEMP}")
print(f"Filas con requires_review=True: {n_review} (índices {indices_review})")
print("\nFila(s) que requieren revisión:")
with pd.option_context("display.width", 160):
    print(filas_revision[[col_server, col_ts, "load_score", col_temp, "requires_review"]])
print(f"\nload_score > 1.5: {n_load_gt} fila(s) | temperature_c > 80: {n_temp_gt} fila(s)")
print(f"DataFrame original sin cambios: {original_sin_cambios}")
print(f"Criterio de éxito: {'CUMPLE' if verif['exito'] else 'NO CUMPLE'}")
print(f"Parquet: {RUTA_PARQUET} | Figura: {RUTA_FIG} | Resultados: {RUTA_RESULT}")
