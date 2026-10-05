#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 - Identificar observaciones para revisión.

Crea una COPIA del DataFrame original (sin modificarlo) y agrega, de forma
vectorizada:
  - load_score      : valores calculados en T2 (zscores @ [0.50, 0.30, 0.20])
  - requires_review : bool -> True si load_score > 1.5  O  temperature_c > 80

Imprime la fila 23 (server, timestamp, load_score, temperature_c,
requires_review), verifica contra el resultado esperado, escribe
output/server_analysis.parquet y guarda el contrato resultados.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rutas (relativas a la carpeta actual) y constantes del enunciado
# ---------------------------------------------------------------------------
RUTA_CSV = Path("data/server_measurements.csv")
RUTA_T2 = Path("entrada/T2/resultados.json")
RUTA_PARQUET = Path("output/server_analysis.parquet")

PESOS = np.array([0.50, 0.30, 0.20])   # pesos fijos definidos en T2
UMBRAL_LOAD = 1.5                      # load_score > 1.5 activa revisión
UMBRAL_TEMP = 80.0                     # temperature_c > 80 activa revisión
FILA = 23                              # fila que se debe imprimir/verificar


def leer_original(ruta):
    """Lee el CSV original con el timestamp parseado como fecha-hora."""
    df = pd.read_csv(ruta)
    col_ts = "timestamp"
    if col_ts not in df.columns:
        col_ts = next(c for c in df.columns
                      if "time" in c.lower() or "fecha" in c.lower())
    df[col_ts] = pd.to_datetime(df[col_ts])
    return df


# ---------------------------------------------------------------------------
# 1) DataFrame original (solo lectura: nunca se modifica)
# ---------------------------------------------------------------------------
df_original = leer_original(RUTA_CSV)
col_temp = ("temperature_c" if "temperature_c" in df_original.columns
            else next(c for c in df_original.columns if "temp" in c.lower()))

# ---------------------------------------------------------------------------
# 2) load_score proveniente de T2 (respaldo: recálculo vectorizado idéntico)
# ---------------------------------------------------------------------------
fuente_load = None
load_score = None
if RUTA_T2.exists():
    with open(RUTA_T2, "r", encoding="utf-8") as f:
        t2 = json.load(f)
    if isinstance(t2, dict) and t2.get("load_score") is not None:
        cand = np.asarray(t2["load_score"], dtype=float)
        if cand.ndim == 1 and cand.shape[0] == len(df_original):
            load_score = cand
            fuente_load = "entrada/T2/resultados.json (load_score de T2)"

if load_score is None:
    feats = ["gpu_utilization", "cpu_utilization", "memory_gb"]
    X = df_original[feats].to_numpy(dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd == 0.0, 1.0, sd)           # protección división por cero
    load_score = ((X - mu) / sd) @ PESOS        # vectorizado, igual que en T2
    fuente_load = "recalculado con el procedimiento de T2 (respaldo)"

# ---------------------------------------------------------------------------
# 3) Copia del original + columnas nuevas (operaciones vectorizadas)
# ---------------------------------------------------------------------------
df = df_original.copy()                          # el original queda intacto
df["load_score"] = load_score
df["requires_review"] = (df["load_score"] > UMBRAL_LOAD) | (df[col_temp] > UMBRAL_TEMP)
df["requires_review"] = df["requires_review"].astype(bool)

# ---------------------------------------------------------------------------
# 4) Fila 23: impresión exigida por la subtarea
# ---------------------------------------------------------------------------
fila = df.iloc[FILA]
cols_vista = ["server", "timestamp", "load_score", col_temp, "requires_review"]

print("=== T3 · Fila 23 del nuevo DataFrame ===")
print(df.loc[df.index[[FILA]], cols_vista].to_string(index=True))
print("-" * 60)
print(f"server          : {fila['server']}")
print(f"timestamp       : {pd.Timestamp(fila['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
print(f"load_score      : {float(fila['load_score']):.6f}")
print(f"temperature_c   : {float(fila[col_temp]):g}")
print(f"requires_review : {bool(fila['requires_review'])}")

# ---------------------------------------------------------------------------
# 5) Verificaciones
# ---------------------------------------------------------------------------
ts_str = pd.Timestamp(fila["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
verif = {
    "server_ok": str(fila["server"]) == "AI-SRV-01",
    "timestamp_ok": ts_str == "2026-08-24 09:55:00",
    "load_score_ok": abs(float(fila["load_score"]) - 3.136218) < 5e-7,
    "temperature_c_ok": abs(float(fila[col_temp]) - 66.97) < 1e-9,
    "requires_review_ok": bool(fila["requires_review"]),
}
verif["fila_23_coincide_con_esperado"] = bool(all(verif.values()))

mascara = df["requires_review"].to_numpy()
indices_true = np.flatnonzero(mascara).tolist()
n_true = int(mascara.sum())
n_load = int((df["load_score"] > UMBRAL_LOAD).sum())
n_temp = int((df[col_temp] > UMBRAL_TEMP).sum())
solo_fila23 = bool(n_true == 1 and indices_true == [FILA])

df_releido = leer_original(RUTA_CSV)
original_intacto = bool(
    df_original.equals(df_releido)
    and "load_score" not in df_original.columns
    and "requires_review" not in df_original.columns
)

# ---------------------------------------------------------------------------
# 6) Parquet exigido por el enunciado + resultados.json (contrato)
# ---------------------------------------------------------------------------
RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(RUTA_PARQUET, index=False)

resultados = {
    "subtarea": "T3_identificar_observaciones_para_revision",
    "n_filas": int(len(df)),
    "columnas_originales": list(df_original.columns),
    "columnas_finales": list(df.columns),
    "columnas_agregadas": ["load_score", "requires_review"],
    "requires_review_dtype": str(df["requires_review"].dtype),
    "load_score_fuente": fuente_load,
    "fila_23": {
        "indice": FILA,
        "server": str(fila["server"]),
        "timestamp": ts_str,
        "load_score": float(fila["load_score"]),
        "load_score_redondeado_6_decimales": round(float(fila["load_score"]), 6),
        "temperature_c": float(fila[col_temp]),
        "requires_review": bool(fila["requires_review"]),
    },
    "verificacion_fila_23": verif,
    "review_count_total": n_true,
    "indices_requires_review": indices_true,
    "n_load_score_mayor_1_5": n_load,
    "n_temperature_c_mayor_80": n_temp,
    "solo_fila_23_requiere_revision": solo_fila23,
    "original_no_modificado": original_intacto,
    "parquet": str(RUTA_PARQUET),
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 7) Resumen breve
# ---------------------------------------------------------------------------
print("-" * 60)
print(f"Filas: {len(df)} | columnas agregadas: load_score, requires_review "
      f"(dtype={df['requires_review'].dtype})")
print(f"requires_review=True: {n_true} fila(s) -> índices {indices_true}")
print(f"  por load_score > 1.5: {n_load} | por temperature_c > 80: {n_temp}")
print(f"Fila 23 coincide con lo esperado: {verif['fila_23_coincide_con_esperado']}")
print(f"DataFrame original sin modificar: {original_intacto}")
print(f"Parquet escrito en: {RUTA_PARQUET} | Contrato: resultados.json")
