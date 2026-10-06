#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 — Identificar observaciones para revisión.

A partir del DataFrame original (data/server_measurements.csv, que NO se modifica)
se crea una copia con dos columnas nuevas:
  * load_score      : indicador de T2 = zscores @ [0.50, 0.30, 0.20]
  * requires_review : bool, True si load_score > 1.5 o temperature_c > 80

Muestra la fila marcada para revisión, verifica la fila 23 contra el resultado
esperado, guarda el DataFrame enriquecido en output/server_analysis.parquet y
escribe todas las cifras en resultados.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- rutas
DATA_PATH = Path("data/server_measurements.csv")
T2_JSON = Path("entrada/T2/resultados.json")
OUT_JSON = Path("resultados.json")
PARQUET_PATH = Path("output/server_analysis.parquet")
FIG_PATH = "t3_requires_review.png"

# ------------------------------------------------- parámetros del enunciado
FEATURES = ["gpu_utilization", "cpu_utilization", "memory_gb"]
WEIGHTS = np.array([0.50, 0.30, 0.20], dtype=float)
LOAD_THRESHOLD = 1.5
TEMP_THRESHOLD = 80.0
REVIEW_RULE = "load_score > 1.5 or temperature_c > 80"

EXPECTED_ROW23 = {
    "server": "AI-SRV-01",
    "timestamp": "2026-08-24 09:55:00",
    "load_score": 3.136218,
    "temperature_c": 66.97,
    "requires_review": True,
}

# ----------------------------------------------------------------- datos
if not DATA_PATH.exists():
    raise FileNotFoundError(f"No se encontró el archivo de datos: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
    df["timestamp"] = pd.to_datetime(df["timestamp"])

n_rows = len(df)
original_columns = list(df.columns)
df_original_snapshot = df.copy()  # copia de seguridad para verificar no-mutación

# ------------------------------------------------- load_score (método de T2)
t2 = {}
if T2_JSON.exists():
    try:
        t2 = json.loads(T2_JSON.read_text(encoding="utf-8"))
    except Exception:
        t2 = {}

feature_names = [c for c in t2.get("feature_names", FEATURES) if c in df.columns] or FEATURES
weights = np.asarray(t2.get("weights", WEIGHTS), dtype=float)

X = df[feature_names].to_numpy(dtype=float)
means = np.asarray(t2["means"], dtype=float) if "means" in t2 else X.mean(axis=0)
stds = np.asarray(t2["stds"], dtype=float) if "stds" in t2 else X.std(axis=0, ddof=0)
stds_safe = np.where(stds == 0.0, 1.0, stds)  # evita división por cero

Z = (X - means) / stds_safe   # vectorizado, sin bucles sobre observaciones
load_score = Z @ weights      # forma (300,)

# consistencia con los zscores guardados por T2 (si existen)
t2_zscores_consistent = None
if "zscores" in t2:
    Z2 = np.asarray(t2["zscores"], dtype=float)
    if Z2.shape == Z.shape:
        t2_zscores_consistent = bool(np.allclose(Z2, Z, rtol=0.0, atol=1e-9))

# ----------------------------------------- copia del original + columnas nuevas
df_new = df_original_snapshot.copy()
df_new["load_score"] = load_score
df_new["requires_review"] = (
    (df_new["load_score"] > LOAD_THRESHOLD) | (df_new["temperature_c"] > TEMP_THRESHOLD)
).astype(bool)

original_unmodified = bool(
    "load_score" not in df.columns
    and "requires_review" not in df.columns
    and list(df.columns) == original_columns
    and df.equals(df_original_snapshot)
)

# ------------------------------------------------- fila(s) marcadas para revisión
flagged = df_new[df_new["requires_review"]]
flagged_indices = [int(i) for i in flagged.index]
n_flagged = int(flagged.shape[0])
n_by_load = int((df_new["load_score"] > LOAD_THRESHOLD).sum())
n_by_temp = int((df_new["temperature_c"] > TEMP_THRESHOLD).sum())


def row_record(idx):
    r = df_new.loc[idx]
    return {
        "index": int(idx),
        "server": str(r["server"]),
        "timestamp": str(r["timestamp"]),
        "load_score": float(r["load_score"]),
        "temperature_c": float(r["temperature_c"]),
        "requires_review": bool(r["requires_review"]),
    }


flagged_records = [row_record(i) for i in flagged_indices]

# ------------------------------------------------- verificación de la fila 23
row23 = row_record(23) if 23 in df_new.index else {}
row23_matches_expected = False
if row23:
    row23["load_score_round6"] = round(row23["load_score"], 6)
    row23_matches_expected = bool(
        row23["server"] == EXPECTED_ROW23["server"]
        and row23["timestamp"] == EXPECTED_ROW23["timestamp"]
        and abs(row23["load_score"] - EXPECTED_ROW23["load_score"]) < 1e-6
        and abs(row23["temperature_c"] - EXPECTED_ROW23["temperature_c"]) < 1e-6
        and row23["requires_review"]
    )

# ------------------------------------------------- parquet exigido por el enunciado
PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
df_new.to_parquet(PARQUET_PATH, index=False)
df_rt = pd.read_parquet(PARQUET_PATH)
parquet_roundtrip_ok = bool(
    len(df_rt) == len(df_new) and list(df_rt.columns) == list(df_new.columns)
)

# ------------------------------------------------- figura
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
idx = np.arange(n_rows)

axes[0].scatter(idx, df_new["load_score"], s=12, color="#4878cf", label="load_score")
axes[0].axhline(LOAD_THRESHOLD, color="gray", ls="--", lw=1, label="umbral 1.5")
if flagged_indices:
    axes[0].scatter(
        flagged_indices,
        df_new.loc[flagged_indices, "load_score"],
        s=70, color="crimson", zorder=3, label="requires_review = True",
    )
    for i in flagged_indices:
        axes[0].annotate(
            f"#{i}", (i, df_new.at[i, "load_score"]),
            textcoords="offset points", xytext=(6, 4), fontsize=8, color="crimson",
        )
axes[0].set_ylabel("load_score")
axes[0].set_title("T3 - Observaciones marcadas para revision")
axes[0].legend(loc="lower right", fontsize=8)

axes[1].scatter(idx, df_new["temperature_c"], s=12, color="#2e7d32")
axes[1].axhline(TEMP_THRESHOLD, color="gray", ls="--", lw=1, label="umbral 80 C")
hot = df_new.index[df_new["temperature_c"] > TEMP_THRESHOLD].tolist()
if hot:
    axes[1].scatter(hot, df_new.loc[hot, "temperature_c"], s=70, color="crimson", zorder=3)
axes[1].set_ylabel("temperature_c (C)")
axes[1].set_xlabel("Indice de observacion")
axes[1].legend(loc="lower right", fontsize=8)

fig.tight_layout()
fig.savefig(FIG_PATH, dpi=120)
plt.close(fig)

# ------------------------------------------------- resultados.json (contrato)
results = {
    "task": "T3_identificar_observaciones_para_revision",
    "data_source": str(DATA_PATH),
    "review_rule": REVIEW_RULE,
    "n_rows_original": int(n_rows),
    "n_rows": int(len(df_new)),
    "n_columns": int(df_new.shape[1]),
    "original_columns": original_columns,
    "new_columns": ["load_score", "requires_review"],
    "columns": [str(c) for c in df_new.columns],
    "original_unmodified": original_unmodified,
    "requires_review_dtype": str(df_new["requires_review"].dtype),
    "load_score_dtype": str(df_new["load_score"].dtype),
    "load_score_shape": [int(load_score.shape[0])],
    "weights": [float(w) for w in weights],
    "feature_names": feature_names,
    "means_used": [float(m) for m in means],
    "stds_used": [float(s) for s in stds],
    "std_ddof": 0,
    "t2_source_used": bool(t2),
    "t2_zscores_consistent": t2_zscores_consistent,
    "load_score_min": float(load_score.min()),
    "load_score_max": float(load_score.max()),
    "load_score_mean": float(load_score.mean()),
    "thresholds": {
        "load_score_gt": float(LOAD_THRESHOLD),
        "temperature_c_gt": float(TEMP_THRESHOLD),
    },
    "n_flagged_by_load_score": n_by_load,
    "n_flagged_by_temperature": n_by_temp,
    "n_requires_review": n_flagged,
    "exactly_one_requires_review": bool(n_flagged == 1),
    "flagged_indices": flagged_indices,
    "flagged_rows": flagged_records,
    "row23": row23,
    "expected_row23": EXPECTED_ROW23,
    "row23_matches_expected": row23_matches_expected,
    "max_temperature_c": float(df_new["temperature_c"].max()),
    "load_score": [float(v) for v in load_score],
    "requires_review": [bool(v) for v in df_new["requires_review"].to_numpy()],
    "parquet_path": str(PARQUET_PATH),
    "parquet_rows": int(len(df_rt)),
    "parquet_columns": [str(c) for c in df_rt.columns],
    "parquet_roundtrip_ok": parquet_roundtrip_ok,
    "figure": FIG_PATH,
}

OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

# ------------------------------------------------- resumen en consola
pd.set_option("display.precision", 6)
print("=" * 70)
print("T3 - Identificar observaciones para revision")
print("=" * 70)
print(f"DataFrame original : {df.shape[0]} filas x {df.shape[1]} columnas "
      f"(sin modificar: {original_unmodified})")
print(f"DataFrame nuevo    : {df_new.shape[0]} filas x {df_new.shape[1]} columnas")
print(f"Columnas nuevas    : load_score, requires_review "
      f"(dtype={df_new['requires_review'].dtype})")
print(f"load_score         : min={load_score.min():.6f}  max={load_score.max():.6f}  "
      f"media={load_score.mean():.6f}")
print(f"Regla              : {REVIEW_RULE}")
print(f"Marcadas (OR)      : {n_flagged} fila(s) "
      f"(por load_score>1.5: {n_by_load}, por temperature_c>80: {n_by_temp})")
print("\nFila(s) marcada(s) para revision:")
cols_show = ["server", "timestamp", "load_score", "temperature_c", "requires_review"]
print(flagged[cols_show].to_string())
print(f"\nVerificacion fila 23 -> coincide con lo esperado: {row23_matches_expected}")
if row23:
    print(f"  server={row23['server']}  timestamp={row23['timestamp']}")
    print(f"  load_score={row23['load_score']:.6f}  "
          f"temperature_c={row23['temperature_c']}  requires_review={row23['requires_review']}")
print(f"\nParquet    : {PARQUET_PATH} ({len(df_rt)} filas, round-trip OK: {parquet_roundtrip_ok})")
print(f"Figura     : {FIG_PATH}")
print(f"Resultados : {OUT_JSON}")
