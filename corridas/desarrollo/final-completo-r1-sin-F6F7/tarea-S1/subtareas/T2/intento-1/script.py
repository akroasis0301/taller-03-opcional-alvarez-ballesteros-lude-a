# -*- coding: utf-8 -*-
"""
T2 — Normalizar (Z-score) y calcular el indicador de carga (load_score).

Usa la matriz de T1 (gpu_utilization, cpu_utilization, memory_gb), calcula
media y desviación estándar por columna, normaliza con Z-score de forma
vectorizada (evitando división por cero si alguna std es 0) y calcula
load_score = zscores @ [0.50, 0.30, 0.20] con operaciones vectorizadas
(sin bucles sobre observaciones).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Parámetros fijos del enunciado
# ----------------------------------------------------------------------------
PESOS = np.array([0.50, 0.30, 0.20])
TOL = 1e-10
FEATURES = ("gpu_utilization", "cpu_utilization", "memory_gb")
MEANS_ESPERADOS = np.array([57.99313333, 50.54163333, 29.45533333])

RUTA_CSV = Path("data/server_measurements.csv")
RUTA_RESULTADOS = Path("resultados.json")
RUTA_T1 = Path("entrada/T1/results.json")
RUTA_PARQUET = Path("output/server_analysis.parquet")

# ----------------------------------------------------------------------------
# 1) Matriz de T1: se reconstruye desde el CSV con la misma selección de T1
# ----------------------------------------------------------------------------
df = pd.read_csv(RUTA_CSV, parse_dates=["timestamp"])
values = df[list(FEATURES)].to_numpy(dtype=float)
values_shape_ok = bool(values.ndim == 2 and values.shape == (300, 3))

# Comprobación opcional contra los resultados de T1 (si están disponibles)
t1_means = None
if RUTA_T1.exists():
    try:
        t1 = json.loads(RUTA_T1.read_text(encoding="utf-8"))
        t1_means = np.array(
            [t1["media_por_feature"][f] for f in FEATURES], dtype=float
        )
    except Exception:
        t1_means = None

# ----------------------------------------------------------------------------
# 2) Media y desviación estándar por columna (vectorizado, ddof=0)
# ----------------------------------------------------------------------------
means = values.mean(axis=0)
stds = values.std(axis=0, ddof=0)

# ----------------------------------------------------------------------------
# 3) Z-score vectorizado evitando división por cero
#    (si std == 0, se divide por 1 y la columna queda toda en 0)
# ----------------------------------------------------------------------------
stds_safe = np.where(stds == 0.0, 1.0, stds)
zscores = (values - means) / stds_safe  # broadcasting (300,3) - sin bucles

# ----------------------------------------------------------------------------
# 4) load_score = zscores @ pesos (vectorizado, sin for sobre observaciones)
# ----------------------------------------------------------------------------
load_score = zscores @ PESOS

# ----------------------------------------------------------------------------
# 5) Verificaciones del criterio de éxito
# ----------------------------------------------------------------------------
col_means = zscores.mean(axis=0)
col_stds = zscores.std(axis=0, ddof=0)

means_ok = bool(np.allclose(means, MEANS_ESPERADOS, atol=1e-6))
shape_z_ok = bool(zscores.shape == (300, 3))
shape_ls_ok = bool(load_score.shape == (300,))
media_cero_ok = bool(np.all(np.abs(col_means) <= TOL))
std_uno_ok = bool(np.all(np.abs(col_stds - 1.0) <= TOL))

# Comprobación independiente (también vectorizada) de la identidad del producto
load_score_alt = (zscores * PESOS).sum(axis=1)
load_score_ok = bool(np.allclose(load_score, load_score_alt, rtol=0.0, atol=1e-12))

t1_ok = (
    None
    if t1_means is None
    else bool(np.allclose(means, t1_means, rtol=0.0, atol=1e-12))
)

exito = bool(
    values_shape_ok
    and means_ok
    and shape_z_ok
    and shape_ls_ok
    and media_cero_ok
    and std_uno_ok
    and load_score_ok
)

# ----------------------------------------------------------------------------
# 6) Parquet exigido por el enunciado: df + load_score
# ----------------------------------------------------------------------------
RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df_out = df.copy()
df_out["load_score"] = load_score
df_out.to_parquet(RUTA_PARQUET, index=False)

df_check = pd.read_parquet(RUTA_PARQUET)
parquet_filas = int(len(df_check))
parquet_ok = bool("load_score" in df_check.columns and parquet_filas == len(df))

# ----------------------------------------------------------------------------
# 7) Figuras PNG
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(load_score, bins=30, color="#4C72B0", edgecolor="white")
ax.axvline(
    float(load_score.mean()), color="crimson", linestyle="--",
    label=f"media = {load_score.mean():.4f}",
)
ax.set_title("Distribución de load_score (T2)")
ax.set_xlabel("load_score")
ax.set_ylabel("frecuencia")
ax.legend()
fig.tight_layout()
fig.savefig("load_score_histograma.png", dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.boxplot([zscores[:, 0], zscores[:, 1], zscores[:, 2]])
ax.set_xticklabels(list(FEATURES))
ax.axhline(0.0, color="gray", linestyle=":", linewidth=1)
ax.set_title("Z-scores por característica (media 0, desviación 1)")
ax.set_ylabel("z")
fig.tight_layout()
fig.savefig("zscores_boxplot.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# 8) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------------
resultados = {
    "means": means.tolist(),
    "stds": stds.tolist(),
    "stds_usados_para_dividir": stds_safe.tolist(),
    "columnas_con_std_cero": int(np.sum(stds == 0.0)),
    "pesos": PESOS.tolist(),
    "values_shape": list(values.shape),
    "zscores_shape": list(zscores.shape),
    "load_score_shape": list(load_score.shape),
    "zscores": zscores.tolist(),
    "load_score": load_score.tolist(),
    "zscores_media_por_columna": col_means.tolist(),
    "zscores_std_por_columna": col_stds.tolist(),
    "load_score_stats": {
        "min": float(load_score.min()),
        "max": float(load_score.max()),
        "media": float(load_score.mean()),
        "std": float(load_score.std()),
    },
    "means_esperados": MEANS_ESPERADOS.tolist(),
    "verificacion": {
        "values_shape_correcto": values_shape_ok,
        "means_coinciden_esperados": means_ok,
        "zscores_shape_correcto": shape_z_ok,
        "load_score_shape_correcto": shape_ls_ok,
        "media_cero_tol_1e-10": media_cero_ok,
        "std_uno_tol_1e-10": std_uno_ok,
        "load_score_igual_a_zscores_por_pesos": load_score_ok,
        "means_coinciden_con_T1": t1_ok,
        "exito": exito,
    },
    "parquet": {
        "ruta": str(RUTA_PARQUET),
        "filas": parquet_filas,
        "columnas": list(df_check.columns),
        "escrito_y_verificado": parquet_ok,
    },
    "figuras": ["load_score_histograma.png", "zscores_boxplot.png"],
}

RUTA_RESULTADOS.write_text(
    json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
)

# ----------------------------------------------------------------------------
# 9) Resumen
# ----------------------------------------------------------------------------
print("T2 — Normalización Z-score y load_score")
print(f"  means            = {means}")
print(f"  stds             = {stds} (columnas con std=0: {int(np.sum(stds == 0.0))})")
print(f"  zscores.shape    = {zscores.shape}")
print(f"  load_score.shape = {load_score.shape}")
print(f"  media por columna de zscores = {col_means} (|media|<=1e-10: {media_cero_ok})")
print(f"  std por columna de zscores   = {col_stds} (|std-1|<=1e-10: {std_uno_ok})")
print(
    f"  load_score: min={load_score.min():.6f} max={load_score.max():.6f} "
    f"media={load_score.mean():.6f}"
)
print(f"  load_score == zscores @ [0.50, 0.30, 0.20]: {load_score_ok}")
if t1_ok is not None:
    print(f"  means coinciden con T1: {t1_ok}")
print(f"  Parquet: {RUTA_PARQUET} ({parquet_filas} filas, verificado: {parquet_ok})")
print(f"  CRITERIO DE ÉXITO: {'CUMPLE' if exito else 'NO CUMPLE'}")
