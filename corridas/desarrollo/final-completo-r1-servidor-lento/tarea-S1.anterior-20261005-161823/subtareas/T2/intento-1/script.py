# -*- coding: utf-8 -*-
"""
T2: Sobre la matriz de características de T1 (gpu_utilization, cpu_utilization,
memory_gb): media y desviación estándar por columna, normalización Z-score
vectorizada (evitando división por cero) y load_score = zscores @ [0.50, 0.30, 0.20].
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

np.set_printoptions(precision=8, suppress=False)

RUTA_CSV = "data/server_measurements.csv"
RUTA_PARQUET = "output/server_analysis.parquet"
RUTA_RESULTADOS = "resultados.json"
PESOS = np.array([0.50, 0.30, 0.20])
TOL = 1e-10


# ---------------------------------------------------------------------------
# 1) Matriz de características de T1 (misma selección y conversión que T1)
# ---------------------------------------------------------------------------
class ServerMeasurements:
    """Representa la matriz de características construida en T1."""

    FEATURE_NAMES = ("gpu_utilization", "cpu_utilization", "memory_gb")

    def __init__(self, values, feature_names=FEATURE_NAMES):
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("values debe ser una matriz 2D")
        names = tuple(feature_names)
        if arr.shape[1] != len(names):
            raise ValueError("El número de columnas no coincide con feature_names")
        if arr.size and not np.isfinite(arr).all():
            raise ValueError("Los datos deben ser numéricos (sin NaN/inf)")
        self.values = arr
        self.feature_names = names

    def __repr__(self):
        return f"ServerMeasurements(shape={self.values.shape}, features={self.feature_names})"


def build_measurements(df):
    columnas = ["gpu_utilization", "cpu_utilization", "memory_gb"]
    valores = df[columnas].to_numpy(dtype=np.float64)
    return ServerMeasurements(valores, tuple(columnas))


df = pd.read_csv(RUTA_CSV, parse_dates=["timestamp"])
mediciones = build_measurements(df)
X = mediciones.values  # matriz (300, 3) de T1

# ---------------------------------------------------------------------------
# 2) Media y desviación estándar por columna (vectorizado)
# ---------------------------------------------------------------------------
means = X.mean(axis=0)
stds = X.std(axis=0, ddof=0)  # desviación poblacional (Z-score estándar)

# ---------------------------------------------------------------------------
# 3) Normalización Z-score vectorizada, evitando división por cero
# ---------------------------------------------------------------------------
safe_stds = np.where(stds == 0.0, 1.0, stds)  # si std == 0 -> columna constante -> z = 0
zscores = (X - means) / safe_stds

# ---------------------------------------------------------------------------
# 4) load_score = zscores @ [0.50, 0.30, 0.20] (vectorizado, sin for)
# ---------------------------------------------------------------------------
load_score = zscores @ PESOS

# ---------------------------------------------------------------------------
# 5) Verificaciones del criterio de éxito
# ---------------------------------------------------------------------------
means_esperados = np.array([57.99313333, 50.54163333, 29.45533333])
means_coinciden = bool(np.allclose(means, means_esperados, rtol=0.0, atol=1e-8))

z_col_means = zscores.mean(axis=0)
z_col_stds = zscores.std(axis=0, ddof=0)
z_col_stds_ddof1 = zscores.std(axis=0, ddof=1)  # informativo
max_abs_media = float(np.max(np.abs(z_col_means)))
max_abs_std = float(np.max(np.abs(z_col_stds - 1.0)))
media_cero_ok = bool(max_abs_media < TOL)
std_uno_ok = bool(max_abs_std < TOL)

shapes_ok = bool(zscores.shape == (300, 3) and load_score.shape == (300,))
# comprobación independiente (einsum) de que load_score = zscores @ pesos
load_score_check = bool(
    np.allclose(load_score, np.einsum("ij,j->i", zscores, PESOS), rtol=0.0, atol=1e-12)
)

exito = bool(means_coinciden and shapes_ok and media_cero_ok and std_uno_ok and load_score_check)

# ---------------------------------------------------------------------------
# 6) Archivo exigido por el enunciado: output/server_analysis.parquet
# ---------------------------------------------------------------------------
Path(RUTA_PARQUET).parent.mkdir(parents=True, exist_ok=True)
df_salida = df.copy()
df_salida["load_score"] = load_score
df_salida.to_parquet(RUTA_PARQUET, index=False)

# ---------------------------------------------------------------------------
# 7) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "T2_normalizar_y_calcular_load_score",
    "feature_names": list(mediciones.feature_names),
    "n_observaciones": int(X.shape[0]),
    "shape_matriz_caracteristicas": [int(X.shape[0]), int(X.shape[1])],
    "means": means.tolist(),
    "means_esperados": [57.99313333, 50.54163333, 29.45533333],
    "means_coinciden": means_coinciden,
    "stds": stds.tolist(),
    "ddof": 0,
    "alguna_std_cero": bool(np.any(stds == 0.0)),
    "pesos": PESOS.tolist(),
    "zscores_shape": [int(zscores.shape[0]), int(zscores.shape[1])],
    "load_score_shape": [int(load_score.shape[0])],
    "zscores": zscores.tolist(),
    "load_score": load_score.tolist(),
    "primeras_5_load_score": load_score[:5].tolist(),
    "zscores_column_means": z_col_means.tolist(),
    "zscores_column_stds": z_col_stds.tolist(),
    "zscores_column_stds_ddof1": z_col_stds_ddof1.tolist(),
    "max_abs_media_zscores": max_abs_media,
    "max_abs_std_menos_uno": max_abs_std,
    "media_cero_ok": media_cero_ok,
    "std_uno_ok": std_uno_ok,
    "shapes_ok": shapes_ok,
    "load_score_igual_a_zscores_por_pesos": load_score_check,
    "load_score_stats": {
        "media": float(load_score.mean()),
        "desv": float(load_score.std(ddof=0)),
        "min": float(load_score.min()),
        "max": float(load_score.max()),
    },
    "parquet_escrito": RUTA_PARQUET,
    "exito": exito,
}

with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# 8) Resumen
# ---------------------------------------------------------------------------
print("Matriz de T1:", repr(mediciones))
print("Medias:", means)
print("Desviaciones estandar:", stds)
print("zscores.shape    ->", zscores.shape)
print("load_score.shape ->", load_score.shape)
print("Media de columnas de zscores:", z_col_means)
print("Desv. de columnas de zscores:", z_col_stds)
print("load_score (primeros 5):", load_score[:5])
print(
    "Verificaciones: means_coinciden=%s shapes_ok=%s media_cero_ok=%s std_uno_ok=%s load_score_ok=%s"
    % (means_coinciden, shapes_ok, media_cero_ok, std_uno_ok, load_score_check)
)
print("Parquet escrito en:", RUTA_PARQUET)
print("EXITO:", exito)
