from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# T2: media/desviación por columna, Z-score vectorizado (sin div/0) y load_score
# Matriz base: la de T1 (gpu_utilization, cpu_utilization, memory_gb, float)
# =============================================================================

RUTA_CSV = Path("data/server_measurements.csv")
RUTA_PARQUET = Path("output/server_analysis.parquet")
RUTA_RESULTADOS = Path("resultados.json")
RUTA_T1 = Path("entrada/T1/resultados.json")

FEATURES = ("gpu_utilization", "cpu_utilization", "memory_gb")
PESOS = np.array([0.50, 0.30, 0.20], dtype=float)
TOL = 1e-10


# --- Definición de T1 (misma clase) para reconstruir la matriz ---------------
class ServerMeasurements:
    def __init__(self, values, feature_names):
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError("values debe ser una matriz 2D")
        if values.shape[1] != len(feature_names):
            raise ValueError("El número de columnas no coincide con feature_names")
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError("values debe contener datos numéricos")
        self.values = values.astype(float)
        self.feature_names = tuple(feature_names)

    def __repr__(self):
        return (f"ServerMeasurements(shape={self.values.shape}, "
                f"features={self.feature_names})")


def build_measurements(df: pd.DataFrame) -> ServerMeasurements:
    matriz = df.loc[:, list(FEATURES)].to_numpy(dtype=float)
    return ServerMeasurements(matriz, FEATURES)


# --- Carga de datos y matriz de T1 -------------------------------------------
df = pd.read_csv(RUTA_CSV, parse_dates=["timestamp"])
mediciones = build_measurements(df)
X = mediciones.values  # matriz (300, 3) de T1

# Comprobación opcional de coherencia con T1
if RUTA_T1.exists():
    t1 = json.loads(RUTA_T1.read_text(encoding="utf-8"))
    assert list(t1.get("feature_names", [])) == list(FEATURES), \
        "Los nombres de características no coinciden con T1"

# --- Estadísticos por columna (vectorizados, sin bucles sobre observaciones) --
means = X.mean(axis=0)                 # media por columna
stds = X.std(axis=0, ddof=0)           # desviación estándar poblacional (numpy)

# Z-score vectorizado con protección de división por cero:
# si std == 0 la columna es constante y su zscore se define como 0.
std_safe = np.where(stds == 0.0, 1.0, stds)
zscores = (X - means) / std_safe       # broadcasting, forma (300, 3)

# --- load_score vectorizado ---------------------------------------------------
load_score = zscores @ PESOS           # forma (300,)

# --- Verificaciones del criterio de éxito -------------------------------------
means_esperados = np.array([57.99313333, 50.54163333, 29.45533333])
medias_z = zscores.mean(axis=0)
stds_z = zscores.std(axis=0, ddof=0)
load_recalc = zscores @ PESOS

medias_cero = bool(np.all(np.abs(medias_z) <= TOL))
stds_uno = bool(np.all(np.abs(stds_z - 1.0) <= TOL))
load_ok = bool(np.array_equal(load_score, load_recalc))
means_ok = bool(np.allclose(means, means_esperados, rtol=0.0, atol=1e-6))
shapes_ok = (zscores.shape == (300, 3)) and (load_score.shape == (300,))

verificacion = {
    "means_coinciden_esperados": means_ok,
    "zscores_shape": list(zscores.shape),
    "load_score_shape": list(load_score.shape),
    "zscores_medias_cero_tol_1e-10": medias_cero,
    "zscores_stds_uno_tol_1e-10": stds_uno,
    "load_score_igual_a_zscores_por_pesos": load_ok,
    "max_abs_media_zscores": float(np.max(np.abs(medias_z))),
    "max_abs_std_menos_uno": float(np.max(np.abs(stds_z - 1.0))),
    "max_abs_dif_load_score": float(np.max(np.abs(load_score - load_recalc))),
    "sin_nans": bool(np.isfinite(zscores).all() and np.isfinite(load_score).all()),
}

# --- Parquet exigido por el enunciado: output/server_analysis.parquet ---------
RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
z_df = pd.DataFrame(
    zscores,
    columns=[f"{nombre}_z" for nombre in FEATURES],
    index=df.index,
)
df_salida = pd.concat([df, z_df, pd.DataFrame({"load_score": load_score})], axis=1)
df_salida.to_parquet(RUTA_PARQUET, index=False)

# --- resultados.json (contrato de la subtarea) --------------------------------
resultados = {
    "means": means.tolist(),
    "stds": stds.tolist(),
    "stds_ddof1": X.std(axis=0, ddof=1).tolist(),
    "pesos": PESOS.tolist(),
    "zscores_shape": list(zscores.shape),
    "load_score_shape": list(load_score.shape),
    "zscores_medias_por_columna": medias_z.tolist(),
    "zscores_stds_por_columna": stds_z.tolist(),
    "load_score_resumen": {
        "media": float(load_score.mean()),
        "desviacion_estandar": float(load_score.std(ddof=0)),
        "minimo": float(load_score.min()),
        "maximo": float(load_score.max()),
    },
    "n_columnas_std_cero": int(np.sum(stds == 0.0)),
    "verificacion_criterio_exito": verificacion,
    "criterio_exito_cumplido": bool(
        means_ok and shapes_ok and medias_cero and stds_uno and load_ok
    ),
    "parquet_escrito": str(RUTA_PARQUET),
    "n_filas_parquet": int(len(df_salida)),
    "columnas_parquet": list(df_salida.columns),
}

RUTA_RESULTADOS.write_text(json.dumps(resultados, indent=2), encoding="utf-8")

# --- Resumen ------------------------------------------------------------------
print("Matriz de T1:", repr(mediciones))
print("means:", means)
print("stds :", stds)
print("zscores.shape:", zscores.shape, "| load_score.shape:", load_score.shape)
print("Verificación del criterio de éxito:")
for clave, valor in verificacion.items():
    print(f"  {clave}: {valor}")
print("Criterio de éxito cumplido:", resultados["criterio_exito_cumplido"])
print(f"Parquet escrito en {RUTA_PARQUET} ({len(df_salida)} filas, "
      f"{len(df_salida.columns)} columnas)")
