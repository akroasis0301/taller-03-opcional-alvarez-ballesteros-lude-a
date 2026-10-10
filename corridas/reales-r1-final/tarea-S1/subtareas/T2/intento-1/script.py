#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2 · Normalizar con Z-score la matriz de características de T1 y calcular load_score.

Pasos:
  1. Recupera la matriz de características de T1 (gpu_utilization, cpu_utilization,
     memory_gb) desde entrada/T1/ o, en su defecto, la reconstruye con la lógica
     exacta de T1 (clase ServerMeasurements + build_measurements) a partir de
     data/server_measurements.csv.
  2. Calcula media y desviación estándar por columna (operaciones vectorizadas).
  3. Normaliza con Z-score evitando la división por cero (si std == 0 se usa 1.0).
  4. Calcula load_score = zscores @ [0.50, 0.30, 0.20] (producto matricial vectorizado).
  5. Verifica numéricamente: media 0 y desviación 1 por columna (tol 1e-10) y
     load_score igual al producto matricial.

Salidas: resultados.json, t2_load_score_histograma.png y
output/server_analysis.parquet (df + load_score).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOL = 1e-10
WEIGHTS = np.array([0.50, 0.30, 0.20])
DEFAULT_FEATURES = ["gpu_utilization", "cpu_utilization", "memory_gb"]
EXPECTED_MEANS = np.array([57.99313333, 50.54163333, 29.45533333])

# ----------------------------------------------------------------------
# 1) Recuperar la matriz de características de T1
# ----------------------------------------------------------------------
t1_dir = Path("entrada/T1")
t1_results = {}
if (t1_dir / "results.json").exists():
    try:
        t1_results = json.loads((t1_dir / "results.json").read_text(encoding="utf-8"))
    except Exception:
        t1_results = {}

feature_names = [str(c) for c in (t1_results.get("feature_names") or DEFAULT_FEATURES)]
expected_shape = t1_results.get("shape")
expected_shape = tuple(int(v) for v in expected_shape) if expected_shape else None


class ServerMeasurements:
    """Réplica de la clase de T1: matriz 2D NumPy + nombres de columnas."""

    def __init__(self, values, feature_names):
        arr = np.asarray(values)
        if arr.ndim != 2:
            raise ValueError("values debe ser una matriz 2D")
        names = tuple(feature_names)
        if arr.shape[1] != len(names):
            raise ValueError("el número de columnas no coincide con feature_names")
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError("los datos deben ser numéricos")
        self.values = arr.astype(float)
        self.feature_names = names

    def __repr__(self):
        return f"ServerMeasurements(shape={self.values.shape}, features={self.feature_names})"


def build_measurements(df, feature_names):
    """Lógica exacta de T1: selecciona las características en orden, como flotantes."""
    return ServerMeasurements(df[list(feature_names)].to_numpy(dtype=float), feature_names)


def load_matrix_from_t1_files():
    """Busca la matriz que T1 pudo haber guardado en entrada/T1/ (.npy/.parquet/.csv)."""
    if not t1_dir.exists():
        return None, None
    for path in sorted(t1_dir.glob("*.npy")):
        try:
            cand = np.load(path, allow_pickle=False)
        except Exception:
            continue
        if (cand.ndim == 2 and cand.shape[1] == len(feature_names)
                and np.issubdtype(cand.dtype, np.number)
                and (expected_shape is None or cand.shape == expected_shape)):
            return cand.astype(float), str(path)
    for pattern, reader in (("*.parquet", pd.read_parquet), ("*.csv", pd.read_csv)):
        for path in sorted(t1_dir.glob(pattern)):
            try:
                cand = reader(path)
            except Exception:
                continue
            if all(c in cand.columns for c in feature_names):
                return cand[feature_names].to_numpy(dtype=float), str(path)
    return None, None


matrix_t1, matrix_t1_source = load_matrix_from_t1_files()

df = None
csv_path = Path("data/server_measurements.csv")
if csv_path.exists():
    try:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    except Exception:
        df = pd.read_csv(csv_path)
    if not all(c in df.columns for c in feature_names):
        feature_names = list(DEFAULT_FEATURES)

t1_matrix_consistent = None
if df is not None:
    # Fuente principal: reconstrucción determinista con la lógica exacta de T1
    measurements = build_measurements(df, feature_names)
    values = measurements.values
    values_source = "data/server_measurements.csv (build_measurements de T1)"
    if matrix_t1 is not None:
        t1_matrix_consistent = bool(
            matrix_t1.shape == values.shape
            and np.allclose(matrix_t1, values, rtol=0.0, atol=1e-9)
        )
elif matrix_t1 is not None:
    values = matrix_t1
    values_source = str(matrix_t1_source)
else:
    raise RuntimeError("No se pudo obtener la matriz de características de T1.")

# ----------------------------------------------------------------------
# 2) Media y desviación estándar por columna + Z-score (vectorizado,
#    sin ningún for sobre observaciones). ddof=0 (criterio de np.std y
#    StandardScaler) para que cada columna de zscores tenga desviación 1.
# ----------------------------------------------------------------------
means = values.mean(axis=0)
stds = values.std(axis=0, ddof=0)
any_zero_std = bool(np.any(stds == 0.0))
safe_stds = np.where(stds == 0.0, 1.0, stds)      # evita la división por cero
zscores = (values - means) / safe_stds            # broadcasting (n,3) - (3,) / (3,)

# ----------------------------------------------------------------------
# 3) load_score con los pesos fijos (operación vectorizada)
# ----------------------------------------------------------------------
load_score = zscores @ WEIGHTS

# ----------------------------------------------------------------------
# 4) Verificaciones numéricas
# ----------------------------------------------------------------------
col_means = zscores.mean(axis=0)
col_stds = zscores.std(axis=0, ddof=0)
col_stds_ddof1 = zscores.std(axis=0, ddof=1)
mean_zero_ok = bool(np.all(np.abs(col_means) <= TOL))
std_one_ok = bool(np.all(np.abs(col_stds - 1.0) <= TOL))
load_score_alt = (zscores * WEIGHTS).sum(axis=1)  # comprobación independiente
load_score_ok = bool(np.allclose(load_score, load_score_alt, rtol=1e-12, atol=1e-12))
shapes_ok = bool(zscores.shape == (values.shape[0], values.shape[1])
                 and load_score.shape == (values.shape[0],))
means_match_expected = bool(np.allclose(means, EXPECTED_MEANS, rtol=0.0, atol=1e-6))
all_ok = bool(mean_zero_ok and std_one_ok and load_score_ok and shapes_ok)

# ----------------------------------------------------------------------
# 5) Impresiones exigidas (medias y formas) + verificaciones
# ----------------------------------------------------------------------
print("=" * 66)
print("T2 · Z-score de la matriz de T1 y load_score")
print("=" * 66)
print(f"Fuente de la matriz : {values_source}")
print(f"Características     : {tuple(feature_names)}")
print(f"values.shape        -> {values.shape}")
print(f"means               = {means}")
print(f"stds                = {stds}")
print(f"zscores.shape       -> {zscores.shape}")
print(f"load_score.shape    -> {load_score.shape}")
print(f"¿alguna std == 0?   : {any_zero_std}")
print("-" * 66)
print(f"media por columna de zscores   = {col_means}  (|media| <= {TOL:g}: {mean_zero_ok})")
print(f"desv. por columna de zscores   = {col_stds}  (|desv-1| <= {TOL:g}: {std_one_ok})")
print(f"load_score == zscores @ pesos  : {load_score_ok}")
print(f"means ≈ esperado {EXPECTED_MEANS.tolist()} : {means_match_expected}")
print(f"primeros 5 load_score          : {load_score[:5]}")

# ----------------------------------------------------------------------
# 6) Figura PNG (backend Agg)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 4.5))
ax.hist(load_score, bins=30, color="#4C72B0", edgecolor="white")
ax.axvline(1.5, color="crimson", linestyle="--", linewidth=1.5,
           label="umbral de revisión 1.5 (se aplica en T3)")
ax.set_xlabel("load_score")
ax.set_ylabel("frecuencia")
ax.set_title("T2 · load_score = zscores @ [0.50, 0.30, 0.20]")
ax.legend()
fig.tight_layout()
fig.savefig("t2_load_score_histograma.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 7) Parquet exigido por el enunciado (df + load_score)
# ----------------------------------------------------------------------
parquet_written = False
if df is not None:
    out_df = df.copy()
    out_df["load_score"] = load_score
    parquet_path = Path("output/server_analysis.parquet")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(parquet_path, index=False)
    parquet_written = True

# ----------------------------------------------------------------------
# 8) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
results = {
    "means": means.tolist(),
    "stds": stds.tolist(),
    "expected_means": EXPECTED_MEANS.tolist(),
    "means_match_expected": means_match_expected,
    "weights": WEIGHTS.tolist(),
    "feature_names": feature_names,
    "matrix_source": values_source,
    "t1_matrix_consistent": t1_matrix_consistent,
    "n_observations": int(values.shape[0]),
    "values_shape": [int(v) for v in values.shape],
    "zscores_shape": [int(v) for v in zscores.shape],
    "load_score_shape": [int(v) for v in load_score.shape],
    "std_ddof": 0,
    "any_zero_std": any_zero_std,
    "zero_std_columns": [feature_names[int(i)] for i in np.flatnonzero(stds == 0.0)],
    "zscores": zscores.tolist(),
    "load_score": load_score.tolist(),
    "load_score_head": load_score[:5].tolist(),
    "load_score_stats": {
        "min": float(load_score.min()),
        "max": float(load_score.max()),
        "mean": float(load_score.mean()),
        "std": float(load_score.std()),
    },
    "verification": {
        "tol": TOL,
        "zscores_col_means": col_means.tolist(),
        "zscores_col_stds_ddof0": col_stds.tolist(),
        "zscores_col_stds_ddof1": col_stds_ddof1.tolist(),
        "mean_zero_ok": mean_zero_ok,
        "std_one_ok": std_one_ok,
        "load_score_equals_matrix_product": load_score_ok,
        "shapes_ok": shapes_ok,
        "all_ok": all_ok,
    },
    "parquet_written": parquet_written,
    "figure": "t2_load_score_histograma.png",
}
Path("resultados.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
)

print("-" * 66)
print("resultados.json escrito | t2_load_score_histograma.png guardado"
      + (" | output/server_analysis.parquet escrito" if parquet_written else ""))
print(f"RESUMEN T2: means={np.array2string(means, precision=8)} | "
      f"zscores{zscores.shape} | load_score{load_score.shape} | "
      f"verificación (media 0, desv 1, producto matricial) = {all_ok}")
