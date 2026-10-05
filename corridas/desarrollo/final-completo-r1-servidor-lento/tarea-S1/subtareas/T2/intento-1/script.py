"""
T2 — Normalizar con Z-score la matriz de T1 y calcular load_score (vectorizado).

Entradas:
  - data/server_measurements.csv   (datos de la tarea)
  - entrada/T1/results.json        (nombres de las 3 características en el orden de T1)

Salidas:
  - resultados.json                (contrato de la subtarea)
  - output/server_analysis.parquet (archivo exigido por el enunciado; T3 lo ampliará)
  - t2_zscores_load_score.png      (figura ilustrativa)
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
RUTA_CSV = Path("data/server_measurements.csv")
RUTA_T1 = Path("entrada/T1/results.json")
RUTA_RESULTADOS = Path("resultados.json")
RUTA_PARQUET = Path("output/server_analysis.parquet")

PESOS = np.array([0.50, 0.30, 0.20])   # pesos de load_score
TOL = 1e-10                            # tolerancia exigida para media 0 / std 1
FEATURES_POR_DEFECTO = ["gpu_utilization", "cpu_utilization", "memory_gb"]
MEANS_ESPERADOS = np.array([57.99313333, 50.54163333, 29.45533333])

# ----------------------------------------------------------------------------
# 1) Recuperar la matriz de T1 (mismas 3 características, mismo orden, float64)
# ----------------------------------------------------------------------------
feature_names = list(FEATURES_POR_DEFECTO)
if RUTA_T1.exists():
    with open(RUTA_T1, encoding="utf-8") as f:
        t1 = json.load(f)
    nombres = t1.get("feature_names")
    if isinstance(nombres, list) and len(nombres) == 3:
        feature_names = [str(n) for n in nombres]

df = pd.read_csv(RUTA_CSV, parse_dates=["timestamp"])
X = df[feature_names].to_numpy(dtype=np.float64)  # matriz de T1, forma (300, 3)

# ----------------------------------------------------------------------------
# 2) Media y desviación estándar por columna (ddof=0 => tras Z-score, std = 1)
# ----------------------------------------------------------------------------
means = X.mean(axis=0)
stds = X.std(axis=0, ddof=0)

# ----------------------------------------------------------------------------
# 3) Z-score vectorizado con protección de división por cero
#    (si std == 0 la columna es constante; se usa 1.0 para evitar NaN/inf)
# ----------------------------------------------------------------------------
std_seguro = np.where(stds == 0, 1.0, stds)
zscores = (X - means) / std_seguro

# ----------------------------------------------------------------------------
# 4) load_score con operaciones vectorizadas: zscores @ [0.50, 0.30, 0.20]
# ----------------------------------------------------------------------------
load_score = zscores @ PESOS

# ----------------------------------------------------------------------------
# 5) Verificaciones (el conjunto de prueba no interviene: son chequeos internos)
# ----------------------------------------------------------------------------
media_col = zscores.mean(axis=0)
std_col = zscores.std(axis=0, ddof=0)

media_cero_ok = bool(np.all(np.abs(media_col) <= TOL))
std_uno_ok = bool(np.all(np.abs(std_col - 1.0) <= TOL))

# Contraste independiente: producto elemento a elemento + suma vs. matmul
load_score_alt = (zscores * PESOS).sum(axis=1)
load_score_ok = bool(
    load_score.shape == (X.shape[0],)
    and np.allclose(load_score, load_score_alt, rtol=0.0, atol=1e-12)
)
means_ok = bool(np.allclose(means, MEANS_ESPERADOS, rtol=0.0, atol=1e-8))

# ----------------------------------------------------------------------------
# 6) Escribir resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------------
resultados = {
    "feature_names": feature_names,
    "means": means.tolist(),
    "stds": stds.tolist(),
    "means_coinciden_con_esperados": means_ok,
    "zscores_shape": list(zscores.shape),
    "load_score_shape": list(load_score.shape),
    "zscores_media_por_columna": media_col.tolist(),
    "zscores_std_por_columna": std_col.tolist(),
    "verificaciones": {
        "media_0_por_columna_tol_1e-10": media_cero_ok,
        "std_1_por_columna_tol_1e-10": std_uno_ok,
        "load_score_igual_a_zscores_por_pesos": load_score_ok,
        "n_stds_cero_protegidos": int(np.sum(stds == 0)),
    },
    "load_score": load_score.tolist(),
    "load_score_resumen": {
        "mean": float(load_score.mean()),
        "std": float(load_score.std(ddof=0)),
        "min": float(load_score.min()),
        "max": float(load_score.max()),
    },
}
with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------------------
# 7) Parquet exigido por el enunciado (datos originales + load_score;
#    la subtarea T3 agregará requires_review sobre esta base)
# ----------------------------------------------------------------------------
RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df_salida = df.copy()
df_salida["load_score"] = load_score
df_salida.to_parquet(RUTA_PARQUET, index=False)

# ----------------------------------------------------------------------------
# 8) Figura ilustrativa (PNG, backend Agg)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(zscores[:, 0], bins=30, color="#4C72B0", alpha=0.85)
axes[0].set_title(f"{feature_names[0]} (Z-score)")
axes[0].set_xlabel("z")
axes[0].set_ylabel("frecuencia")
axes[1].hist(load_score, bins=30, color="#DD8452", alpha=0.85)
axes[1].set_title("load_score = zscores @ [0.50, 0.30, 0.20]")
axes[1].set_xlabel("load_score")
axes[1].set_ylabel("frecuencia")
fig.suptitle("T2: normalización Z-score e indicador de carga")
fig.tight_layout()
fig.savefig("t2_zscores_load_score.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# 9) Resumen breve por consola
# ----------------------------------------------------------------------------
print("=== T2: Z-score y load_score ===")
print("Características (orden T1):", feature_names)
print("Matriz de entrada X.shape :", X.shape)
print("Medias por columna        :", means)
print("Desv. estándar por columna:", stds)
print("zscores.shape             :", zscores.shape)
print("load_score.shape          :", load_score.shape)
print("Media de zscores (por col):", media_col, "-> media 0 ok:", media_cero_ok)
print("Std de zscores   (por col):", std_col, "-> std 1 ok  :", std_uno_ok)
print("load_score == zscores @ w :", load_score_ok)
print("means coincide con lo esperado:", means_ok)
print("load_score resumen        :",
      {"mean": float(load_score.mean()), "min": float(load_score.min()),
       "max": float(load_score.max())})
print("Resultados en             :", RUTA_RESULTADOS)
print("Parquet escrito en        :", RUTA_PARQUET)
print("Figura guardada en        : t2_zscores_load_score.png")
