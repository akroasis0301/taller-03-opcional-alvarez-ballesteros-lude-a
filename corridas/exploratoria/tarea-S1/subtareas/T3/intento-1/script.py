# -*- coding: utf-8 -*-
"""
T3: Crear una copia del DataFrame original (sin modificarlo) agregando las columnas
    load_score (de T2) y requires_review (tipo bool), definida vectorizadamente:
        requires_review = (load_score > 1.5) | (temperature_c > 80)
Salidas: resultados.json (contrato), parquet con el DataFrame nuevo y una figura PNG.
"""

import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUTA_DATOS = "data/server_measurements.csv"
RUTA_T2_JSON = "entrada/T2/resultados.json"
RUTA_SALIDA = "resultados.json"
RUTA_PARQUET = "T3_server_measurements_con_revision.parquet"
RUTA_FIGURA = "T3_load_score_requires_review.png"

FEATURES = ["gpu_utilization", "cpu_utilization", "memory_gb"]
PESOS = np.array([0.50, 0.30, 0.20], dtype=float)
UMBRAL_LOAD = 1.5
UMBRAL_TEMP = 80.0
FILA_ESPERADA = {
    "server": "AI-SRV-01",
    "timestamp": "2026-08-24 09:55:00",
    "load_score": 3.136218,
    "temperature_c": 66.97,
    "requires_review": True,
}

# ---------------------------------------------------------------- 1) Datos
df_original = pd.read_csv(RUTA_DATOS, parse_dates=["timestamp"])
cols_originales = list(df_original.columns)
n_filas = int(df_original.shape[0])

# ------------------------------------------------- 2) load_score (de T2)
t2 = None
try:
    with open(RUTA_T2_JSON, "r", encoding="utf-8") as f:
        t2 = json.load(f)
except Exception:
    t2 = None

X = df_original[FEATURES].to_numpy(dtype=float)
fuente_zscores = None
zscores = None

if t2 is not None:
    z_t2 = t2.get("zscores")
    if z_t2 is not None:
        z_arr = np.asarray(z_t2, dtype=float)
        if z_arr.shape == (n_filas, len(FEATURES)):
            zscores = z_arr
            fuente_zscores = "entrada/T2/resultados.json (zscores)"

if zscores is None:
    means, stds = None, None
    if t2 is not None:
        m = t2.get("means")
        s = t2.get("stds_ddof0_usadas_en_zscore")
        if m is not None and s is not None:
            means = np.asarray(m, dtype=float)
            stds = np.asarray(s, dtype=float)
    if means is None or stds is None or means.shape[0] != X.shape[1]:
        means = X.mean(axis=0)
        stds = X.std(axis=0, ddof=0)
        fuente_zscores = "recalculado desde data (medias y std ddof=0)"
    else:
        fuente_zscores = "entrada/T2/resultados.json (means y stds ddof=0)"
    stds_seguras = np.where(stds == 0.0, 1.0, stds)   # evita división por cero
    zscores = (X - means) / stds_seguras              # vectorizado, forma (300, 3)

load_score = zscores @ PESOS                          # vectorizado, forma (300,)

# ------------------------------------------- 3) Verificación contra T2
verif_t2 = {"t2_disponible": bool(t2 is not None), "fuente_zscores": fuente_zscores}
if t2 is not None:
    p5 = t2.get("load_score_primeros_5")
    if p5 is not None:
        verif_t2["max_abs_diff_primeros_5_vs_T2"] = float(
            np.max(np.abs(load_score[:5] - np.asarray(p5, dtype=float)))
        )
    for clave, valor in (("load_score_max", load_score.max()),
                         ("load_score_min", load_score.min()),
                         ("load_score_media", load_score.mean()),
                         ("load_score_std", load_score.std(ddof=0))):
        if clave in t2:
            verif_t2[f"abs_diff_{clave}_vs_T2"] = float(abs(float(valor) - float(t2[clave])))

# ------------------------- 4) Copia del original + columnas nuevas (vectorizado)
df_nuevo = df_original.copy()                         # el original queda intacto
df_nuevo["load_score"] = load_score
df_nuevo["requires_review"] = (
    (df_nuevo["load_score"] > UMBRAL_LOAD) | (df_nuevo["temperature_c"] > UMBRAL_TEMP)
).astype(bool)

# ------------------------------------------------------- 5) Comprobaciones
df_fresco = pd.read_csv(RUTA_DATOS, parse_dates=["timestamp"])
original_sin_modificar = bool(
    df_original.equals(df_fresco)
    and "load_score" not in df_original.columns
    and "requires_review" not in df_original.columns
    and list(df_original.columns) == cols_originales
)
requires_bool = bool(df_nuevo["requires_review"].dtype == np.dtype(bool))

fila = df_nuevo.loc[23]
fila_23 = {
    "server": str(fila["server"]),
    "timestamp": pd.Timestamp(fila["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
    "load_score": float(fila["load_score"]),
    "temperature_c": float(fila["temperature_c"]),
    "requires_review": bool(fila["requires_review"]),
}
fila_23_coincide = bool(
    fila_23["server"] == FILA_ESPERADA["server"]
    and fila_23["timestamp"] == FILA_ESPERADA["timestamp"]
    and abs(fila_23["load_score"] - FILA_ESPERADA["load_score"]) <= 5e-7
    and abs(fila_23["temperature_c"] - FILA_ESPERADA["temperature_c"]) <= 1e-9
    and fila_23["requires_review"] == FILA_ESPERADA["requires_review"]
)

n_true = int(df_nuevo["requires_review"].sum())
n_false = int(n_filas - n_true)
indices_true = [int(i) for i in df_nuevo.index[np.asarray(df_nuevo["requires_review"])]]

# ------------------------------------------------------------- 6) Salidas
df_nuevo.to_parquet(RUTA_PARQUET, index=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].hist(df_nuevo["load_score"], bins=30, color="#4C72B0", edgecolor="white")
axes[0].axvline(UMBRAL_LOAD, color="crimson", ls="--", label="umbral load_score = 1.5")
axes[0].set_xlabel("load_score")
axes[0].set_ylabel("frecuencia")
axes[0].set_title("Distribución de load_score")
axes[0].legend()
mask = df_nuevo["requires_review"].to_numpy()
axes[1].scatter(df_nuevo.loc[~mask, "load_score"], df_nuevo.loc[~mask, "temperature_c"],
                s=14, alpha=0.6, label="requires_review = False")
axes[1].scatter(df_nuevo.loc[mask, "load_score"], df_nuevo.loc[mask, "temperature_c"],
                s=20, alpha=0.85, color="crimson", label="requires_review = True")
axes[1].axvline(UMBRAL_LOAD, color="gray", ls=":")
axes[1].axhline(UMBRAL_TEMP, color="gray", ls=":")
axes[1].set_xlabel("load_score")
axes[1].set_ylabel("temperature_c")
axes[1].set_title("Criterio de revisión (vectorizado)")
axes[1].legend()
fig.tight_layout()
fig.savefig(RUTA_FIGURA, dpi=120)
plt.close(fig)

resultados = {
    "subtarea": "T3_copia_df_con_load_score_y_requires_review",
    "archivo_datos": RUTA_DATOS,
    "n_filas": n_filas,
    "columnas_originales": cols_originales,
    "columnas_nuevas": ["load_score", "requires_review"],
    "columnas_finales": list(df_nuevo.columns),
    "dtype_load_score": str(df_nuevo["load_score"].dtype),
    "dtype_requires_review": str(df_nuevo["requires_review"].dtype),
    "requires_review_es_bool": requires_bool,
    "criterio": "requires_review = (load_score > 1.5) | (temperature_c > 80), vectorizado",
    "umbrales": {"load_score": UMBRAL_LOAD, "temperature_c": UMBRAL_TEMP},
    "pesos_load_score": PESOS.tolist(),
    "fuente_zscores": fuente_zscores,
    "verificacion_contra_T2": verif_t2,
    "load_score_min": float(load_score.min()),
    "load_score_max": float(load_score.max()),
    "load_score_media": float(load_score.mean()),
    "load_score_std_ddof0": float(load_score.std(ddof=0)),
    "n_requires_review_true": n_true,
    "n_requires_review_false": n_false,
    "fraccion_requires_review": n_true / n_filas,
    "indices_requires_review_true": indices_true,
    "fila_23": fila_23,
    "fila_23_esperada": FILA_ESPERADA,
    "fila_23_coincide": fila_23_coincide,
    "original_sin_modificar": original_sin_modificar,
    "load_score": [float(v) for v in load_score.tolist()],
    "requires_review": [bool(v) for v in df_nuevo["requires_review"].tolist()],
    "archivos_generados": {"parquet": RUTA_PARQUET, "figura": RUTA_FIGURA},
}

with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------- 7) Resumen
print("=== T3: copia del DataFrame con load_score y requires_review ===")
print(f"Filas: {n_filas} | Columnas finales: {list(df_nuevo.columns)}")
print(f"dtypes nuevos: load_score={resultados['dtype_load_score']}, "
      f"requires_review={resultados['dtype_requires_review']} (bool={requires_bool})")
print(f"requires_review True: {n_true} / {n_filas} ({n_true / n_filas:.2%}) | False: {n_false}")
print(f"load_score: min={load_score.min():.6f}, max={load_score.max():.6f}")
print(f"Fila 23: {fila_23}")
print(f"Fila 23 coincide con lo esperado: {fila_23_coincide}")
print(f"Original sin modificar: {original_sin_modificar}")
print(f"Verificación vs T2: {verif_t2}")
print(f"Salidas: {RUTA_SALIDA}, {RUTA_PARQUET}, {RUTA_FIGURA}")
