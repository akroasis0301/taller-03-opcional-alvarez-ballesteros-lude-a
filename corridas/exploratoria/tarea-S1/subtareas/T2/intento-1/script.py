# -*- coding: utf-8 -*-
"""
T2 — Normalización Z-score (vectorizada) y cálculo de load_score.

Parte de la matriz de características de T1 (gpu_utilization, cpu_utilization,
memory_gb, forma (300, 3), float64). La matriz se reconstruye con la misma
lógica de selección de build_measurements (T1) sobre data/server_measurements.csv
y se verifica contra las estadísticas reportadas en entrada/T1/resultados.json.

Pasos:
  1) media y desviación estándar por columna (operaciones vectorizadas);
  2) Z-score por columna con protección de división por cero (std == 0 -> z = 0);
  3) load_score = zscores @ [0.50, 0.30, 0.20] (producto punto vectorizado);
  4) verificaciones del criterio de éxito (tolerancia 1e-10).

Salidas:
  - resultados.json        (contrato de la subtarea)
  - T2_distribuciones.png  (figura, backend Agg)
  - zscores_T2.npy, load_score_T2.npy (artefactos para subtareas posteriores)
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Parámetros fijos de la subtarea
# --------------------------------------------------------------------------
FEATURES = ("gpu_utilization", "cpu_utilization", "memory_gb")
PESOS = np.array([0.50, 0.30, 0.20], dtype=float)
TOLERANCIA = 1e-10
RUTA_CSV = "data/server_measurements.csv"
RUTA_T1 = os.path.join("entrada", "T1")
MEANS_ESPERADOS = np.array([57.99313333, 50.54163333, 29.45533333])

# --------------------------------------------------------------------------
# 1) Matriz de características de T1 (misma selección y orden que T1)
# --------------------------------------------------------------------------
df = pd.read_csv(RUTA_CSV)
if "timestamp" in df.columns:
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception:
        pass  # la marca de tiempo no interviene en esta subtarea

faltantes = sorted(set(FEATURES) - set(df.columns))
if faltantes:
    raise ValueError(f"Faltan columnas requeridas en el DataFrame: {faltantes}")

valores = df[list(FEATURES)].to_numpy(dtype=float)
if valores.ndim != 2:
    raise ValueError("La matriz debe ser bidimensional (ndim=2).")
if valores.shape[1] != len(FEATURES):
    raise ValueError(
        f"El número de columnas ({valores.shape[1]}) no coincide con la "
        f"cantidad de nombres de características ({len(FEATURES)})."
    )

# --------------------------------------------------------------------------
# 2) Verificación de consistencia con los resultados de T1
# --------------------------------------------------------------------------
verificacion_t1 = {
    "resultados_json_leido": False,
    "medias_coinciden_con_T1": None,
    "desv_std_T1_coincide_con_ddof": None,
}
try:
    with open(os.path.join(RUTA_T1, "resultados.json"), "r", encoding="utf-8") as f:
        t1 = json.load(f)
    verificacion_t1["resultados_json_leido"] = True
    stats_t1 = t1.get("estadisticas_por_caracteristica", {})
    medias_t1 = np.array(
        [
            float(stats_t1["gpu_utilization"]["media"]),
            float(stats_t1["cpu_utilization"]["media"]),
            float(stats_t1["memory_gb"]["media"]),
        ]
    )
    verificacion_t1["medias_coinciden_con_T1"] = bool(
        np.allclose(valores.mean(axis=0), medias_t1, rtol=0.0, atol=1e-9)
    )
    std_t1 = np.array(
        [
            float(stats_t1["gpu_utilization"]["desv_std"]),
            float(stats_t1["cpu_utilization"]["desv_std"]),
            float(stats_t1["memory_gb"]["desv_std"]),
        ]
    )
    if np.allclose(valores.std(axis=0, ddof=0), std_t1, rtol=0.0, atol=1e-9):
        verificacion_t1["desv_std_T1_coincide_con_ddof"] = 0
    elif np.allclose(valores.std(axis=0, ddof=1), std_t1, rtol=0.0, atol=1e-9):
        verificacion_t1["desv_std_T1_coincide_con_ddof"] = 1
except Exception:
    pass

# --------------------------------------------------------------------------
# 3) Estadísticos por columna y Z-score — vectorizado, sin for
# --------------------------------------------------------------------------
means = valores.mean(axis=0)             # media por columna -> shape (3,)
stds = valores.std(axis=0, ddof=0)       # desviación estándar por columna -> shape (3,)

# Protección contra división por cero: si std == 0 la columna es constante y
# su Z-score se define como 0 en todas las filas.
stds_seguras = np.where(stds == 0.0, 1.0, stds)
zscores = (valores - means) / stds_seguras      # broadcasting (300,3) con (3,)
zscores = np.where(stds == 0.0, 0.0, zscores)   # columnas constantes -> 0

# --------------------------------------------------------------------------
# 4) load_score = zscores @ [0.50, 0.30, 0.20] — vectorizado, sin for
# --------------------------------------------------------------------------
load_score = zscores @ PESOS                    # shape (300,)

# --------------------------------------------------------------------------
# 5) Verificaciones del criterio de éxito
# --------------------------------------------------------------------------
medias_z = zscores.mean(axis=0)
stds_z = zscores.std(axis=0)

media_cero_ok = bool(np.all(np.abs(medias_z) <= TOLERANCIA))
std_uno_ok = bool(np.all(np.abs(stds_z - 1.0) <= TOLERANCIA))

load_dot = np.dot(zscores, PESOS)
load_ponderado = (
    0.50 * zscores[:, 0] + 0.30 * zscores[:, 1] + 0.20 * zscores[:, 2]
)
load_ok = bool(
    np.allclose(load_score, load_dot, rtol=0.0, atol=1e-12)
    and np.allclose(load_score, load_ponderado, rtol=0.0, atol=1e-12)
)

means_ok = bool(np.allclose(means, MEANS_ESPERADOS, rtol=0.0, atol=1e-8))
shapes_ok = bool(zscores.shape == (300, 3) and load_score.shape == (300,))
exito = bool(means_ok and shapes_ok and media_cero_ok and std_uno_ok and load_ok)

# --------------------------------------------------------------------------
# 6) Figura (backend Agg -> PNG)
# --------------------------------------------------------------------------
def dibujar_hist(ax, datos, titulo):
    ax.hist(datos, bins=30, color="#4C72B0", edgecolor="white")
    ax.set_title(titulo)
    ax.set_xlabel("valor")
    ax.set_ylabel("frecuencia")


fig, ejes = plt.subplots(2, 2, figsize=(10, 7))
dibujar_hist(ejes[0, 0], zscores[:, 0], "gpu_utilization (z-score)")
dibujar_hist(ejes[0, 1], zscores[:, 1], "cpu_utilization (z-score)")
dibujar_hist(ejes[1, 0], zscores[:, 2], "memory_gb (z-score)")
dibujar_hist(ejes[1, 1], load_score, "load_score")
fig.suptitle("T2: distribuciones de los z-scores y de load_score")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("T2_distribuciones.png", dpi=120)
plt.close(fig)

# --------------------------------------------------------------------------
# 7) Artefactos para subtareas posteriores y resultados.json (contrato)
# --------------------------------------------------------------------------
np.save("zscores_T2.npy", zscores)
np.save("load_score_T2.npy", load_score)

resultados = {
    "subtarea": "T2_normalizar_y_calcular_load_score",
    "archivo_datos": RUTA_CSV,
    "origen_matriz": (
        "Matriz de características de T1: columnas gpu_utilization, "
        "cpu_utilization, memory_gb (en ese orden), float64, leídas de "
        "data/server_measurements.csv con la misma selección de "
        "build_measurements y verificadas contra entrada/T1/resultados.json."
    ),
    "feature_names": list(FEATURES),
    "n_filas": int(valores.shape[0]),
    "n_columnas": int(valores.shape[1]),
    "n_no_finitos": int(np.sum(~np.isfinite(valores))),
    "means": means.tolist(),
    "means_esperados": MEANS_ESPERADOS.tolist(),
    "means_coinciden_con_esperados": means_ok,
    "stds_ddof0_usadas_en_zscore": stds.tolist(),
    "stds_ddof1_informativo": valores.std(axis=0, ddof=1).tolist(),
    "pesos": PESOS.tolist(),
    "tolerancia": TOLERANCIA,
    "zscores_shape": [int(zscores.shape[0]), int(zscores.shape[1])],
    "load_score_shape": [int(load_score.shape[0])],
    "zscores_medias_por_columna": medias_z.tolist(),
    "zscores_stds_por_columna": stds_z.tolist(),
    "media_cero_ok": media_cero_ok,
    "std_uno_ok": std_uno_ok,
    "load_score_igual_producto_punto": load_ok,
    "load_score_min": float(load_score.min()),
    "load_score_max": float(load_score.max()),
    "load_score_media": float(load_score.mean()),
    "load_score_std": float(load_score.std()),
    "load_score_primeros_5": load_score[:5].tolist(),
    "zscores": zscores.tolist(),
    "load_score": load_score.tolist(),
    "verificacion_contra_T1": verificacion_t1,
    "figuras": ["T2_distribuciones.png"],
    "archivos_generados": ["zscores_T2.npy", "load_score_T2.npy"],
    "exito": exito,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# --------------------------------------------------------------------------
# 8) Resumen
# --------------------------------------------------------------------------
print("=" * 64)
print("T2 — Z-score vectorizado y load_score")
print("=" * 64)
print(
    f"Matriz de características   : {valores.shape} (float64), "
    f"no finitos={int(np.sum(~np.isfinite(valores)))}"
)
print(f"means                       : {means}")
print(f"stds (ddof=0)               : {stds}")
print(f"zscores.shape               : {zscores.shape}")
print(f"load_score.shape            : {load_score.shape}")
print(f"media de zscores (por col)  : {medias_z}  -> |media|<=1e-10: {media_cero_ok}")
print(f"std de zscores (por col)    : {stds_z}  -> |std-1|<=1e-10: {std_uno_ok}")
print(
    f"load_score min/max/media    : {load_score.min():.6f} / "
    f"{load_score.max():.6f} / {load_score.mean():.6f}"
)
print(f"load_score == zscores @ w   : {load_ok}")
print(f"means ≈ esperados           : {means_ok}")
print(f"Verificación contra T1      : {verificacion_t1}")
print(f"ÉXITO                       : {exito}")
