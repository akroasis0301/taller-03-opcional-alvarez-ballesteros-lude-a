#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T4 — Construir un resumen por servidor.

Agrupa el DataFrame de T3 (datos base + load_score + requires_review) por la
columna `server` y construye una tabla con una fila por servidor y exactamente
estas columnas, en este orden:
    server, observations, mean_power_w, max_temperature_c, mean_load, review_count

- observations      : número de filas del servidor.
- mean_power_w      : media de power_w (W).
- max_temperature_c : máximo de temperature_c (°C).
- mean_load         : media de load_score.
- review_count      : conteo de filas con requires_review == True.

Salidas:
- resultados.json                    (contrato de la subtarea)
- tabla_resumen_por_servidor.parquet (tabla resumen)
- t4_resumen_por_servidor.png        (figura resumen)
"""

import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUTA_CSV = "data/server_measurements.csv"
RUTA_T3 = "entrada/T3/resultados.json"
RUTA_RESULTADOS = "resultados.json"
RUTA_PARQUET = "tabla_resumen_por_servidor.parquet"
RUTA_PNG = "t4_resumen_por_servidor.png"

COLUMNAS = ["server", "observations", "mean_power_w",
            "max_temperature_c", "mean_load", "review_count"]
SERVIDORES_ESPERADOS = [f"AI-SRV-{i:02d}" for i in range(1, 7)]


def a_native(v):
    """Convierte escalares de numpy a tipos nativos de Python (JSON-safe)."""
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def registros(df):
    """DataFrame -> lista de dicts con tipos nativos."""
    return [{k: a_native(v) for k, v in rec.items()}
            for rec in df.to_dict(orient="records")]


# ------------------------------------------------------------------ 1) Carga
df_base = pd.read_csv(RUTA_CSV, parse_dates=["timestamp"])
n = len(df_base)

t3 = None
try:
    with open(RUTA_T3, "r", encoding="utf-8") as f:
        t3 = json.load(f)
except (OSError, ValueError):
    t3 = None

df = df_base.copy()
if t3 is not None and isinstance(t3.get("load_score"), list) \
        and len(t3["load_score"]) == n:
    df["load_score"] = np.asarray(t3["load_score"], dtype=float)
    fuente_load = "entrada/T3/resultados.json (lista load_score)"
else:
    # Respaldo: recalcular load_score = 0.5*z_gpu + 0.3*z_cpu + 0.2*z_mem
    from scipy.stats import zscore
    df["load_score"] = np.zeros(n, dtype=float)
    for col, p in {"gpu_utilization": 0.5, "cpu_utilization": 0.3,
                   "memory_gb": 0.2}.items():
        df["load_score"] += p * zscore(df[col].to_numpy(dtype=float), ddof=0)
    fuente_load = "recalculado (zscore ddof=0, pesos 0.5/0.3/0.2)"

# requires_review con el criterio exacto de T3, vectorizado
df["requires_review"] = ((df["load_score"] > 1.5) |
                         (df["temperature_c"] > 80.0)).astype(bool)

# ------------------------------------------------- 2) Agrupación por server
resumen = (
    df.groupby("server", sort=True)
      .agg(observations=("server", "size"),
           mean_power_w=("power_w", "mean"),
           max_temperature_c=("temperature_c", "max"),
           mean_load=("load_score", "mean"),
           review_count=("requires_review", "sum"))
      .reset_index()
      .sort_values("server", kind="stable")
      .reset_index(drop=True)
)
resumen["observations"] = resumen["observations"].astype(int)
resumen["review_count"] = resumen["review_count"].astype(int)
resumen = resumen[COLUMNAS]  # orden exacto de columnas

# Copia redondeada solo para presentación (el JSON guarda valores completos)
resumen_red = resumen.copy()
resumen_red["mean_power_w"] = resumen_red["mean_power_w"].round(4)
resumen_red["max_temperature_c"] = resumen_red["max_temperature_c"].round(2)
resumen_red["mean_load"] = resumen_red["mean_load"].round(6)

# ------------------------------------------------------- 3) Verificaciones
orden_ok = list(resumen.columns) == COLUMNAS
seis_filas_ok = len(resumen) == 6
servidores_ok = resumen["server"].tolist() == SERVIDORES_ESPERADOS

fila1 = resumen.loc[resumen["server"] == "AI-SRV-01"].iloc[0]


def check(nombre, obtenido, esperado, decimales=None, atol=None):
    obtenido = float(obtenido)
    esperado = float(esperado)
    ok = (round(obtenido, decimales) == esperado) if decimales is not None \
        else (obtenido == esperado)
    if not ok and atol is not None:
        ok = abs(obtenido - esperado) <= atol
    return {"metrica": nombre, "obtenido": obtenido,
            "esperado": esperado, "ok": bool(ok)}


verif_srv01 = [
    check("observations", fila1["observations"], 50),
    check("mean_power_w", fila1["mean_power_w"], 363.4598, decimales=4, atol=5e-5),
    check("max_temperature_c", fila1["max_temperature_c"], 86.50, decimales=2),
    check("mean_load", fila1["mean_load"], -0.614527, decimales=6, atol=1e-6),
    check("review_count", fila1["review_count"], 2),
]
verif_srv01_ok = all(c["ok"] for c in verif_srv01)

total_review = int(resumen["review_count"].sum())
n_review_t3 = (int(t3["n_requires_review_true"])
               if t3 and "n_requires_review_true" in t3
               else int(df["requires_review"].sum()))
indices_rr = np.flatnonzero(df["requires_review"].to_numpy()).tolist()
indices_ok = (indices_rr == list(t3["indices_requires_review_true"])
              if t3 and "indices_requires_review_true" in t3 else None)

# ------------------------------------------------- 4) Archivos de salida
resumen.to_parquet(RUTA_PARQUET, index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].bar(resumen["server"], resumen["mean_power_w"], color="#4C72B0")
axes[0].set_title("Potencia media por servidor")
axes[0].set_ylabel("mean_power_w (W)")
axes[0].tick_params(axis="x", rotation=45)
axes[1].bar(resumen["server"], resumen["review_count"], color="#C44E52")
axes[1].set_title("Observaciones marcadas para revisión")
axes[1].set_ylabel("review_count")
axes[1].tick_params(axis="x", rotation=45)
fig.suptitle("T4 — Resumen por servidor")
fig.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(RUTA_PNG, dpi=120)
plt.close(fig)

resultados = {
    "subtarea": "T4_tabla_resumen_por_servidor",
    "fuente_datos": RUTA_CSV,
    "fuente_t3": RUTA_T3,
    "fuente_load_score": fuente_load,
    "n_filas_entrada": int(n),
    "columnas_tabla": COLUMNAS,
    "orden_columnas_correcto": bool(orden_ok),
    "n_filas_tabla": int(len(resumen)),
    "seis_filas_una_por_servidor": bool(seis_filas_ok),
    "servidores": resumen["server"].tolist(),
    "servidores_esperados_ok": bool(servidores_ok),
    "tabla_resumen": registros(resumen),
    "tabla_resumen_redondeada": registros(resumen_red),
    "verificacion_AI-SRV-01": verif_srv01,
    "verificacion_AI-SRV-01_ok": bool(verif_srv01_ok),
    "verificacion_global": {
        "total_observations": int(resumen["observations"].sum()),
        "total_review_count": total_review,
        "n_requires_review_T3": n_review_t3,
        "review_count_coincide_con_T3": bool(total_review == n_review_t3),
        "indices_requires_review_coinciden_con_T3": indices_ok,
    },
    "archivos_generados": {
        "parquet": RUTA_PARQUET,
        "figura_png": RUTA_PNG,
    },
}

with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------- 5) Resumen
print("T4 — Tabla resumen por servidor")
print(resumen_red.to_string(index=False))
print(f"\nFilas: {len(resumen)} (una por servidor) | "
      f"Columnas: {list(resumen.columns)}")
print(f"Verificación AI-SRV-01 (50 | 363.4598 | 86.50 | -0.614527 | 2): "
      f"{'OK' if verif_srv01_ok else 'FALLO'}")
print(f"Totales: observations={int(resumen['observations'].sum())}, "
      f"review_count={total_review} (T3 reporta {n_review_t3})")
print(f"Salidas: {RUTA_RESULTADOS}, {RUTA_PARQUET}, {RUTA_PNG}")
