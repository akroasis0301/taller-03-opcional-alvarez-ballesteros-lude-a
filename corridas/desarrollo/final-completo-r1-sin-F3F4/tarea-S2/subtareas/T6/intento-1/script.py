#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T6 — Bootstrap pareado (análisis de sensibilidad aproximado).

- Unidad de remuestreo: TIENDAS COMPLETAS, vía el vector de diferencias
  difference = error_a - error_b (pareado por store_id). NUNCA se remuestrean
  error_a y error_b por separado.
- Generador: np.random.default_rng(SEED=20260829); BOOTSTRAP_RESAMPLES=10000.
- Se reporta el intervalo percentil del 95 % de la media de las diferencias.
- Reproducibilidad: dos ejecuciones con la misma semilla producen exactamente
  el mismo intervalo (verificado dentro del script).
- Salidas exigidas por el enunciado: output/store_differences.csv y
  output/store_differences.png; además resultados.json (contrato) y una
  figura del bootstrap en la carpeta actual.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ------------------------------ Parámetros ----------------------------------
SEED = 20260829
BOOTSTRAP_RESAMPLES = 10_000
CSV_PATH = Path("data/model_errors.csv")
OUT_DIR = Path("output")
T2_RESULTS = Path("entrada/T2/resultados.json")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------- 1) Carga y validación de los datos -------------------
if not CSV_PATH.exists():
    raise FileNotFoundError(f"No se encontró el archivo de datos: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

faltantes = [c for c in ("store_id", "error_a", "error_b") if c not in df.columns]
if faltantes:
    raise ValueError(f"Columnas faltantes en {CSV_PATH}: {faltantes}")

if df["store_id"].isna().any():
    raise ValueError("store_id contiene valores nulos")
duplicados = df.loc[df["store_id"].duplicated(), "store_id"].tolist()
if duplicados:
    raise ValueError(f"store_id duplicado: {duplicados}")

for col in ("error_a", "error_b"):
    num = pd.to_numeric(df[col], errors="coerce")
    if num.isna().any():
        raise ValueError(f"{col} no numérico en tiendas: {df.loc[num.isna(), 'store_id'].tolist()}")
    arr = num.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f"{col} contiene valores no finitos")
    if (arr < 0).any():
        raise ValueError(f"{col} contiene valores negativos")
    df[col] = arr

df["difference"] = df["error_a"] - df["error_b"]
df["favors"] = np.where(df["difference"] > 0, "B",
                        np.where(df["difference"] < 0, "A", "tie"))

diffs = df["difference"].to_numpy(dtype=float)
err_a = df["error_a"].to_numpy(dtype=float)
err_b = df["error_b"].to_numpy(dtype=float)
store_ids = df["store_id"].tolist()
n_tiendas = int(diffs.size)

# ------------- 2) Cruce opcional con la tabla pareada de T2 -----------------
cruce_t2 = {"archivo": str(T2_RESULTS), "verificado": False}
if T2_RESULTS.exists():
    try:
        t2 = json.loads(T2_RESULTS.read_text(encoding="utf-8"))
        tabla_t2 = t2.get("tabla_pareada") or []
        if tabla_t2:
            difs_t2 = {r["store_id"]: float(r["difference"]) for r in tabla_t2}
            fav_t2 = {r["store_id"]: str(r["favors"]) for r in tabla_t2}
            ok_dif = set(difs_t2) == set(store_ids) and all(
                np.isclose(difs_t2[s], d, rtol=0.0, atol=1e-12)
                for s, d in zip(store_ids, diffs)
            )
            ok_fav = all(fav_t2.get(s) == str(f) for s, f in zip(store_ids, df["favors"]))
            cruce_t2 = {
                "archivo": str(T2_RESULTS),
                "verificado": bool(ok_dif and ok_fav),
                "diferencias_coinciden": bool(ok_dif),
                "favors_coinciden": bool(ok_fav),
            }
    except Exception as exc:  # lectura opcional: no debe romper T6
        cruce_t2 = {"archivo": str(T2_RESULTS), "verificado": False,
                    "nota": f"no legible: {exc}"}

# ------------------- 3) Bootstrap pareado (tiendas completas) ---------------
def bootstrap_pareado(vector_diferencias: np.ndarray, semilla: int,
                      n_remuestras: int) -> np.ndarray:
    """Bootstrap PAREADO: se sortean TIENDAS COMPLETAS (índices de tiendas) con
    reemplazo y se promedia el vector de diferencias. Es equivalente a
    remuestrear filas completas (error_a, error_b) y diferenciar dentro de cada
    tienda; NUNCA se remuestrean error_a y error_b por separado."""
    v = np.asarray(vector_diferencias, dtype=float)
    m = v.size
    rng = np.random.default_rng(semilla)
    indices = rng.integers(0, m, size=(n_remuestras, m))  # cada fila = una remuestra de tiendas
    return v[indices].mean(axis=1)


boot_1 = bootstrap_pareado(diffs, SEED, BOOTSTRAP_RESAMPLES)
boot_2 = bootstrap_pareado(diffs, SEED, BOOTSTRAP_RESAMPLES)  # misma semilla → mismo stream

ci_1 = np.percentile(boot_1, [2.5, 97.5])
ci_2 = np.percentile(boot_2, [2.5, 97.5])
ic_lo, ic_hi = float(ci_1[0]), float(ci_1[1])
remuestras_identicas = bool(np.array_equal(boot_1, boot_2))
intervalo_identico = bool(ci_1[0] == ci_2[0] and ci_1[1] == ci_2[1])

# Verificación del pareo: remuestrear filas completas (A, B) conjuntamente da
# la misma distribución que remuestrear el vector de diferencias.
rng_chk = np.random.default_rng(SEED)
idx_chk = rng_chk.integers(0, n_tiendas, size=(BOOTSTRAP_RESAMPLES, n_tiendas))
boot_filas = err_a[idx_chk].mean(axis=1) - err_b[idx_chk].mean(axis=1)
max_abs_diff = float(np.max(np.abs(boot_filas - boot_1)))
pareo_preservado = bool(np.allclose(boot_filas, boot_1, rtol=0.0, atol=1e-9))

# ------------------------- 4) Estadísticos y resumen ------------------------
obs_mean = float(diffs.mean())
obs_sd = float(diffs.std(ddof=1))
obs_median = float(np.median(diffs))
boot_mean = float(boot_1.mean())
boot_se = float(boot_1.std(ddof=1))
boot_bias = boot_mean - obs_mean
pct_pos = float((boot_1 > 0).mean() * 100.0)
pct_neg = float((boot_1 < 0).mean() * 100.0)
b_min, b_q25, b_med, b_q75, b_max = (float(x) for x in
                                     np.percentile(boot_1, [0, 25, 50, 75, 100]))

n_fav_A = int((df["favors"] == "A").sum())
n_fav_B = int((df["favors"] == "B").sum())
n_ties = int((df["favors"] == "tie").sum())

tabla_registros = [
    {"store_id": str(s), "error_a": float(a), "error_b": float(b),
     "difference": float(d), "favors": str(f)}
    for s, a, b, d, f in zip(store_ids, err_a, err_b, diffs, df["favors"])
]

# ------------------------------ 5) Figuras ----------------------------------
colores = ["#1f77b4" if f == "B" else ("#d62728" if f == "A" else "#7f7f7f")
           for f in df["favors"]]

# Figura exigida: output/store_differences.png (diferencias + distribución bootstrap)
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

ax1.bar(np.arange(n_tiendas), diffs, color=colores, edgecolor="black", linewidth=0.4)
ax1.axhline(0.0, color="black", linewidth=0.8)
ax1.axhline(obs_mean, color="green", linestyle="--", linewidth=1.3)
ax1.set_xticks(np.arange(n_tiendas))
ax1.set_xticklabels(store_ids, rotation=90, fontsize=8)
ax1.set_ylabel("difference = error_a - error_b")
ax1.set_title("Diferencias pareadas por tienda (unidad: tienda completa)")
handles = [
    Patch(facecolor="#1f77b4", edgecolor="black", label="favorece a B (dif > 0)"),
    Patch(facecolor="#d62728", edgecolor="black", label="favorece a A (dif < 0)"),
    Line2D([0], [0], color="green", linestyle="--",
           label=f"media observada = {obs_mean:.4f}"),
]
ax1.legend(handles=handles, fontsize=8, loc="lower right")

ax2.hist(boot_1, bins=60, color="#9ecae1", edgecolor="white", linewidth=0.3)
ax2.axvspan(ic_lo, ic_hi, color="red", alpha=0.10)
ax2.axvline(obs_mean, color="green", linestyle="--", linewidth=1.5, label="media observada")
ax2.axvline(ic_lo, color="red", linewidth=1.5)
ax2.axvline(ic_hi, color="red", linewidth=1.5)
ax2.set_xlabel("media de la diferencia en la remuestra bootstrap")
ax2.set_ylabel("frecuencia")
ax2.set_title(f"Bootstrap pareado — {BOOTSTRAP_RESAMPLES} remuestras (semilla {SEED})\n"
              f"IC 95 % percentil = [{ic_lo:.4f}, {ic_hi:.4f}]")
ax2.legend(fontsize=8, loc="upper left")

fig1.tight_layout()
fig1.savefig(str(OUT_DIR / "store_differences.png"), dpi=120)
plt.close(fig1)

# Figura adicional del bootstrap en la carpeta actual
fig2, ax = plt.subplots(figsize=(8.5, 5))
ax.hist(boot_1, bins=60, color="#9ecae1", edgecolor="white", linewidth=0.3)
ax.axvspan(ic_lo, ic_hi, color="red", alpha=0.10, label="IC 95 % percentil")
ax.axvline(obs_mean, color="green", linestyle="--", linewidth=1.5,
           label=f"media observada = {obs_mean:.4f}")
ax.axvline(ic_lo, color="red", linewidth=1.5)
ax.axvline(ic_hi, color="red", linewidth=1.5)
ax.set_xlabel("media de la diferencia en la remuestra bootstrap")
ax.set_ylabel("frecuencia")
ax.set_title(f"Bootstrap pareado — {BOOTSTRAP_RESAMPLES} remuestras de tiendas completas\n"
             f"semilla {SEED} · IC 95 % percentil = [{ic_lo:.4f}, {ic_hi:.4f}]")
ax.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig("bootstrap_percentil_95.png", dpi=120)
plt.close(fig2)

# ------------------- 6) CSV exigido: output/store_differences.csv -----------
columnas_csv = ["store_id", "error_a", "error_b", "difference", "favors"]
df[columnas_csv].to_csv(OUT_DIR / "store_differences.csv", index=False, encoding="utf-8")

# ------------------------- 7) resultados.json (contrato) --------------------
resultados = {
    "subtarea": "T6_bootstrap_pareado_intervalo_percentil_95",
    "seed": SEED,
    "semilla": SEED,
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "bootstrap_remuestras": BOOTSTRAP_RESAMPLES,
    "n_tiendas": n_tiendas,
    "unidad_de_remuestreo": ("tiendas completas: se remuestrea el vector de diferencias "
                             "difference = error_a - error_b (pareado por store_id); "
                             "error_a y error_b nunca se remuestrean por separado"),
    "estadistico_bootstrap": "media de las diferencias en cada remuestra",
    "metodo_intervalo": "percentil empírico (2.5 y 97.5) sobre las 10000 medias remuestradas",
    "tabla_diferencias": tabla_registros,
    "conteo_favors": {"favorecen_A": n_fav_A, "favorecen_B": n_fav_B, "empates": n_ties},
    "estadisticos_observados": {
        "media_diferencia": obs_mean,
        "sd_muestral_diferencia": obs_sd,
        "mediana_diferencia": obs_median,
        "n": n_tiendas,
    },
    "ic95_percentil": {
        "nivel": 95.0,
        "percentiles": [2.5, 97.5],
        "limite_inferior": ic_lo,
        "limite_superior": ic_hi,
    },
    "ic95_percentil_lista": [ic_lo, ic_hi],
    "bootstrap_resumen": {
        "media_remuestras": boot_mean,
        "error_estandar_bootstrap": boot_se,
        "sesgo_bootstrap": boot_bias,
        "min": b_min,
        "q25": b_q25,
        "mediana": b_med,
        "q75": b_q75,
        "max": b_max,
        "porcentaje_remuestras_positivas": pct_pos,
        "porcentaje_remuestras_negativas": pct_neg,
    },
    "reproducibilidad": {
        "semilla": SEED,
        "remuestras": BOOTSTRAP_RESAMPLES,
        "ejecucion_1_ic95": [float(ci_1[0]), float(ci_1[1])],
        "ejecucion_2_ic95": [float(ci_2[0]), float(ci_2[1])],
        "remuestras_identicas_bit_a_bit": remuestras_identicas,
        "intervalo_identico": intervalo_identico,
        "nota": ("np.random.default_rng(SEED) con la misma semilla genera el mismo stream de "
                 "números: dos ejecuciones producen exactamente el mismo intervalo."),
    },
    "verificacion_pareo": {
        "remuestreo_conjunto_de_filas_equivalente_a_remuestrear_diferencias": pareo_preservado,
        "max_dif_absoluta": max_abs_diff,
        "nota": ("Remuestrear filas completas (error_a, error_b) y diferenciar dentro de cada "
                 "tienda coincide (salvo redondeo de punto flotante) con remuestrear el vector "
                 "de diferencias: se preserva el emparejamiento por tienda."),
    },
    "cruce_con_T2": cruce_t2,
    "explicaciones": {
        "por_que_se_remuestrean_tiendas_completas": (
            "Cada tienda es la unidad experimental y aporta un único par (error_a, error_b); la "
            "cantidad de interés es la diferencia dentro de la misma tienda. Los errores de A y B "
            "en una tienda están correlacionados porque comparten sus condiciones (demanda, "
            "ubicación, temporada). Si se remuestrearan error_a y error_b por separado se rompería "
            "el emparejamiento: las diferencias ya no corresponderían a tiendas reales, se "
            "destruiría esa correlación y el intervalo describiría la diferencia entre dos "
            "muestras independientes (un problema distinto, con varianza distinta). Remuestrear "
            "tiendas completas preserva la estructura de pares y la distribución muestral de la "
            "diferencia."
        ),
        "que_no_puede_corregir_el_bootstrap": (
            "El bootstrap solo remuestrea los datos observados: no corrige que las 16 tiendas no "
            "sean una muestra aleatoria representativa de la población (sesgo de selección y "
            "límites de extrapolación), ni dependencias entre tiendas (p. ej., misma cadena o "
            "zona), ni sesgos de medición o confundidores no controlados. Tampoco crea "
            "información: con n=16 el intervalo percentil es solo aproximado y puede ser sensible "
            "a asimetría o valores atípicos; si el diseño está sesgado, el bootstrap hereda ese "
            "sesgo."
        ),
        "papel_del_intervalo": (
            "El intervalo percentil del 95 % del bootstrap pareado se reporta como análisis de "
            "sensibilidad aproximado: complementa la prueba t y la prueba exacta de cambios de "
            "signo mostrando el rango plausible de la diferencia media de errores (A - B) bajo "
            "remuestreo de tiendas completas."
        ),
    },
    "archivos_generados": [
        "output/store_differences.csv",
        "output/store_differences.png",
        "bootstrap_percentil_95.png",
        "resultados.json",
    ],
}

Path("resultados.json").write_text(
    json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
)

# ------------------------------ 8) Resumen ----------------------------------
print("=" * 72)
print("T6 — Bootstrap pareado (análisis de sensibilidad aproximado)")
print(f"  Semilla: {SEED} | Remuestras: {BOOTSTRAP_RESAMPLES} | Tiendas: {n_tiendas}")
print(f"  Media observada de la diferencia (A - B): {obs_mean:.6f}")
print(f"  IC 95 % percentil: [{ic_lo:.6f}, {ic_hi:.6f}]")
print(f"  SE bootstrap: {boot_se:.6f} | sesgo: {boot_bias:.6f}")
print(f"  Remuestras > 0: {pct_pos:.2f} % | Remuestras < 0: {pct_neg:.2f} %")
print(f"  Reproducibilidad (2 ejecuciones, misma semilla): intervalo idéntico = "
      f"{intervalo_identico}")
print(f"  Pareo preservado (remuestreo conjunto de filas ≡ diferencias): "
      f"{pareo_preservado} (max dif = {max_abs_diff:.2e})")
print("  Archivos: output/store_differences.csv, output/store_differences.png, "
      "bootstrap_percentil_95.png, resultados.json")
print("=" * 72)
