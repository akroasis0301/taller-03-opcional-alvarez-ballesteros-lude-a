#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5 — Referencia nula exacta por cambios de signo (enumeración completa 2^16 = 65536).

Enumera TODAS las configuraciones de signos de las 16 diferencias pareadas
(error_a − error_b por tienda), calcula la media de cada configuración, cuenta
cuántas tienen magnitud al menos tan extrema como la media observada y reporta:
total de configuraciones, conteo de extremas y p-value = conteo/total,
en contraste con el p-value t de la T4.

Salidas:
  - resultados.json (contrato de la subtarea)
  - output/store_differences.png y output/store_differences.csv (exigidos por el enunciado)
  - store_differences.png (copia en carpeta actual) y signflip_null_distribution.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1) Datos: diferencias pareadas por tienda
# ---------------------------------------------------------------------------
DATA_CSV = Path("data/model_errors.csv")
T4_JSON = Path("entrada/T4/resultados.json")
OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_CSV)
faltan = {"store_id", "error_a", "error_b"} - set(df.columns)
if faltan:
    raise ValueError(f"Faltan columnas {sorted(faltan)} en {DATA_CSV}")

store_ids = df["store_id"].astype(str).tolist()
err_a = df["error_a"].to_numpy(dtype=float)
err_b = df["error_b"].to_numpy(dtype=float)
diffs = err_a - err_b                        # diferencia pareada por tienda
n = int(diffs.size)
if n != 16:
    raise ValueError(f"Se esperaban 16 tiendas para 2^16 configuraciones; hay {n}")

# Representación EXACTA en décimas (los datos tienen 1 decimal): la comparación de
# extremidad se hace con enteros y es inmune a artefactos de punto flotante.
d_tenths = np.round(diffs * 10.0).astype(np.int64)
obs_sum_tenths = int(d_tenths.sum())                 # 67
obs_mean = float(diffs.mean())                       # 0.4187499999999997 (fp)
obs_mean_exact = obs_sum_tenths / (10.0 * n)         # 0.41875 exacto
thr_tenths = abs(obs_sum_tenths)                     # umbral: |suma| >= 67 décimas

# ---------------------------------------------------------------------------
# 2) p-value t de la T4 (se lee de entrada/T4/resultados.json; si no está
#    disponible, se recalcula la MISMA prueba con scipy; no se ajusta nada)
# ---------------------------------------------------------------------------
t_res = stats.ttest_1samp(diffs, popmean=0.0)        # bilateral por defecto
t_stat_mine = float(t_res.statistic)
t_p_mine = float(t_res.pvalue)
t_df = n - 1

t_p_used, t_stat_used, t_df_used = t_p_mine, t_stat_mine, t_df
t4_source = "recalculado con scipy.stats.ttest_1samp (entrada/T4/resultados.json no disponible)"
t4_p_reported = None
if T4_JSON.exists():
    try:
        t4 = json.loads(T4_JSON.read_text(encoding="utf-8"))
        tt = t4.get("t_test", {}) if isinstance(t4, dict) else {}
        if "p_value" in tt:
            t4_p_reported = float(tt["p_value"])
            t_p_used = t4_p_reported
            t_stat_used = float(tt.get("statistic", t_stat_mine))
            t_df_used = int(tt.get("df", t_df))
            t4_source = "entrada/T4/resultados.json"
    except Exception:
        pass
t_coincide = bool(t4_p_reported is None or abs(t4_p_reported - t_p_mine) < 1e-9)

# ---------------------------------------------------------------------------
# 3) Enumeración EXACTA de las 2^16 = 65536 configuraciones de signos
# ---------------------------------------------------------------------------
n_config = 2 ** n                                                    # 65536
idx = np.arange(n_config, dtype=np.int64)
bits = (idx[:, None] >> np.arange(n, dtype=np.int64)[None, :]) & 1   # (65536,16) 0/1
signs = (1 - 2 * bits).astype(np.int64)                              # +1 / -1
del bits
sums_tenths = signs @ d_tenths                                       # (65536,) int64
null_means = sums_tenths / (10.0 * n)                                # medias nulas

extreme_mask = np.abs(sums_tenths) >= thr_tenths
count_extreme = int(extreme_mask.sum())
p_exact = count_extreme / n_config

n_ge = int((sums_tenths >= thr_tenths).sum())    # cola positiva
n_le = int((sums_tenths <= -thr_tenths).sum())   # cola negativa

# Verificaciones estructurales
check_all_plus = bool(int(sums_tenths[0]) == obs_sum_tenths)         # config. observada
check_all_minus = bool(int(sums_tenths[-1]) == -obs_sum_tenths)      # espejo
mean_null = float(null_means.mean())
sd_null_emp = float(null_means.std(ddof=0))
sd_null_teo = float(np.sqrt(np.sum(diffs ** 2) / n ** 2))            # sqrt(Σd²/n²)

null_summary = {
    "n_valores": n_config,
    "media": mean_null,
    "sd_ddof0": sd_null_emp,
    "sd_teorica_sqrt_sum_d2_sobre_n2": sd_null_teo,
    "min": float(null_means.min()),
    "max": float(null_means.max()),
    "q_025": float(np.quantile(null_means, 0.025)),
    "q_975": float(np.quantile(null_means, 0.975)),
    "n_sumas_distintas": int(np.unique(sums_tenths).size),
    "nota": ("distribución simétrica alrededor de 0; todas las sumas nulas son impares "
             "en décimas porque la suma observada (67) es impar y cada cambio de signo "
             "la altera en una cantidad par"),
}

alpha = 0.05
decision_sf = "rechazar_H0" if p_exact < alpha else "no_rechazar_H0"
decision_t = "rechazar_H0" if t_p_used < alpha else "no_rechazar_H0"

# ---------------------------------------------------------------------------
# 4) Explicaciones (p-value condicional a H0, diseño, supuestos, no intercambiabilidad)
# ---------------------------------------------------------------------------
exp_pvalue = (
    "El p-value se define de forma condicional a la hipótesis nula: es la probabilidad, "
    "CALCULADA SUPONIENDO H0 VERDADERA, de obtener una estadística al menos tan extrema "
    "como la observada. En esta referencia nula H0 dice que, para cada tienda, el signo de "
    "la diferencia pareada es aleatorio (+d_i o −d_i con probabilidad 1/2, independiente "
    "entre tiendas), de modo que las 2^16 = 65536 configuraciones de signos son igualmente "
    "probables bajo H0. El p-value = conteo/65536 es la proporción de esos mundos nulos con "
    "|media| >= |media observada| = 0.41875. NO es la probabilidad de que H0 sea verdadera "
    "ni la probabilidad de que el efecto 'sea puro azar'."
)
exp_diseno = (
    "Diseño pareado dentro de tienda: cada tienda recibe ambas modalidades y la unidad de "
    "análisis es la diferencia d_i = error_a − error_b. La referencia por cambios de signo "
    "fija las magnitudes observadas y aleatoriza únicamente los signos; la estadística es la "
    "media de las diferencias y la referencia se construye por enumeración exhaustiva "
    "(prueba exacta, sin Monte Carlo). Supone simetría de cada diferencia alrededor de 0 "
    "bajo H0 e independencia entre tiendas; no supone normalidad."
)
exp_supuestos = (
    "Supuestos distintos: la prueba t (T4) modela las 16 diferencias como i.i.d. de una "
    "población con media μ y distribución aproximadamente normal, estima la sd muestral "
    "(1.253379) y calibra con la distribución t de Student con 15 grados de libertad; su H0 "
    "es μ = 0. La referencia por cambios de signo no asume ninguna forma distribucional: su "
    "H0 es la simetría de signos de las diferencias observadas (intercambiabilidad de la "
    "modalidad dentro de cada tienda) y su calibración es la enumeración discreta exacta de "
    "las 65536 medias."
)
exp_no_intercambiables = (
    "Aunque los p-values sean numéricamente cercanos (t: {:.6f}; cambios de signo: {:.6f}), "
    "los procedimientos NO son intercambiables: (1) contrastan hipótesis nulas distintas "
    "(media poblacional 0 bajo normalidad frente a simetría de signos); (2) usan supuestos "
    "distintos (normalidad e i.i.d. con varianza estimada frente a aleatorización de signos "
    "sin supuesto de forma); (3) sus referencias nulas son distintas (distribución t continua "
    "con 15 gl frente a enumeración discreta exacta de 65536 medias); y (4) por tanto pueden "
    "divergir cuando los supuestos fallan (colas pesadas, n pequeño, dependencia entre "
    "tiendas). Aquí coinciden porque con n=16, diferencias sin colas pesadas y un efecto "
    "moderado, la aproximación t y la referencia exacta dan calibraciones muy similares; la "
    "coincidencia numérica no los vuelve equivalentes."
).format(t_p_used, p_exact)

# ---------------------------------------------------------------------------
# 5) Figuras
# ---------------------------------------------------------------------------
# Figura 1: diferencias pareadas por tienda (exigida: output/store_differences.png)
fig1, ax1 = plt.subplots(figsize=(10, 5.5))
colors = ["#d62728" if d < 0 else "#1f77b4" for d in diffs]
bars = ax1.bar(store_ids, diffs, color=colors, edgecolor="black", linewidth=0.4)
ax1.axhline(0.0, color="black", linewidth=0.8)
ax1.axhline(obs_mean_exact, color="green", linestyle="--", linewidth=1.2,
            label=f"media observada = {obs_mean_exact:.4f}")
for rect, d in zip(bars, diffs):
    if d >= 0:
        ax1.text(rect.get_x() + rect.get_width() / 2.0, d + 0.06, f"{d:+.1f}",
                 ha="center", va="bottom", fontsize=8)
    else:
        ax1.text(rect.get_x() + rect.get_width() / 2.0, d - 0.06, f"{d:+.1f}",
                 ha="center", va="top", fontsize=8)
ax1.set_ylim(float(diffs.min()) - 0.7, float(diffs.max()) + 0.7)
ax1.set_xlabel("tienda")
ax1.set_ylabel("diferencia pareada error_a − error_b (unidades)")
ax1.set_title("Diferencias pareadas por tienda (n=16) — base de la prueba exacta por cambios de signo")
ax1.legend(loc="lower left", fontsize=9)
fig1.tight_layout()
fig1.savefig(str(OUT_DIR / "store_differences.png"), dpi=120)
fig1.savefig("store_differences.png", dpi=120)
plt.close(fig1)

# Figura 2: referencia nula exacta vs referencia t
fig2, ax2 = plt.subplots(figsize=(10, 5.5))
ax2.hist(null_means, bins=81, density=True, color="#9ecae1",
         edgecolor="white", linewidth=0.3, label="medias nulas (cambios de signo, 2$^{16}$)")
xmin, xmax = float(null_means.min()), float(null_means.max())
ax2.axvspan(xmin, -obs_mean_exact, color="#d62728", alpha=0.20, zorder=0)
ax2.axvspan(obs_mean_exact, xmax, color="#d62728", alpha=0.20, zorder=0)
ax2.axvline(obs_mean_exact, color="#d62728", linestyle="--", linewidth=1.4,
            label=f"±media observada ({obs_mean_exact:.4f})")
ax2.axvline(-obs_mean_exact, color="#d62728", linestyle="--", linewidth=1.4)
ax2.axvline(0.0, color="black", linewidth=0.8)
se = float(np.std(diffs, ddof=1) / np.sqrt(n))
xg = np.linspace(xmin - 0.15, xmax + 0.15, 600)
ax2.plot(xg, stats.t.pdf(xg, df=t_df_used, scale=se), color="black", linewidth=1.3,
         label=f"referencia t({t_df_used}) escalada con se (T4)")
ax2.set_title("Referencia nula exacta por cambios de signo (2$^{16}$=65536 medias) vs. referencia t")
ax2.set_xlabel("media de las diferencias con signos aleatorizados (unidades)")
ax2.set_ylabel("densidad")
ax2.text(0.02, 0.97,
         f"configuraciones extremas: {count_extreme} / {n_config}\n"
         f"p-value exacto = {count_extreme}/{n_config} = {p_exact:.6f}\n"
         f"p-value t (T4) = {t_p_used:.6f}",
         transform=ax2.transAxes, va="top", ha="left", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.6", alpha=0.9))
ax2.legend(loc="upper right", fontsize=9)
fig2.tight_layout()
fig2.savefig("signflip_null_distribution.png", dpi=120)
plt.close(fig2)

# ---------------------------------------------------------------------------
# 6) CSV exigido por el enunciado: output/store_differences.csv
# ---------------------------------------------------------------------------
csv_df = pd.DataFrame({
    "store_id": store_ids,
    "error_a": err_a,
    "error_b": err_b,
    "difference": diffs,
    "abs_difference": np.abs(diffs),
    "difference_decimas": d_tenths,
    "signo": np.where(diffs > 0, "positivo", np.where(diffs < 0, "negativo", "cero")),
    "favors": np.where(diffs > 0, "B", np.where(diffs < 0, "A", "empate")),
})
csv_df.to_csv(OUT_DIR / "store_differences.csv", index=False)

# ---------------------------------------------------------------------------
# 7) resultados.json (contrato de la subtarea)
# ---------------------------------------------------------------------------
por_tienda = [
    {
        "store_id": sid,
        "error_a": float(a),
        "error_b": float(b),
        "difference": float(d),
        "difference_decimas": int(round(float(d) * 10.0)),
        "signo": "positivo" if d > 0 else ("negativo" if d < 0 else "cero"),
    }
    for sid, a, b, d in zip(store_ids, err_a, err_b, diffs)
]

resultados = {
    "subtarea": "T5_referencia_nula_exacta_cambios_de_signo",
    "datos": {
        "csv": "data/model_errors.csv",
        "n_filas": int(len(df)),
        "columnas": [str(c) for c in df.columns],
    },
    "diferencias_pareadas": por_tienda,
    "n": n,
    "media_observada": obs_mean,
    "media_observada_exacta": obs_mean_exact,
    "suma_observada_decimas": obs_sum_tenths,
    "prueba_exacta_cambios_de_signo": {
        "metodo": ("enumeración exhaustiva de las 2^16 configuraciones de signos de las 16 "
                   "diferencias pareadas; se calcula la media de cada configuración y se "
                   "compara su magnitud con la de la media observada (bilateral)"),
        "estadistico_de_prueba": "media de las diferencias con signos aleatorizados",
        "estadistico_observado": obs_mean_exact,
        "criterio_extremidad": ("|media nula| >= |media observada| (comparación exacta en "
                                "décimas: |suma| >= 67)"),
        "total_configuraciones": n_config,
        "total_configuraciones_check_2_elevado_16": bool(n_config == 2 ** 16),
        "conteo_extremas": count_extreme,
        "conteo_extremas_cola_positiva": n_ge,
        "conteo_extremas_cola_negativa": n_le,
        "p_value": p_exact,
        "p_value_formula": "conteo_extremas / total_configuraciones",
        "incluye_configuracion_observada_y_espejo": True,
        "enumeracion": "exacta (no Monte Carlo); se evalúan las 65536/65536 configuraciones",
        "alpha": 0.05,
        "decision": decision_sf,
    },
    "distribucion_nula_cambios_de_signo": null_summary,
    "contraste_con_prueba_t_T4": {
        "fuente_p_value_t": t4_source,
        "t_statistic": t_stat_used,
        "t_df": t_df_used,
        "t_p_value": t_p_used,
        "t_p_value_recalculado_scipy": t_p_mine,
        "t_p_value_coincide_con_T4": t_coincide,
        "signflip_p_value": p_exact,
        "diferencia_absoluta_p": abs(p_exact - t_p_used),
        "razon_p_signflip_entre_p_t": p_exact / t_p_used,
        "decision_t_alpha_0_05": decision_t,
        "misma_decision_alpha_0_05": bool(decision_t == decision_sf),
        "comentario": ("p-values cercanos y misma decisión al 5% (no rechazar H0), pero "
                       "provienen de referencias nulas y supuestos distintos (ver explicaciones)"),
    },
    "p16": {
        "p_value_cambios_de_signo": p_exact,
        "total_configuraciones": n_config,
        "conteo_extremas": count_extreme,
        "que_representa_el_p_value": exp_pvalue,
        "diseno_y_referencia": exp_diseno,
        "por_que_supuestos_no_identicos_a_t": exp_supuestos,
    },
    "explicaciones": {
        "p_value_condicional_a_H0": exp_pvalue,
        "diseno_y_supuestos_cambios_de_signo": exp_diseno,
        "supuestos_prueba_t": exp_supuestos,
        "no_intercambiabilidad_t_vs_cambios_de_signo": exp_no_intercambiables,
    },
    "verificaciones": {
        "config_todos_mas_reproduce_suma_observada": check_all_plus,
        "config_todos_menos_reproduce_suma_negada": check_all_minus,
        "simetria_cola_positiva_igual_cola_negativa": bool(n_ge == n_le),
        "media_nula_practicamente_cero": bool(abs(mean_null) < 1e-10),
        "sd_empirica_coincide_teorica": bool(abs(sd_null_emp - sd_null_teo) < 1e-10),
        "comparacion_en_enteros_decimas": ("la extremidad se decide con sumas enteras en "
                                           "décimas (|suma| >= 67), inmune a errores de "
                                           "punto flotante"),
    },
    "archivos_escritos": [
        "resultados.json",
        "output/store_differences.png",
        "output/store_differences.csv",
        "store_differences.png",
        "signflip_null_distribution.png",
    ],
    "figuras": {
        "output/store_differences.png": ("barras de las diferencias pareadas por tienda con la "
                                         "media observada marcada (exigido por el enunciado)"),
        "store_differences.png": "copia de la figura anterior en la carpeta actual",
        "signflip_null_distribution.png": ("histograma de las 65536 medias nulas por cambios de "
                                           "signo, región extrema sombreada y referencia t(15) "
                                           "superpuesta"),
    },
}

def _native(obj):
    if isinstance(obj, dict):
        return {str(k): _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _native(obj.tolist())
    return obj

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(_native(resultados), f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 8) Resumen
# ---------------------------------------------------------------------------
print("=" * 74)
print("T5 — Referencia nula exacta por cambios de signo (enumeración 2^16)")
print("=" * 74)
print(f"n diferencias pareadas          : {n}")
print(f"media observada                 : {obs_mean_exact:.6f} "
      f"(suma {obs_sum_tenths} décimas / {10 * n})")
print(f"configuraciones enumeradas      : {n_config}  (= 2^16: {n_config == 2 ** 16})")
print(f"configuraciones extremas        : {count_extreme}  (|media| >= {obs_mean_exact:.5f})")
print(f"p-value exacto (conteo/total)   : {count_extreme}/{n_config} = {p_exact:.6f}")
print(f"p-value t (T4)                  : {t_p_used:.6f}  [fuente: {t4_source}]")
print(f"diferencia |p_signflip - p_t|   : {abs(p_exact - t_p_used):.6f}")
print(f"decisión al 5%                  : cambios de signo -> {decision_sf} | t -> {decision_t}")
print("archivos: resultados.json, output/store_differences.png, output/store_differences.csv,")
print("          store_differences.png, signflip_null_distribution.png")
