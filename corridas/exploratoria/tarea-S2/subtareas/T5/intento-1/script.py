#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5 — Referencia nula exacta por cambios de signo (enumeración completa 2^16 = 65536).

Pasos:
 1) Carga data/model_errors.csv y reconstruye las diferencias pareadas d_i = error_a - error_b.
 2) Enumera las 2^16 = 65536 configuraciones de signos (+1/-1) de las diferencias y calcula
    la media nula de cada configuración.
 3) Cuenta cuántas medias nulas tienen magnitud al menos tan extrema como la observada
    (criterio bilateral) y reporta: total de configuraciones, conteo de extremas y p-value.
 4) Verifica el conteo con una DP de aritmética entera exacta (sumas escaladas x10).
 5) Compara numéricamente con el p-value t bilateral obtenido en T4 (entrada/T4/resultados.json).

Salidas: resultados.json, figura PNG y CSV de resumen (todo en la carpeta actual).
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RUTA_CSV = Path("data/model_errors.csv")
RUTA_T4 = Path("entrada/T4/resultados.json")
RUTA_RESULTADOS = Path("resultados.json")
FIGURA_PNG = "t5_referencia_nula_cambios_signo.png"
CSV_RESUMEN = "t5_resumen_referencia_nula.csv"
TOL = 1e-9      # tolerancia de punto flotante para '|media nula| >= |media observada|'
ALPHA = 0.05

# ------------------------------------------------------------------ 1) Datos
df = pd.read_csv(RUTA_CSV)
columnas_esperadas = ["store_id", "error_a", "error_b"]
if list(df.columns) != columnas_esperadas:
    raise ValueError(f"Columnas inesperadas en {RUTA_CSV}: {list(df.columns)}")
if len(df) != 16:
    raise ValueError(f"Se esperaban 16 tiendas; se encontraron {len(df)}")

d = df["error_a"].to_numpy(dtype=float) - df["error_b"].to_numpy(dtype=float)
n = int(d.size)
if not np.all(np.isfinite(d)):
    raise ValueError("Las diferencias contienen valores no finitos")

media_obs = float(np.mean(d))
suma_obs = float(np.sum(d))
magnitud_obs = abs(media_obs)

diferencias_por_tienda = [
    {
        "store_id": str(row.store_id),
        "error_a": float(row.error_a),
        "error_b": float(row.error_b),
        "difference": float(row.error_a - row.error_b),
    }
    for row in df.itertuples(index=False)
]

# --------------------------------- 2) Enumeración completa de las 2^16 configuraciones
n_config = 2 ** n  # 65536
ints = np.arange(n_config, dtype=np.int64)
bits = (ints[:, None] >> np.arange(n, dtype=np.int64)[None, :]) & 1
signos = (1 - 2 * bits).astype(np.int8)  # bit=0 -> +1 ; bit=1 -> -1
del bits
sumas_nulas = signos.astype(np.float64) @ d   # suma_i s_i * d_i para cada configuración
medias_nulas = sumas_nulas / n

# --------------------------------------- 3) Conteo de configuraciones extremas
extremas_mask = np.abs(medias_nulas) >= magnitud_obs - TOL
conteo_extremas = int(np.count_nonzero(extremas_mask))
p_value_exacto = conteo_extremas / n_config

cola_positiva = int(np.count_nonzero(medias_nulas >= magnitud_obs - TOL))
cola_negativa = int(np.count_nonzero(medias_nulas <= -(magnitud_obs - TOL)))
conteo_misma_magnitud = int(
    np.count_nonzero(np.isclose(np.abs(medias_nulas), magnitud_obs, rtol=0, atol=TOL))
)

check_obs = bool(np.isclose(medias_nulas[0], media_obs, rtol=0, atol=1e-12))
check_neg = bool(np.isclose(medias_nulas[-1], -media_obs, rtol=0, atol=1e-12))
observada_incluida = bool(extremas_mask[0] and extremas_mask[-1])

# ------------------- 4) Verificación exacta con aritmética entera (DP de sumas)
escala = 10
d_int = np.rint(d * escala).astype(np.int64)
if not np.allclose(d_int / escala, d, rtol=0, atol=1e-9):
    raise ValueError("Las diferencias no son representables exactamente a escala entera x10")
objetivo_int = int(round(abs(suma_obs) * escala))  # |suma escalada| >= |suma observada| x10

dp = {0: 1}
for di in d_int:
    di = int(di)
    nxt = defaultdict(int)
    for s, c in dp.items():
        nxt[s + di] += c
        nxt[s - di] += c
    dp = dict(nxt)

total_dp = int(sum(dp.values()))
conteo_dp = int(sum(c for s, c in dp.items() if abs(s) >= objetivo_int))
coincide_dp = bool(total_dp == n_config and conteo_dp == conteo_extremas)

# ------------------------------ 5) p-value t de T4 y comparación numérica
p_t = None
fuente_p_t = None
estadistico_t = None
gl = None
media_t4 = None
if RUTA_T4.exists():
    try:
        with RUTA_T4.open(encoding="utf-8") as fh:
            t4 = json.load(fh)
        p_t = float(t4["prueba_t"]["p_value_bilateral"])
        fuente_p_t = "entrada/T4/resultados.json"
        if "estadistico_t" in t4["prueba_t"]:
            estadistico_t = float(t4["prueba_t"]["estadistico_t"])
        if "grados_de_libertad" in t4["prueba_t"]:
            gl = int(t4["prueba_t"]["grados_de_libertad"])
        if "mean" in t4:
            media_t4 = float(t4["mean"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        p_t, estadistico_t, gl, media_t4, fuente_p_t = None, None, None, None, None
if p_t is None:
    res_t = stats.ttest_1samp(d, 0.0, alternative="two-sided")
    estadistico_t = float(res_t.statistic)
    gl = n - 1
    p_t = float(res_t.pvalue)
    fuente_p_t = "recalculado con scipy.stats.ttest_1samp (T4 no disponible)"

dif_abs = abs(p_value_exacto - p_t)
dif_rel = dif_abs / p_t if p_t != 0 else float("inf")
misma_decision = bool(p_value_exacto > ALPHA and p_t > ALPHA)
coincide_media = (
    None if media_t4 is None else bool(np.isclose(media_obs, media_t4, rtol=0, atol=1e-9))
)

# Contexto: dispersión de la referencia nula y aproximación normal (solo informativa)
sd_nulas = float(np.std(medias_nulas, ddof=1))
p_normal_aprox = float(2.0 * stats.norm.sf(magnitud_obs / sd_nulas))

# ------------------------------------------------------------------ Figura
bins = np.linspace(float(np.min(medias_nulas)), float(np.max(medias_nulas)), 97)
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.hist(medias_nulas, bins=bins, color="#9ecae1", edgecolor="#4292c6", linewidth=0.3,
        label=f"Medias nulas (2^{n} = {n_config} configuraciones)")
ax.hist(medias_nulas[extremas_mask], bins=bins, color="firebrick", alpha=0.75,
        label=f"Extremas: |media| ≥ {magnitud_obs:.4f} ({conteo_extremas})")
ax.axvline(media_obs, color="darkred", lw=2, label=f"Media observada = {media_obs:.4f}")
ax.axvline(-media_obs, color="darkred", lw=2, ls="--", label=f"−Media observada = {-media_obs:.4f}")
ax.set_xlabel("Media nula de las diferencias (por cambios de signo)")
ax.set_ylabel("Frecuencia (número de configuraciones)")
ax.set_title(
    f"T5 · Referencia nula exacta por cambios de signo (2^{n} = {n_config})\n"
    f"Extremas: {conteo_extremas} → p exacto = {p_value_exacto:.4f} | p t bilateral (T4) = {p_t:.4f}",
    fontsize=10,
)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURA_PNG, dpi=120)
plt.close(fig)

# ------------------------------------------------------------------ CSV de resumen
resumen = pd.DataFrame(
    {
        "metrica": [
            "n_tiendas",
            "media_observada",
            "suma_observada",
            "magnitud_media_observada",
            "total_configuraciones",
            "conteo_configuraciones_extremas",
            "p_value_exacto_cambios_signo",
            "p_value_t_bilateral_T4",
            "diferencia_absoluta_p",
            "diferencia_relativa_p_vs_t",
            "sd_de_las_medias_nulas",
            "p_value_aproximacion_normal",
        ],
        "valor": [
            n, media_obs, suma_obs, magnitud_obs, n_config, conteo_extremas,
            p_value_exacto, p_t, dif_abs, dif_rel, sd_nulas, p_normal_aprox,
        ],
    }
)
resumen.to_csv(CSV_RESUMEN, index=False, encoding="utf-8")

# ------------------------------------------------------------------ resultados.json
interpretacion = (
    f"Bajo H0 (media de las diferencias = 0) y el supuesto de intercambiabilidad del signo de cada d_i, "
    f"las 2^16 = 65536 configuraciones de signos son igualmente probables; el p-value exacto es la fracción "
    f"de configuraciones cuya media nula tiene magnitud >= |media observada| = {magnitud_obs:.6f}: "
    f"{conteo_extremas}/65536 = {p_value_exacto:.6f}. Es una probabilidad CONDICIONAL a H0 (frecuencia de "
    f"resultados al menos tan extremos si H0 fuera cierta), NO la probabilidad de que H0 sea verdadera. "
    f"A alpha = {ALPHA}, p_exacto > alpha: no se rechaza H0, la misma decisión que con la prueba t de T4 "
    f"(p = {p_t:.6f})."
)
por_que_no_intercambiables = (
    "La prueba t evalúa t = media/SE bajo una distribución t de Student con 15 grados de libertad y, para su "
    "calibración exacta, asume que las diferencias d_i provienen de una población normal (o apoya su p-value "
    "en el TCL). La referencia por cambios de signo no asume normalidad: solo requiere que, bajo H0, el signo "
    "de cada d_i sea intercambiable, y calibra el p-value con la distribución FINITA y exacta de las 65536 "
    "medias nulas. Al basarse en supuestos y distribuciones de referencia distintas, los dos procedimientos "
    "no son intercambiables; que aquí sus p-values queden cerca (diferencia absoluta = "
    f"{dif_abs:.3e}, relativa = {dif_rel:.2%}) es una coincidencia empírica (la referencia de signos es "
    "aproximadamente simétrica y campanuda), no una identidad matemática."
)

resultados = {
    "subtarea": "T5",
    "titulo": "Referencia nula exacta por cambios de signo: enumeración completa de las 2^16 configuraciones",
    "archivo_entrada": "data/model_errors.csv",
    "entrada_T4_usada": str(RUTA_T4),
    "n": n,
    "diferencias_por_tienda": diferencias_por_tienda,
    "media_observada": media_obs,
    "suma_observada": suma_obs,
    "magnitud_media_observada": magnitud_obs,
    "enumeracion": {
        "total_configuraciones": n_config,
        "formula_total": "2^16 = 65536",
        "esquema": "cada diferencia d_i se multiplica por +1 o -1; bit=0 -> +1, bit=1 -> -1; "
                   "la configuración 0 (todos +1) es la observada",
        "determinista": True,
        "sin_semilla_requerida": True,
    },
    "criterio_extremo": "|media nula| >= |media observada| (bilateral)",
    "tolerancia_punto_flotante": TOL,
    "conteo_extremas": conteo_extremas,
    "p_value_exacto": p_value_exacto,
    "p_value_exacto_como_fraccion": f"{conteo_extremas}/{n_config}",
    "desglose_colas": {
        "configuraciones_cola_positiva": cola_positiva,
        "configuraciones_cola_negativa": cola_negativa,
        "simetria_exacta_de_colas": bool(cola_positiva == cola_negativa),
        "configuraciones_con_magnitud_exacta_de_la_observada": conteo_misma_magnitud,
    },
    "verificaciones": {
        "configuracion_todos_mas1_reproduce_media_observada": check_obs,
        "configuracion_todos_menos1_reproduce_menos_media_observada": check_neg,
        "observada_incluida_en_extremas": observada_incluida,
        "verificacion_aritmetica_entera_dp": {
            "descripcion": "DP exacta sobre sumas escaladas x10 (d*10 enteros): cuenta configuraciones "
                           "con |suma| >= 67, equivalente a |media| >= |media observada|",
            "total_configuraciones_dp": total_dp,
            "conteo_extremas_dp": conteo_dp,
            "p_value_dp": conteo_dp / total_dp,
            "coincide_con_enumeracion_float": coincide_dp,
        },
    },
    "distribucion_nula": {
        "media_de_las_medias_nulas": float(np.mean(medias_nulas)),
        "sd_de_las_medias_nulas": sd_nulas,
        "minimo": float(np.min(medias_nulas)),
        "maximo": float(np.max(medias_nulas)),
        "p_value_aproximacion_normal_bilateral": p_normal_aprox,
        "nota": "La aproximación normal se incluye solo como contexto; el p-value reportado es el exacto "
                "por enumeración completa.",
    },
    "comparacion_con_prueba_t_T4": {
        "p_value_exacto_cambios_signo": p_value_exacto,
        "p_value_t_bilateral_T4": p_t,
        "fuente_p_value_t": fuente_p_t,
        "estadistico_t_T4": estadistico_t,
        "grados_de_libertad_T4": gl,
        "media_T4": media_t4,
        "coincide_media_con_T4": coincide_media,
        "diferencia_absoluta": dif_abs,
        "diferencia_relativa_respecto_t": dif_rel,
        "alpha": ALPHA,
        "misma_decision_alpha_0_05_no_rechazar_H0": misma_decision,
    },
    "diseno_y_supuestos": {
        "diseno": "pareado por store_id (16 tiendas), misma métrica de error en ambos modelos",
        "h0": "media de las diferencias d_i = 0",
        "alternativa": "bilateral",
        "supuestos_cambios_de_signo": "intercambiabilidad/simetría del signo de cada d_i bajo H0; "
                                      "no requiere normalidad",
        "supuestos_prueba_t": "diferencias iid de población normal (o TCL para n grande)",
    },
    "interpretacion": interpretacion,
    "por_que_no_son_intercambiables": por_que_no_intercambiables,
    "figura_guardada_en": FIGURA_PNG,
    "csv_resumen_guardado_en": CSV_RESUMEN,
    "resultados_guardados_en": "resultados.json",
}

with RUTA_RESULTADOS.open("w", encoding="utf-8") as fh:
    json.dump(resultados, fh, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------ Resumen
print("=" * 72)
print("T5 · Referencia nula exacta por cambios de signo")
print(f"  n = {n} diferencias pareadas | media observada = {media_obs:.6f}")
print(f"  Total de configuraciones de signos: 2^{n} = {n_config}")
print(f"  Configuraciones extremas (|media nula| >= {magnitud_obs:.6f}): {conteo_extremas}")
print(f"  p-value exacto = {conteo_extremas}/{n_config} = {p_value_exacto:.6f}")
print(f"  Verificación DP entera: total = {total_dp}, extremas = {conteo_dp}, coincide = {coincide_dp}")
print(f"  p-value t bilateral (T4, fuente: {fuente_p_t}) = {p_t:.6f}")
print(f"  |p_exacto - p_t| = {dif_abs:.6e} (relativa = {dif_rel:.2%}) | "
      f"misma decisión a alpha={ALPHA} (no rechazar H0): {misma_decision}")
print(f"  Salidas: resultados.json | {FIGURA_PNG} | {CSV_RESUMEN}")
print("=" * 72)
