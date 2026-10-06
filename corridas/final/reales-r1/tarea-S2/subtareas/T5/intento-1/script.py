#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5 - Referencia nula exacta por cambios de signo (enumeracion completa 2^16).

Enumera las 2^16 = 65536 configuraciones de signos de las 16 diferencias pareadas
(difference = error_a - error_b), calcula la media nula de cada configuracion,
cuenta cuantas tienen magnitud al menos tan extrema como la observada y reporta:
  - total de configuraciones (65536)
  - conteo de configuraciones extremas
  - p-value exacto = conteo / total
  - p-value de la prueba t de T4 (leido de entrada/T4/resultados.json, con
    recálculo de respaldo via scipy.stats.ttest_1samp) para el contraste.

Salidas: resultados.json (contrato), output/store_differences.csv y
output/store_differences.png (archivos exigidos por el enunciado).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy import stats

# Muy por debajo de la separacion entre medias nulas distintas (0.1/16 = 0.00625)
# y muy por encima del ruido de punto flotante (~1e-14): clasifica exactamente.
TOL = 1e-9


def cargar_diferencias():
    csv_path = Path("data/model_errors.csv")
    df = pd.read_csv(csv_path)
    requeridas = ["store_id", "error_a", "error_b"]
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en {csv_path}: {faltantes}")
    df = df[requeridas].copy()
    ea = df["error_a"].to_numpy(dtype=np.float64)
    eb = df["error_b"].to_numpy(dtype=np.float64)
    if not (np.isfinite(ea).all() and np.isfinite(eb).all()):
        raise ValueError("error_a y error_b deben ser numericos y finitos.")
    diffs = ea - eb
    n = int(diffs.size)
    if n != 16:
        raise ValueError(f"Se esperaban 16 pares (2^16 configuraciones); hay {n}.")
    return df, ea, eb, diffs, n


def main():
    # ------------------------------------------------------------------
    # 1) Datos y vector de diferencias pareadas
    # ------------------------------------------------------------------
    df, ea, eb, diffs, n = cargar_diferencias()
    store_ids = df["store_id"].astype(str).tolist()

    obs_mean = float(diffs.mean())
    obs_sum = float(diffs.sum())
    abs_obs = abs(obs_mean)
    n_ceros = int(np.sum(diffs == 0.0))

    # ------------------------------------------------------------------
    # 2) Enumeracion exhaustiva de las 2^16 configuraciones de signos
    #    bit k del indice -> signo de diffs[k]: bit 0 -> +1, bit 1 -> -1
    # ------------------------------------------------------------------
    total = 1 << n  # 65536
    idx = np.arange(total, dtype=np.int64)
    bits = (idx[:, None] >> np.arange(n, dtype=np.int64)) & 1
    signs = 1.0 - 2.0 * bits.astype(np.float64)  # (65536, 16) con +1/-1
    null_sums = signs @ diffs
    null_means = null_sums / float(n)
    abs_null = np.abs(null_means)
    m_max = float(abs_null.max())

    # ------------------------------------------------------------------
    # 3) Conteo de configuraciones extremas y p-value exacto
    # ------------------------------------------------------------------
    extreme_mask = abs_null >= abs_obs - TOL
    conteo_extremas = int(extreme_mask.sum())
    p_exacto = conteo_extremas / total
    conteo_estrictas = int(np.sum(abs_null > abs_obs + TOL))
    conteo_empatadas = conteo_extremas - conteo_estrictas

    # ------------------------------------------------------------------
    # 4) Verificacion independiente: DP de conteo subset-sum (escala x10)
    #    |sum| >= |obs_sum|  <=>  S10 <= (T-O)/2  o  S10 >= T-(T-O)/2
    #    con S10 = 10*sum(|d_i| del subconjunto volteado), T = 10*sum|d_i|
    # ------------------------------------------------------------------
    abs10 = [int(round(abs(d) * 10.0)) for d in diffs]
    total10 = int(sum(abs10))
    obs10 = int(round(abs_obs * n * 10.0))
    lim_inf = (total10 - obs10) // 2
    lim_sup = total10 - lim_inf
    dp = np.zeros(total10 + 1, dtype=np.int64)
    dp[0] = 1
    for a in abs10:
        nxt = dp.copy()
        nxt[a:] += dp[: total10 + 1 - a]
        dp = nxt
    n_le = int(dp[: lim_inf + 1].sum())
    n_ge = int(dp[lim_sup:].sum())
    extremas_dp = n_le + n_ge
    dp_coincide = bool(extremas_dp == conteo_extremas and int(dp.sum()) == total)

    # La configuracion observada (y su espejo) deben estar en la enumeracion
    idx_obs = 0
    for k in range(n):
        if diffs[k] < 0:
            idx_obs |= 1 << k
    media_en_obs = float(null_means[idx_obs])
    media_espejo = float(null_means[total - 1 - idx_obs])
    obs_presente = bool(
        abs(media_en_obs - obs_mean) <= 1e-12 and abs(media_espejo + obs_mean) <= 1e-12
    )

    # ------------------------------------------------------------------
    # 5) p-value de la prueba t (T4) para el contraste
    # ------------------------------------------------------------------
    t4_path = Path("entrada/T4/resultados.json")
    t_p, t_stat, t_gl, t4_media = None, None, None, None
    fuente_t = "recalculado con scipy.stats.ttest_1samp (T4 no disponible)"
    if t4_path.exists():
        try:
            t4 = json.loads(t4_path.read_text(encoding="utf-8"))
            pt = t4.get("prueba_t", {}) if isinstance(t4, dict) else {}
            v = t4.get("p_value_bilateral", pt.get("p_value")) if isinstance(t4, dict) else None
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                t_p = float(v)
                t_stat = t4.get("estadistico_t", pt.get("estadistico"))
                t_gl = t4.get("grados_de_libertad", pt.get("grados_de_libertad"))
                t4_media = t4.get("media")
                fuente_t = "entrada/T4/resultados.json"
        except (OSError, json.JSONDecodeError):
            t_p = None

    tt = stats.ttest_1samp(diffs, 0.0, alternative="two-sided")
    t_recalc = float(tt.pvalue)
    if t_p is None:
        t_p = t_recalc
        t_stat = float(tt.statistic)
        t_gl = int(n - 1)
    t_stat = float(t_stat) if t_stat is not None else float(tt.statistic)
    t_gl = int(t_gl) if t_gl is not None else int(n - 1)
    t_coincide = bool(abs(t_recalc - t_p) <= 1e-9)
    media_coincide = bool(t4_media is not None and abs(float(t4_media) - obs_mean) <= 1e-12)

    # ------------------------------------------------------------------
    # 6) Comparacion con la prueba t
    # ------------------------------------------------------------------
    dec_e = "rechazar H0" if p_exacto < 0.05 else "no rechazar H0"
    dec_t = "rechazar H0" if t_p < 0.05 else "no rechazar H0"
    dif_abs = p_exacto - t_p
    razon = (p_exacto / t_p) if t_p > 0 else None

    # ------------------------------------------------------------------
    # 7) Archivos exigidos: output/store_differences.csv y .png
    # ------------------------------------------------------------------
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    favors = np.where(diffs > 0, "B", np.where(diffs < 0, "A", "empate"))
    pd.DataFrame(
        {
            "store_id": store_ids,
            "error_a": ea,
            "error_b": eb,
            "difference": diffs,
            "favors": favors,
        }
    ).to_csv(out_dir / "store_differences.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 8.2))
    colores = np.where(diffs > 0, "#2a9d8f", "#e76f51")
    ax1.bar(range(n), diffs, color=colores, edgecolor="black", linewidth=0.5)
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.axhline(obs_mean, color="#264653", linestyle="--", linewidth=1.3,
                label=f"media observada = {obs_mean:.4f}")
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(store_ids, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("difference = error_a - error_b")
    ax1.set_title("T5 - Diferencias pareadas por tienda y referencia exacta por cambios de signo")
    ax1.legend(loc="lower right", fontsize=9)

    ax2.hist(null_means, bins=80, color="#a8dadc", edgecolor="#457b9d", linewidth=0.3)
    ax2.axvspan(-m_max, -abs_obs, color="#d62828", alpha=0.18)
    ax2.axvspan(abs_obs, m_max, color="#d62828", alpha=0.18)
    ax2.axvline(abs_obs, color="#d62828", linestyle="--", linewidth=1.4)
    ax2.axvline(-abs_obs, color="#d62828", linestyle="--", linewidth=1.4,
                label=f"±|media observada| = ±{abs_obs:.4f}")
    ax2.set_xlabel("media nula (2^16 configuraciones de signo)")
    ax2.set_ylabel("frecuencia")
    ax2.set_title(
        f"Referencia nula exacta: {conteo_extremas}/{total} extremas -> "
        f"p exacto = {p_exacto:.6f} | p t (T4) = {t_p:.6f}"
    )
    ax2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig("output/store_differences.png", dpi=120)
    fig.savefig("store_differences.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 8) Contrato: resultados.json
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T5_referencia_exacta_cambios_de_signo",
        "fuente_datos": "data/model_errors.csv",
        "definicion_diferencia": "difference = error_a - error_b (pareado por store_id)",
        "metodo": (
            "Enumeracion exhaustiva de las 2^16 configuraciones de signos de las 16 "
            "diferencias; para cada configuracion se calcula la media nula y se compara "
            "su magnitud con la de la media observada (bilateral)."
        ),
        "n_diferencias": n,
        "media_observada": obs_mean,
        "suma_observada": obs_sum,
        "total_configuraciones": int(total),
        "total_configuraciones_formula": "2^16 = 65536",
        "conteo_configuraciones_extremas": conteo_extremas,
        "criterio_extremo": "|media_nula| >= |media_observada| (empates incluidos)",
        "p_value_exacto": p_exacto,
        "p_value_exacto_formula": "conteo_configuraciones_extremas / total_configuraciones",
        "p_value_prueba_t_T4": t_p,
        "cifras_para_preguntas": {
            "total_configuraciones": int(total),
            "conteo_extremas": conteo_extremas,
            "p_value_exacto": p_exacto,
            "p_value_t_T4": t_p,
        },
        "comparacion_con_prueba_t": {
            "p_value_exacto": p_exacto,
            "p_value_t_T4": t_p,
            "fuente_p_value_t": fuente_t,
            "diferencia_absoluta": dif_abs,
            "razon_exacto_sobre_t": razon,
            "decision_al_5pct_exacto": dec_e,
            "decision_al_5pct_t": dec_t,
            "misma_decision_al_5pct": bool((p_exacto < 0.05) == (t_p < 0.05)),
        },
        "prueba_t_contexto": {
            "estadistico_t": t_stat,
            "grados_de_libertad": t_gl,
            "popmean_nulo": 0.0,
            "alternativa": "two-sided",
        },
        "diagnostico": {
            "tolerancia_comparacion": TOL,
            "conteo_estrictamente_extremas": conteo_estrictas,
            "conteo_empatadas_con_observada": conteo_empatadas,
            "max_abs_media_nula": m_max,
            "media_de_las_medias_nulas": float(null_means.mean()),
            "cuantiles_medias_nulas": {
                "q025": float(np.quantile(null_means, 0.025)),
                "q50": float(np.quantile(null_means, 0.5)),
                "q975": float(np.quantile(null_means, 0.975)),
            },
            "indice_configuracion_observada": int(idx_obs),
            "media_nula_en_configuracion_observada": media_en_obs,
            "media_nula_en_configuracion_espejo": media_espejo,
            "configuracion_observada_presente": obs_presente,
            "fraccion_no_extremas": 1.0 - p_exacto,
            "diferencias_cero_empates": n_ceros,
        },
        "verificacion_independiente": {
            "metodo": "DP de conteo subset-sum sobre |diferencias| x 10 (enteros exactos)",
            "umbral_S10_le": int(lim_inf),
            "umbral_S10_ge": int(lim_sup),
            "n_subsets_le": n_le,
            "n_subsets_ge": n_ge,
            "extremas_dp": extremas_dp,
            "total_subsets_dp": int(dp.sum()),
            "coincide_con_enumeracion": dp_coincide,
            "simetria_n_le_igual_n_ge": bool(n_le == n_ge),
        },
        "verificacion_con_T4": {
            "media_T4": (float(t4_media) if t4_media is not None else None),
            "coincide_media_con_T4": media_coincide,
            "p_value_t_recalculado_scipy": t_recalc,
            "coincide_p_value_t": t_coincide,
        },
        "nota_p_value_condicional": (
            "El p-value es una probabilidad condicional a la hipotesis nula: la probabilidad "
            "de que, si los signos de las diferencias fueran puro azar (H0: media de las "
            "diferencias = 0, signos intercambiables), se obtuviera una media con magnitud "
            "al menos tan extrema como la observada."
        ),
        "nota_no_intercambiabilidad": (
            "La prueba t compara el estadistico con una referencia continua t de Student que "
            "supone diferencias normales (o n grande) y estima la varianza; la prueba exacta "
            "por cambios de signo no supone normalidad: su referencia nula es discreta (65536 "
            "medias) y se construye re-asignando signos a las propias diferencias observadas "
            "(intercambiabilidad/simetria). No son intercambiables porque dependen de supuestos "
            "distintos y generan referencias distintas (continua vs discreta); aqui coinciden "
            "numericamente (p exacto = {:.6f} vs p t = {:.6f}) porque con n=16 y diferencias "
            "razonablemente simetricas la aproximacion t es buena, pero podrian divergir con "
            "datos asimetricos, atipicos o muestras pequenas."
        ).format(p_exacto, t_p),
        "archivos_generados": {
            "csv": "output/store_differences.csv",
            "figura": "output/store_differences.png",
            "figura_copia_carpeta_actual": "store_differences.png",
            "resultados_json": "resultados.json",
        },
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 9) Resumen
    # ------------------------------------------------------------------
    print("=" * 66)
    print("T5 - Referencia exacta por cambios de signo (enumeracion 2^16)")
    print("=" * 66)
    print(f"n diferencias pareadas              : {n}")
    print(f"media observada                     : {obs_mean:.6f}")
    print(f"total de configuraciones            : {total}")
    print(f"configuraciones extremas (>=|obs|)  : {conteo_extremas}")
    print(f"p-value exacto (conteo/total)       : {p_exacto:.6f}")
    print(f"p-value prueba t (T4, bilateral)    : {t_p:.6f}")
    print(f"|p exacto - p t|                    : {abs(dif_abs):.6f}")
    print(f"decision al 5% (exacto | t)         : {dec_e} | {dec_t}")
    print(f"verificacion DP coincide            : {dp_coincide} (extremas_dp={extremas_dp})")
    print(f"configuracion observada presente    : {obs_presente}")
    print("Archivos: resultados.json, output/store_differences.csv, output/store_differences.png")


if __name__ == "__main__":
    main()
