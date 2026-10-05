#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T4: Estadísticos del vector de diferencias (error_a - error_b), prueba t de
una muestra bilateral e intervalo t bilateral del 95 % para la media pareada.

Entradas : data/model_errors.csv (store_id, error_a, error_b)
           entrada/T2/resultados.json (contraste opcional de consistencia, si existe)
Salidas  : resultados.json, t4_resumen_estadistico.csv, t4_media_ic_diferencias.png
"""

import json
import math

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUTA_CSV = "data/model_errors.csv"
RUTA_T2 = "entrada/T2/resultados.json"
NIVEL = 0.95
ALPHA = 0.05


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Carga y validación ligera de los datos pareados
    # ------------------------------------------------------------------
    df = pd.read_csv(RUTA_CSV)

    faltantes = [c for c in ("store_id", "error_a", "error_b") if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan las columnas {faltantes} en {RUTA_CSV}")

    for col in ("error_a", "error_b"):
        valores = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(valores)):
            raise ValueError(f"La columna '{col}' contiene valores no numéricos o no finitos")
        if np.any(valores < 0):
            raise ValueError(f"La columna '{col}' contiene valores negativos")
        df[col] = valores

    if df["store_id"].isna().any() or df["store_id"].duplicated().any():
        raise ValueError("store_id debe estar presente y ser único por tienda")

    # ------------------------------------------------------------------
    # 2) Vector de diferencias d_i = error_a - error_b (pareado por tienda)
    # ------------------------------------------------------------------
    df["difference"] = df["error_a"] - df["error_b"]
    d = df["difference"].to_numpy(dtype=float)
    store_ids = df["store_id"].astype(str).tolist()
    n = int(d.size)

    # ------------------------------------------------------------------
    # 3) Estadísticos descriptivos: n, mean, sample_sd (ddof=1), SE = sd/sqrt(n)
    # ------------------------------------------------------------------
    mean = float(np.mean(d))
    sample_sd = float(np.std(d, ddof=1))                 # desviación muestral
    standard_error = float(sample_sd / math.sqrt(n))     # SE = sd / sqrt(n)

    # Verificación explícita de SE = sd / sqrt(n)
    se_recalculado = float(np.std(d, ddof=1) / np.sqrt(n))
    se_dif = abs(standard_error - se_recalculado)
    se_verificado = bool(se_dif <= 1e-12 * max(1.0, abs(standard_error)))

    # ------------------------------------------------------------------
    # 4) Prueba t de una muestra bilateral sobre las diferencias (t pareada)
    # ------------------------------------------------------------------
    t_res = stats.ttest_1samp(d, popmean=0.0, alternative="two-sided")
    t_stat = float(t_res.statistic)
    p_value = float(t_res.pvalue)
    gl = int(n - 1)

    # Verificación: t = mean / SE
    t_desde_se = mean / standard_error
    t_verificado = bool(math.isclose(t_stat, t_desde_se, rel_tol=1e-12, abs_tol=1e-12))

    # ------------------------------------------------------------------
    # 5) Intervalo t bilateral del 95 % para la media pareada
    # ------------------------------------------------------------------
    ci_low, ci_high = stats.t.interval(NIVEL, df=gl, loc=mean, scale=standard_error)
    ci_low, ci_high = float(ci_low), float(ci_high)
    t_critico = float(stats.t.ppf(0.5 + NIVEL / 2.0, gl))
    margen = float(t_critico * standard_error)

    ic_manual_ok = bool(
        math.isclose(ci_low, mean - margen, abs_tol=1e-12)
        and math.isclose(ci_high, mean + margen, abs_tol=1e-12)
    )

    ci_scipy_check = None
    if hasattr(t_res, "confidence_interval"):
        ci_obj = t_res.confidence_interval(confidence_level=NIVEL)
        ci_scipy_check = [float(ci_obj.low), float(ci_obj.high)]

    contiene_cero = bool(ci_low <= 0.0 <= ci_high)
    rechaza_h0 = bool(p_value < ALPHA)

    # ------------------------------------------------------------------
    # 6) Contraste opcional de consistencia con la subtarea T2
    # ------------------------------------------------------------------
    coincide_t2 = None
    try:
        with open(RUTA_T2, "r", encoding="utf-8") as fh:
            t2 = json.load(fh)
        tabla_t2 = sorted(t2.get("tabla_pareada", []), key=lambda r: str(r["store_id"]))
        if tabla_t2:
            d_t2 = np.array([float(r["difference"]) for r in tabla_t2])
            orden = np.argsort(np.array(store_ids))
            coincide_t2 = bool(
                d_t2.shape == d[orden].shape
                and np.allclose(d_t2, d[orden], rtol=0.0, atol=1e-9)
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        coincide_t2 = None

    # ------------------------------------------------------------------
    # 7) Figura: diferencias por tienda + distinción sd (dispersión) vs SE (precisión)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    ax = axes[0]
    colores = ["#d62728" if v < 0 else "#1f77b4" for v in d]
    ax.bar(range(n), d, color=colores)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(mean, color="#2ca02c", linewidth=1.6, label=f"media = {mean:.4f}")
    ax.axhspan(ci_low, ci_high, color="#2ca02c", alpha=0.15,
               label=f"IC t 95% = [{ci_low:.4f}, {ci_high:.4f}]")
    ax.set_xticks(range(n))
    ax.set_xticklabels(store_ids, rotation=90, fontsize=8)
    ax.set_ylabel("Diferencia d_i = error_a - error_b")
    ax.set_title("Diferencias por tienda y media pareada")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    ax.errorbar([0], [mean], yerr=[sample_sd], fmt="o", capsize=6, color="#1f77b4",
                label=f"media ± sd = {sample_sd:.4f} (dispersión entre tiendas)")
    ax.errorbar([1], [mean], yerr=[standard_error], fmt="o", capsize=6, color="#ff7f0e",
                label=f"media ± SE = {standard_error:.4f} (error estándar de la media)")
    ax.errorbar([2], [mean], yerr=[[mean - ci_low], [ci_high - mean]], fmt="o",
                capsize=6, color="#2ca02c", label="IC t 95% para la media")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["± sd", "± SE", "IC 95%"])
    ax.set_ylabel("Unidades de diferencia")
    ax.set_title("Dispersión entre tiendas vs. precisión de la media")
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig("t4_media_ic_diferencias.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 8) CSV de resumen
    # ------------------------------------------------------------------
    resumen = pd.DataFrame({
        "metrica": ["n", "mean", "sample_sd", "standard_error",
                    "t_statistic", "degrees_of_freedom", "p_value_two_sided",
                    "ci95_low", "ci95_high"],
        "valor": [n, mean, sample_sd, standard_error, t_stat, gl, p_value, ci_low, ci_high],
    })
    resumen.to_csv("t4_resumen_estadistico.csv", index=False)

    # ------------------------------------------------------------------
    # 9) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    if mean > 0:
        sentido = f"el modelo A comete {mean:.4f} unidades MÁS de error que el modelo B (favorece a B)"
    elif mean < 0:
        sentido = f"el modelo A comete {abs(mean):.4f} unidades MENOS de error que el modelo B (favorece a A)"
    else:
        sentido = "no hay diferencia media entre los modelos"

    decision = "se rechaza H0" if rechaza_h0 else "no se rechaza H0"

    resultados = {
        "subtarea": "T4",
        "archivo_entrada": RUTA_CSV,
        "n": n,
        "n_esperado_enunciado": 16,
        "n_coincide_con_enunciado": bool(n == 16),
        "mean": mean,
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "se_verificacion": {
            "formula": "SE = sample_sd / sqrt(n)",
            "se_recalculado": se_recalculado,
            "diferencia_absoluta": se_dif,
            "verificado": se_verificado,
        },
        "prueba_t": {
            "descripcion": ("Prueba t de una muestra bilateral sobre el vector de diferencias "
                            "d_i = error_a - error_b (equivalente a la t pareada)"),
            "hipotesis_nula": "media de las diferencias = 0",
            "alternativa": "bilateral",
            "estadistico_t": t_stat,
            "grados_de_libertad": gl,
            "p_value_bilateral": p_value,
            "alpha": ALPHA,
            "rechaza_H0": rechaza_h0,
            "t_verificado_como_mean_over_se": t_verificado,
        },
        "intervalo_confianza": {
            "nivel": NIVEL,
            "tipo": "Intervalo t bilateral del 95% para la media pareada",
            "grados_de_libertad": gl,
            "t_critico": t_critico,
            "margen_de_error": margen,
            "limite_inferior": ci_low,
            "limite_superior": ci_high,
            "contiene_cero": contiene_cero,
            "coincide_con_calculo_manual": ic_manual_ok,
            "scipy_confidence_interval_check": ci_scipy_check,
        },
        "dispersion_vs_precision": {
            "sample_sd_dispersion_entre_tiendas": sample_sd,
            "standard_error_precision_de_la_media": standard_error,
            "razon_sd_sobre_se": float(sample_sd / standard_error),
            "raiz_de_n": float(math.sqrt(n)),
            "nota": ("sample_sd mide la variabilidad de las diferencias entre tiendas; "
                     "standard_error = sd/sqrt(n) mide la incertidumbre de la media muestral. "
                     "Son conceptos distintos: el SE es sqrt(n) veces menor que la sd."),
        },
        "diferencia_media_unidades": mean,
        "diferencias_por_tienda": [
            {"store_id": sid,
             "error_a": float(a),
             "error_b": float(b),
             "difference": float(dd)}
            for sid, a, b, dd in zip(store_ids, df["error_a"], df["error_b"], d)
        ],
        "coincide_con_T2": coincide_t2,
        "interpretacion": (
            f"Con n = {n} tiendas, la diferencia media pareada es {mean:.4f} unidades "
            f"(error_a - error_b): {sentido}. La desviación muestral ({sample_sd:.4f}) describe "
            f"la dispersión de las diferencias entre tiendas, mientras que el error estándar "
            f"({standard_error:.4f} = {sample_sd:.4f}/sqrt({n})) describe la precisión de la media "
            f"muestral; el SE es {sample_sd / standard_error:.1f} veces menor que la sd al promediar "
            f"{n} tiendas. La prueba t bilateral (t = {t_stat:.4f}, gl = {gl}) da p = {p_value:.4f}; "
            f"a alpha = {ALPHA}, {decision}. El IC t del 95% es [{ci_low:.4f}, {ci_high:.4f}] "
            f"({'contiene' if contiene_cero else 'no contiene'} 0), coherente con la decisión de la prueba."
        ),
        "figura_guardada_en": "t4_media_ic_diferencias.png",
        "csv_resumen_guardado_en": "t4_resumen_estadistico.csv",
    }

    with open("resultados.json", "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 10) Resumen por consola
    # ------------------------------------------------------------------
    print("=== T4: Estadísticos de las diferencias y prueba t pareada ===")
    print(f"n                = {n}")
    print(f"mean             = {mean:.6f}")
    print(f"sample_sd (ddof=1) = {sample_sd:.6f}")
    print(f"standard_error   = {standard_error:.6f}  (SE = sd/sqrt(n) verificado: {se_verificado})")
    print(f"t                = {t_stat:.6f}")
    print(f"gl               = {gl}")
    print(f"p (bilateral)    = {p_value:.6f}")
    print(f"IC t 95%         = [{ci_low:.6f}, {ci_high:.6f}]  (contiene 0: {contiene_cero})")
    print(f"Decisión alpha={ALPHA}: {decision}")
    print(f"sd (dispersión entre tiendas) = {sample_sd:.6f}  vs  SE (precisión de la media) = {standard_error:.6f}")
    if coincide_t2 is not None:
        print(f"Consistencia con T2 (vector de diferencias): {coincide_t2}")
    print("Archivos: resultados.json | t4_resumen_estadistico.csv | t4_media_ic_diferencias.png")


if __name__ == "__main__":
    main()
