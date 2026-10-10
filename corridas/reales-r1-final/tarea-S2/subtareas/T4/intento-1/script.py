#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T4 - Calcular el efecto y su incertidumbre sobre el vector de diferencias pareadas.

Sobre difference = error_a - error_b (pareado por store_id):
  * n, media, desviacion estandar muestral (ddof=1) y SE = sd / sqrt(n).
  * scipy.stats: prueba t de una muestra bilateral (H0: media = 0) e
    intervalo t bilateral del 95% para la media pareada,
    conservando grados de libertad, estadistico, p-value e intervalo.

Entradas : data/model_errors.csv (respaldo: entrada/T2/resultados.json)
Salidas  : output/store_differences.csv, output/store_differences.png, resultados.json
Nota     : tarea determinista; no se usan semillas ni particiones train/test.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_FUENTE = Path("data/model_errors.csv")
RESPALDO_T2 = Path("entrada/T2/resultados.json")
DIR_SALIDA = Path("output")
RUTA_CSV = DIR_SALIDA / "store_differences.csv"
RUTA_PNG = DIR_SALIDA / "store_differences.png"
RUTA_RESULTADOS = Path("resultados.json")

COLS = ["store_id", "error_a", "error_b"]
CONFIANZA = 0.95


def cargar_tabla_pareada():
    """Carga (store_id, error_a, error_b) desde el CSV; respaldo: resultados de T2."""
    if CSV_FUENTE.exists():
        df = pd.read_csv(CSV_FUENTE)
        origen = str(CSV_FUENTE)
    elif RESPALDO_T2.exists():
        with open(RESPALDO_T2, encoding="utf-8") as f:
            previo = json.load(f)
        df = pd.DataFrame(previo["tabla_pareada"])
        origen = str(RESPALDO_T2)
    else:
        raise FileNotFoundError(
            f"No se encontro ni '{CSV_FUENTE}' ni el respaldo '{RESPALDO_T2}'."
        )

    faltan = [c for c in COLS if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas esperadas en {origen}: {faltan}")
    df = df[COLS].copy()

    if df["store_id"].isna().any() or df["store_id"].duplicated().any():
        raise ValueError("store_id debe ser unico y no nulo (pareo verificado por tienda).")
    for c in ("error_a", "error_b"):
        valores = pd.to_numeric(df[c], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(valores).all():
            raise ValueError(f"La columna '{c}' contiene valores no finitos.")
        if (valores < 0).any():
            raise ValueError(f"La columna '{c}' contiene valores negativos.")
        df[c] = valores
    return df, origen


def etiqueta_favors(d):
    """Convencion de T2: 'B' si difference>0, 'A' si difference<0, 'empate' si 0."""
    if d > 0:
        return "B"
    if d < 0:
        return "A"
    return "empate"


def main():
    # ------------------------------------------------------------------ datos
    df, origen = cargar_tabla_pareada()
    df["difference"] = df["error_a"] - df["error_b"]
    df["favors"] = df["difference"].map(etiqueta_favors)
    diffs = df["difference"].to_numpy(dtype=float)

    # ------------------------------------------------- descriptivos del efecto
    n = int(diffs.size)
    if n < 2:
        raise ValueError("Se necesitan al menos 2 diferencias para la sd muestral (ddof=1).")
    media = float(np.mean(diffs))
    sd_muestral = float(np.std(diffs, ddof=1))
    se = float(sd_muestral / np.sqrt(n))

    # --------------------------------------- prueba t de una muestra bilateral
    res = stats.ttest_1samp(diffs, popmean=0.0, alternative="two-sided")
    t_stat = float(res.statistic if hasattr(res, "statistic") else res[0])
    p_value = float(res.pvalue if hasattr(res, "pvalue") else res[1])
    dfree = int(n - 1)

    # ------------------------------------ IC t bilateral del 95% (media pareada)
    if se > 0.0:
        ci_low, ci_high = stats.t.interval(CONFIANZA, df=dfree, loc=media, scale=se)
        ci_low, ci_high = float(ci_low), float(ci_high)
    else:
        ci_low = ci_high = media
    t_crit = float(stats.t.ppf(0.5 + CONFIANZA / 2.0, dfree))
    semiancho = float((ci_high - ci_low) / 2.0)
    contiene_cero = bool(ci_low <= 0.0 <= ci_high)

    # --------------------------------------------------------- verificaciones
    t_manual = media / se
    p_manual = float(2.0 * stats.t.sf(abs(t_stat), dfree))
    verificacion = {
        "t_igual_a_media_sobre_se": bool(abs(t_stat - t_manual) < 1e-9),
        "p_igual_a_2_sf_bilateral": bool(abs(p_value - p_manual) < 1e-9),
        "ic_igual_a_media_pm_tcrit_por_se": bool(
            abs(ci_low - (media - t_crit * se)) < 1e-9
            and abs(ci_high - (media + t_crit * se)) < 1e-9
        ),
        "gl_igual_a_n_menos_1": bool(dfree == n - 1 and dfree == 15),
    }

    # ------------------------------------------------------------- entregables
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUTA_CSV, index=False)

    colores = np.where(diffs > 0, "#2a9d8f", np.where(diffs < 0, "#e76f51", "#8d99ae"))
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.5, 5.0), gridspec_kw={"width_ratios": [2.3, 1.0]}
    )
    ax1.bar(df["store_id"], diffs, color=colores, edgecolor="black", linewidth=0.5)
    ax1.axhline(0.0, color="black", lw=1.0)
    ax1.axhline(media, color="#264653", ls="--", lw=1.6, label=f"media = {media:.4f}")
    ax1.axhspan(ci_low, ci_high, color="#264653", alpha=0.12,
                label=f"IC t 95% = [{ci_low:.4f}, {ci_high:.4f}]")
    ax1.set_title("Diferencia pareada por tienda (error_a - error_b)")
    ax1.set_xlabel("store_id")
    ax1.set_ylabel("Diferencia (unidades)")
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(loc="lower right", fontsize=9)

    ax2.errorbar([0], [media], yerr=[[media - ci_low], [ci_high - media]],
                 fmt="o", color="#264653", ecolor="#264653", capsize=6, markersize=7)
    ax2.axhline(0.0, color="black", lw=1.0, ls=":")
    ax2.set_xlim(-1, 1)
    ax2.set_xticks([])
    ax2.set_title(f"Media pareada e IC t 95%\nt = {t_stat:.4f}, gl = {dfree}, p = {p_value:.4f}")
    ax2.set_ylabel("Diferencia media (unidades)")
    fig.suptitle("T4: Efecto pareado y su incertidumbre (n = 16)", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(RUTA_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------- resultados.json
    if p_value < 0.05:
        decision = (f"Con p = {p_value:.6f} < 0.05 se rechaza H0: media de diferencias = 0 "
                    f"al nivel del 5% (IC t 95%: [{ci_low:.6f}, {ci_high:.6f}]).")
    else:
        decision = (f"Con p = {p_value:.6f} >= 0.05 y un IC t 95% de "
                    f"[{ci_low:.6f}, {ci_high:.6f}] (contiene 0: {contiene_cero}), "
                    "no se rechaza H0: media de diferencias = 0 al nivel del 5%.")
    nota_dispersion = (
        f"La sd muestral ({sd_muestral:.6f}) describe la dispersion de las diferencias "
        f"entre tiendas; el error estandar ({se:.6f}) es la incertidumbre de la media "
        f"pareada y se obtiene como sd/sqrt(n) = {sd_muestral:.6f}/{np.sqrt(n):.6f}."
    )

    resultados = {
        "subtarea": "T4_efecto_e_incertidumbre_diferencias_pareadas",
        "fuente_datos": origen,
        "definicion_diferencia": "difference = error_a - error_b (pareado por store_id)",
        # --- cifras pedidas (preguntas 13, 14 y 15) ---
        "n": n,
        "media": media,
        "sample_sd": sd_muestral,
        "standard_error": se,
        "estadistico_t": t_stat,
        "grados_de_libertad": dfree,
        "p_value_bilateral": p_value,
        "ic95_t_limite_inferior": ci_low,
        "ic95_t_limite_superior": ci_high,
        "ic95_t_nivel": CONFIANZA,
        "ic95_t_semiancho": semiancho,
        "t_critico_95": t_crit,
        "ic95_contiene_cero": contiene_cero,
        "cifras_para_preguntas": {
            "p13_n_media_sd_se": {
                "n": n, "media": media, "sample_sd": sd_muestral, "standard_error": se,
            },
            "p14_prueba_t_bilateral": {
                "estadistico_t": t_stat, "grados_de_libertad": dfree, "p_value": p_value,
            },
            "p15_ic95_t_pareado": {
                "limite_inferior": ci_low, "limite_superior": ci_high, "nivel": CONFIANZA,
            },
        },
        # --- detalle del procedimiento ---
        "prueba_t": {
            "metodo": "scipy.stats.ttest_1samp",
            "popmean_nulo": 0.0,
            "alternativa": "two-sided",
            "estadistico": t_stat,
            "grados_de_libertad": dfree,
            "p_value": p_value,
        },
        "intervalo_t": {
            "tipo": "t bilateral para la media pareada",
            "nivel": CONFIANZA,
            "limite_inferior": ci_low,
            "limite_superior": ci_high,
            "t_critico": t_crit,
            "semiancho": semiancho,
            "contiene_cero": contiene_cero,
        },
        "descriptivos": {
            "n": n,
            "media": media,
            "sample_sd_ddof1": sd_muestral,
            "standard_error": se,
            "formula_se": "SE = sd / sqrt(n)",
            "suma_diferencias": float(np.sum(diffs)),
        },
        "vector_diferencias": {
            "store_ids": df["store_id"].tolist(),
            "diferencias": diffs.tolist(),
        },
        "interpretacion": {
            "diferencia_media_unidades": media,
            "dispersion_entre_tiendas_sd": sd_muestral,
            "error_estandar_de_la_media": se,
            "nota_dispersion_vs_se": nota_dispersion,
            "decision_al_5pct": decision,
        },
        "verificacion": verificacion,
        "archivos_generados": {
            "csv": str(RUTA_CSV),
            "figura": str(RUTA_PNG),
            "resultados_json": str(RUTA_RESULTADOS),
        },
    }
    with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ---------------------------------------------------------------- resumen
    print("=== T4: Efecto e incertidumbre de las diferencias pareadas ===")
    print(f"Fuente: {origen}")
    print(f"n = {n}")
    print(f"media = {media:.6f}")
    print(f"sd muestral (ddof=1) = {sd_muestral:.6f}")
    print(f"SE = sd/sqrt(n) = {se:.6f}")
    print(f"Prueba t bilateral: t = {t_stat:.6f}, gl = {dfree}, p = {p_value:.6f}")
    print(f"IC t 95%: [{ci_low:.6f}, {ci_high:.6f}] (t_critico = {t_crit:.6f})")
    print(f"El IC contiene 0: {contiene_cero}")
    print(f"Verificacion: {verificacion}")
    print(f"Archivos generados: {RUTA_CSV}, {RUTA_PNG}, {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
