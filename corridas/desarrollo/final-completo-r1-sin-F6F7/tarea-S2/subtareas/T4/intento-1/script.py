#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T4 — Calcular el efecto y su incertidumbre sobre las diferencias pareadas.

Sobre el vector de diferencias (error_a - error_b) calcula exactamente:
  - n, media, desviación estándar MUESTRAL (ddof=1) y error estándar SE = sd / sqrt(n)
  - prueba t de una muestra bilateral (scipy.stats.ttest_1samp sobre las diferencias)
  - intervalo t bilateral del 95 % para la media pareada (scipy.stats.t.interval)
Conserva grados de libertad, estadístico, p-value e intervalo, y verifica
SE = sd/sqrt(n) contra scipy.stats.sem y el IC contra el cuantil t.

Salidas:
  - resultados.json (contrato de la subtarea)
  - output/store_differences.csv  (archivo exigido por el enunciado)
  - output/store_differences.png  (archivo exigido por el enunciado)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy import stats


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Carga y validación mínima de los datos de la tarea
    # ------------------------------------------------------------------
    csv_path = Path("data/model_errors.csv")
    if not csv_path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = ["store_id", "error_a", "error_b"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias {missing} en {csv_path}")

    store_ids = df["store_id"].astype(str)
    if store_ids.isna().any() or store_ids.duplicated().any():
        raise ValueError("store_id debe estar presente y ser único en cada fila")

    a = df["error_a"].to_numpy(dtype=float)
    b = df["error_b"].to_numpy(dtype=float)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("error_a y error_b deben ser numéricos y finitos")

    # ------------------------------------------------------------------
    # 2) Vector de diferencias pareadas (emparejamiento por fila/store_id)
    # ------------------------------------------------------------------
    diffs = a - b
    n = int(diffs.size)

    # ------------------------------------------------------------------
    # 3) Estadísticos exactos: n, media, sd muestral y SE = sd / sqrt(n)
    # ------------------------------------------------------------------
    mean_diff = float(np.mean(diffs))
    sample_sd = float(np.std(diffs, ddof=1))       # desviación MUESTRAL (ddof=1)
    se_formula = float(sample_sd / np.sqrt(n))     # SE = sd / sqrt(n)
    se_scipy = float(stats.sem(diffs, ddof=1))     # verificación cruzada con scipy
    se_abs_diff = abs(se_formula - se_scipy)
    se_verified = bool(se_abs_diff < 1e-12)

    # ------------------------------------------------------------------
    # 4) Prueba t de una muestra bilateral sobre las diferencias (scipy.stats)
    # ------------------------------------------------------------------
    res = stats.ttest_1samp(diffs, popmean=0.0)    # bilateral por defecto
    t_stat = float(getattr(res, "statistic", res[0]))
    p_two_sided = float(getattr(res, "pvalue", res[1]))
    dof = n - 1
    dof_scipy = float(getattr(res, "df", dof))     # scipy >= 1.11 expone df

    # Comprobación de coherencia con la prueba t pareada clásica
    res_rel = stats.ttest_rel(a, b)
    paired_consistent = bool(
        np.isclose(float(res_rel.statistic), t_stat, rtol=0, atol=1e-12)
        and np.isclose(float(res_rel.pvalue), p_two_sided, rtol=0, atol=1e-12)
    )

    # ------------------------------------------------------------------
    # 5) Intervalo t bilateral del 95 % para la media pareada
    # ------------------------------------------------------------------
    conf = 0.95
    try:
        ci_low, ci_high = stats.t.interval(
            confidence=conf, df=dof, loc=mean_diff, scale=se_formula
        )
    except TypeError:  # versiones antiguas de scipy usan `alpha`
        ci_low, ci_high = stats.t.interval(
            alpha=conf, df=dof, loc=mean_diff, scale=se_formula
        )
    ci_low, ci_high = float(ci_low), float(ci_high)

    # Verificación manual con el cuantil t (0.975, gl = n-1)
    t_crit = float(stats.t.ppf(1.0 - (1.0 - conf) / 2.0, dof))
    ci_low_manual = float(mean_diff - t_crit * se_formula)
    ci_high_manual = float(mean_diff + t_crit * se_formula)
    ci_verified = bool(
        np.isclose(ci_low, ci_low_manual, rtol=0, atol=1e-10)
        and np.isclose(ci_high, ci_high_manual, rtol=0, atol=1e-10)
    )

    # ------------------------------------------------------------------
    # 6) Archivos exigidos por el enunciado: output/store_differences.csv y .png
    # ------------------------------------------------------------------
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    favors = ["B" if d > 0 else ("A" if d < 0 else "tie") for d in diffs]
    records = [
        {
            "store_id": str(s),
            "error_a": float(x),
            "error_b": float(y),
            "difference": float(d),
            "favors": fv,
        }
        for s, x, y, d, fv in zip(store_ids, a, b, diffs, favors)
    ]
    table = pd.DataFrame(
        records, columns=["store_id", "error_a", "error_b", "difference", "favors"]
    )
    csv_out = out_dir / "store_differences.csv"
    table.to_csv(csv_out, index=False)

    # Figura: diferencias por tienda + media e IC t 95 %
    labels = [r["store_id"] for r in records]
    colors = ["#1f77b4" if r["difference"] > 0 else "#d62728" for r in records]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [3, 1]}
    )
    ax1.bar(labels, diffs, color=colors, edgecolor="black", linewidth=0.4)
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.axhline(
        mean_diff, color="green", linewidth=1.6, label=f"Media = {mean_diff:.4f}"
    )
    ax1.axhspan(
        ci_low,
        ci_high,
        color="green",
        alpha=0.15,
        label=f"IC t 95 % = [{ci_low:.4f}, {ci_high:.4f}]",
    )
    ax1.set_xlabel("Tienda (store_id)")
    ax1.set_ylabel("Diferencia (error_a − error_b)")
    ax1.set_title(
        f"Diferencias pareadas por tienda (n = {n})\n"
        f"t({dof}) = {t_stat:.4f} · p bilateral = {p_two_sided:.4f}"
    )
    ax1.tick_params(axis="x", rotation=90)
    ax1.legend(loc="lower right", fontsize=9)

    ax2.errorbar(
        [0],
        [mean_diff],
        yerr=[[mean_diff - ci_low], [ci_high - mean_diff]],
        fmt="o",
        color="green",
        ecolor="green",
        capsize=8,
        markersize=8,
    )
    ax2.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlim(-1, 1)
    ax2.set_xticks([])
    ax2.set_ylabel("Media de las diferencias")
    ax2.set_title("Media ± IC t 95 %")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig("output/store_differences.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 7) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    rejects_h0 = bool(p_two_sided < 0.05)
    ci_contains_zero = bool(ci_low <= 0.0 <= ci_high)
    decision = (
        "Se rechaza H0 al 5 %: la diferencia media es significativa."
        if rejects_h0
        else "No se rechaza H0 al 5 %: la diferencia media no es significativa."
    )

    results = {
        "subtask": "T4",
        "description": (
            "Efecto e incertidumbre de las diferencias pareadas: n, media, sd muestral, "
            "error estándar (SE = sd/sqrt(n)), prueba t bilateral e IC t del 95 %"
        ),
        "csv_source": "data/model_errors.csv",
        "n": n,
        "n_expected": 16,
        "n_matches_expected": bool(n == 16),
        "mean": mean_diff,
        "sample_sd": sample_sd,
        "standard_error": se_formula,
        "se_definition": "SE = sd / sqrt(n)",
        "se_check": {
            "se_from_formula": se_formula,
            "se_from_scipy_sem": se_scipy,
            "abs_difference": se_abs_diff,
            "verified": se_verified,
        },
        "degrees_of_freedom": dof,
        "degrees_of_freedom_scipy": dof_scipy,
        "t_statistic": t_stat,
        "p_value_two_sided": p_two_sided,
        "test_method": "scipy.stats.ttest_1samp(differences, popmean=0), bilateral",
        "paired_crosscheck": {
            "method": "scipy.stats.ttest_rel(error_a, error_b)",
            "t_statistic": float(res_rel.statistic),
            "p_value": float(res_rel.pvalue),
            "consistent_with_one_sample_on_differences": paired_consistent,
        },
        "confidence_level": conf,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "t_critical_975": t_crit,
        "ci_check": {
            "low_manual": ci_low_manual,
            "high_manual": ci_high_manual,
            "verified": ci_verified,
        },
        "rejects_h0_at_5pct": rejects_h0,
        "ci_contains_zero": ci_contains_zero,
        "interpretation": {
            "mean_difference": (
                f"La diferencia media es {mean_diff:.4f} unidades de la métrica "
                "(error_a − error_b): en promedio, el modelo A comete más error que B."
            ),
            "dispersion_vs_standard_error": (
                f"La sd muestral ({sample_sd:.4f}) describe la dispersión de las diferencias "
                f"entre tiendas; el error estándar ({se_formula:.4f}) es sd/sqrt(n) y mide la "
                "precisión de la media muestral, no la dispersión individual entre tiendas."
            ),
            "decision_at_5pct": (
                f"p-value bilateral = {p_two_sided:.4f}; IC t 95 % = "
                f"[{ci_low:.4f}, {ci_high:.4f}] (contiene 0: {ci_contains_zero}). {decision}"
            ),
        },
        "table": records,
        "outputs": ["output/store_differences.csv", "output/store_differences.png"],
    }

    with open("resultados.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 8) Resumen breve
    # ------------------------------------------------------------------
    print("=== T4 · Efecto e incertidumbre de las diferencias pareadas ===")
    print(f"n                 = {n}")
    print(f"media             = {mean_diff:.6f}")
    print(f"sd muestral       = {sample_sd:.6f}")
    print(f"SE = sd/sqrt(n)   = {se_formula:.6f}  (verificado vs scipy.sem: {se_verified})")
    print(f"gl                = {dof}")
    print(f"estadístico t     = {t_stat:.6f}")
    print(f"p-value bilateral = {p_two_sided:.6f}")
    print(
        f"IC t 95 %         = [{ci_low:.6f}, {ci_high:.6f}]  "
        f"(verificado con cuantil t: {ci_verified})"
    )
    print(f"Coherencia con ttest_rel (pareada): {paired_consistent}")
    print(f"Decisión al 5 %: {decision}")
    print(f"Archivos escritos: resultados.json, {csv_out}, output/store_differences.png")


if __name__ == "__main__":
    main()
