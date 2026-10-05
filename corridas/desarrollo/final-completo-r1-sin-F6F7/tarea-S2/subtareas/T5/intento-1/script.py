#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5 — Prueba exacta por cambios de signo sobre las diferencias pareadas
(error_a − error_b), como referencia nula exacta después de la prueba t (T4).

Procedimiento (enunciado, Tarea 5):
  * Enumerar las 2^16 = 65536 configuraciones de signos de las 16 diferencias.
  * Para cada configuración, calcular la media nula de las diferencias.
  * Contar cuántas medias nulas tienen magnitud al menos tan extrema como la
    media observada (criterio bilateral: |media nula| >= |media observada|).
  * Reportar: total de configuraciones, conteo de extremas y p-value exacto
    (conteo / total).

La enumeración es exhaustiva y determinista: no hay aleatoriedad ni semilla.
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
    # 1) Datos: tabla pareada (misma fuente que T2/T4)
    # ------------------------------------------------------------------
    csv_path = Path("data/model_errors.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = ["store_id", "error_a", "error_b"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en {csv_path}: {missing}")
    if df["store_id"].duplicated().any():
        raise ValueError("El CSV contiene store_id duplicados")

    err_a = df["error_a"].to_numpy(dtype=float)
    err_b = df["error_b"].to_numpy(dtype=float)
    if not (np.isfinite(err_a).all() and np.isfinite(err_b).all()):
        raise ValueError("Los errores deben ser finitos")

    d = err_a - err_b                       # difference = error_a − error_b
    store_ids = df["store_id"].astype(str).tolist()
    favors = np.where(d > 0, "B", np.where(d < 0, "A", "tie"))
    n = int(d.size)

    if n != 16:
        raise ValueError(
            f"El enunciado fija 2^16 = 65536 configuraciones; se esperaban 16 "
            f"diferencias y hay {n}"
        )

    obs_mean = float(d.mean())
    obs_sum = float(d.sum())

    # ------------------------------------------------------------------
    # 2) Contraste con la subtarea previa T4 (misma media de diferencias)
    # ------------------------------------------------------------------
    t4_path = Path("entrada/T4/resultados.json")
    t4_mean = None
    t4_p_value = None
    t4_table_match = None
    if t4_path.exists():
        try:
            t4 = json.loads(t4_path.read_text(encoding="utf-8"))
            t4_mean = t4.get("mean")
            t4_p_value = t4.get("p_value_two_sided")
            t4_table = t4.get("table") or []
            if t4_table:
                t4_diffs = np.array([r["difference"] for r in t4_table], dtype=float)
                t4_table_match = (
                    t4_diffs.size == n
                    and bool(np.allclose(t4_diffs, d, rtol=0.0, atol=1e-9))
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # ------------------------------------------------------------------
    # 3) Enumeración exhaustiva de las 2^16 configuraciones de signos
    # ------------------------------------------------------------------
    n_configs = 2 ** n                      # 65536
    idx = np.arange(n_configs, dtype=np.int64)
    bits = (idx[:, None] >> np.arange(n, dtype=np.int64)[None, :]) & 1   # {0,1}
    signs = (1 - 2 * bits).astype(np.int8)  # +1 / −1, una fila por configuración

    # Medias nulas en aritmética de punto flotante
    null_sums = signs.astype(np.float64) @ d
    null_means = null_sums / n

    # ------------------------------------------------------------------
    # 4) Verificación en aritmética entera exacta (los datos tienen 1
    #    decimal: las diferencias son múltiplos exactos de 0.1)
    # ------------------------------------------------------------------
    scale = 10
    d_int = np.round(d * scale).astype(np.int64)
    on_grid = bool(np.allclose(d * scale, d_int, rtol=0.0, atol=1e-6))
    obs_sum_int = int(d_int.sum())
    null_sums_int = signs.astype(np.int64) @ d_int

    # ------------------------------------------------------------------
    # 5) Conteo de configuraciones extremas y p-value exacto
    # ------------------------------------------------------------------
    # tol solo absorbe ruido de redondeo (~1e-15); los valores alcanzables por
    # la media nula distan >= 0.1/16 = 0.00625 entre sí, por lo que la
    # tolerancia no puede incluir configuraciones genuinamente menos extremas.
    tol = 1e-9
    extreme_float = np.abs(null_means) >= abs(obs_mean) - tol
    extreme_int = np.abs(null_sums_int) >= abs(obs_sum_int)

    extreme_count_float = int(extreme_float.sum())
    extreme_count_int = int(extreme_int.sum())
    agree = on_grid and (extreme_count_float == extreme_count_int)

    # Autoridad: aritmética entera exacta si los datos están en la rejilla decimal
    extreme_count = extreme_count_int if on_grid else extreme_count_float
    p_value = extreme_count / n_configs

    abs_sums_int = np.abs(null_sums_int)
    ties_obs = int((abs_sums_int == abs(obs_sum_int)).sum())
    strict_ext = int((abs_sums_int > abs(obs_sum_int)).sum())
    if obs_sum_int > 0:
        ext_pos = int((null_sums_int >= obs_sum_int).sum())
        ext_neg = int((null_sums_int <= -obs_sum_int).sum())
    else:
        ext_pos = ext_neg = None

    n_pos = int((null_sums_int > 0).sum())
    n_neg = int((null_sums_int < 0).sum())

    null_stats = {
        "mean": float(null_means.mean()),
        "sd": float(null_means.std(ddof=1)),
        "min": float(null_means.min()),
        "max": float(null_means.max()),
        "n_positive_sums": n_pos,
        "n_negative_sums": n_neg,
        "symmetric_counts": bool(n_pos == n_neg == n_configs // 2),
    }

    # ------------------------------------------------------------------
    # 6) Referencia: prueba t de T4 sobre el mismo vector de diferencias
    # ------------------------------------------------------------------
    t_res = stats.ttest_1samp(d, 0.0)
    p_t = float(t_res.pvalue)

    # ------------------------------------------------------------------
    # 7) Entregables del enunciado: output/store_differences.csv y .png
    # ------------------------------------------------------------------
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(
        {
            "store_id": store_ids,
            "error_a": err_a,
            "error_b": err_b,
            "difference": d,
            "favors": favors,
        }
    )
    out_df.to_csv(out_dir / "store_differences.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    ax = axes[0]
    bar_colors = np.where(d >= 0, "#4c72b0", "#c44e52")
    ax.bar(np.arange(1, n + 1), d, color=bar_colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(obs_mean, color="green", linestyle="--", linewidth=1.3,
               label=f"media observada = {obs_mean:.4f}")
    ax.set_xticks(np.arange(1, n + 1))
    ax.set_xticklabels(store_ids, rotation=90)
    ax.set_ylabel("difference = error_a − error_b")
    ax.set_title("Diferencias pareadas por tienda (n = 16)")
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    bin_edges = np.histogram_bin_edges(null_means, bins=80)
    ax.hist(null_means, bins=bin_edges, color="#9ecae1",
            edgecolor="#3182bd", linewidth=0.3,
            label=f"medias nulas ({n_configs} configuraciones)")
    extreme_vals = null_means[np.abs(null_means) >= abs(obs_mean) - tol]
    ax.hist(extreme_vals, bins=bin_edges, color="#fb6a4a",
            edgecolor="#cb181d", linewidth=0.3,
            label=f"al menos tan extremas: {extreme_count}")
    ax.axvline(obs_mean, color="green", linestyle="--", linewidth=1.5)
    ax.axvline(-obs_mean, color="green", linestyle="--", linewidth=1.5,
               label=f"± media observada = ±{abs(obs_mean):.4f}")
    ax.set_xlabel("media nula de las diferencias (signos reasignados)")
    ax.set_ylabel("número de configuraciones")
    ax.set_title(
        "Referencia nula exacta por cambios de signo: "
        f"2^{n} = {n_configs} configuraciones\n"
        f"p-value exacto = {extreme_count} / {n_configs} = {p_value:.6f}"
    )
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "store_differences.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 8) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    table = [
        {
            "store_id": store_ids[i],
            "error_a": float(err_a[i]),
            "error_b": float(err_b[i]),
            "difference": float(d[i]),
            "favors": str(favors[i]),
        }
        for i in range(n)
    ]

    results = {
        "subtask": "T5",
        "description": (
            "Prueba exacta por cambios de signo tras la prueba t (T4): enumeración "
            "exhaustiva de las 2^16 = 65536 configuraciones de signos de las "
            "diferencias pareadas, conteo de medias nulas con magnitud al menos tan "
            "extrema como la observada y p-value exacto"
        ),
        "csv_source": "data/model_errors.csv",
        "n": n,
        "n_zero_differences": int((d == 0).sum()),
        "observed_mean_difference": obs_mean,
        "observed_sum_difference": obs_sum,
        "total_configurations": int(n_configs),
        "n_sign_configurations": int(n_configs),
        "expected_total_configurations": 65536,
        "total_matches_expected": bool(n_configs == 65536),
        "comparison_rule": (
            "|media nula| >= |media observada| (bilateral, 'al menos tan extrema', "
            "incluye empates y la configuración observada)"
        ),
        "extreme_count": extreme_count,
        "count_extreme_configurations": extreme_count,
        "p_value_exact": p_value,
        "p_value": p_value,
        "p_value_definition": "p_value = extreme_count / total_configurations",
        "p_value_fraction": f"{extreme_count}/{n_configs}",
        "enumeration_method": (
            "exhaustiva y determinista: las 2^16 asignaciones de signo (+1/−1) a las "
            "16 diferencias; sin muestreo aleatorio ni semilla"
        ),
        "float_comparison_tolerance": tol,
        "exact_integer_verification": {
            "scale_factor": scale,
            "differences_on_decimal_grid": on_grid,
            "observed_sum_integer_units": obs_sum_int,
            "extreme_count_integer_arithmetic": extreme_count_int,
            "extreme_count_float_arithmetic": extreme_count_float,
            "agreement": bool(agree),
        },
        "ties_at_observed_magnitude": ties_obs,
        "strictly_more_extreme_count": strict_ext,
        "extreme_positive_count": ext_pos,
        "extreme_negative_count": ext_neg,
        "includes_observed_configuration": True,
        "null_distribution_summary": null_stats,
        "t_test_reference": {
            "p_value_two_sided_t_test": p_t,
            "p_value_exact_sign_flip": p_value,
            "abs_difference": abs(p_t - p_value),
            "note": (
                "Ambos evalúan H0: media de las diferencias = 0, pero con referencias "
                "nulas y supuestos distintos; no son procedimientos intercambiables."
            ),
        },
        "rejects_h0_at_5pct": bool(p_value < 0.05),
        "t4_crosscheck": {
            "t4_mean": t4_mean,
            "mean_matches_t4": (
                bool(abs(t4_mean - obs_mean) < 1e-12)
                if isinstance(t4_mean, (int, float)) else None
            ),
            "t4_p_value_two_sided": t4_p_value,
            "t4_table_differences_match": t4_table_match,
        },
        "interpretation": {
            "p_value_condicional_a_h0": (
                f"Condicional a la hipótesis nula (bajo H0 cada una de las {n_configs} "
                "configuraciones de signos es igualmente probable, con probabilidad "
                f"{1.0 / n_configs:.8f}), el p-value es la probabilidad de observar una "
                "media de las diferencias con magnitud al menos tan extrema como la "
                f"observada ({obs_mean:.4f}): {extreme_count} de {n_configs} "
                f"configuraciones, p = {p_value:.6f}."
            ),
            "diseno": (
                "Diseño pareado dentro de tienda: cada tienda aporta una diferencia "
                "error_a − error_b; la unidad de análisis es la tienda (n = 16) y la "
                "estadística de prueba es la media de las diferencias."
            ),
            "supuestos": (
                "La referencia nula por cambios de signo solo supone que, bajo H0, el "
                "signo de cada diferencia es intercambiable (simetría de cada "
                "diferencia respecto a 0); no exige normalidad. La prueba t supone que "
                "la media muestral sigue (aprox.) una distribución t de Student con "
                "n−1 = 15 grados de libertad. Los supuestos no son idénticos."
            ),
            "no_intercambiables": (
                "La prueba t compara el estadístico t observado con una distribución t "
                "teórica continua que depende de la varianza estimada, mientras que la "
                "prueba por cambios de signo compara la media observada con la "
                "distribución exacta y discreta de las 65536 medias obtenidas "
                "reasignando signos. Aunque aquí los p-values son numéricamente "
                f"cercanos (t: {p_t:.4f}; cambios de signo: {p_value:.4f}), provienen "
                "de modelos de nula distintos y no son intercambiables: pueden "
                "divergir con muestras pequeñas, asimetría o colas pesadas."
            ),
        },
        "table": table,
        "outputs": ["output/store_differences.csv", "output/store_differences.png"],
    }

    Path("resultados.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # 9) Resumen
    # ------------------------------------------------------------------
    print("=" * 66)
    print("T5 — Prueba exacta por cambios de signo (referencia nula exacta)")
    print("=" * 66)
    print(f"n diferencias pareadas            : {n}")
    print(f"media observada de las diferencias: {obs_mean:.10f}")
    print(f"total de configuraciones de signos: {n_configs} (2^{n})")
    print(f"configuraciones extremas          : {extreme_count}")
    print(f"  estrictamente más extremas      : {strict_ext}")
    print(f"  empates con la magnitud observada: {ties_obs}")
    print(f"p-value exacto                    : {extreme_count}/{n_configs} = {p_value:.10f}")
    print(f"verificación aritmética exacta    : {'OK' if agree else 'REVISAR'} "
          f"(enteros: {extreme_count_int}, flotantes: {extreme_count_float})")
    print(f"referencia prueba t (T4)          : p = {p_t:.10f} "
          f"(|diferencia| = {abs(p_t - p_value):.6f})")
    print(f"decisión al 5%                    : "
          f"{'se rechaza H0' if p_value < 0.05 else 'no se rechaza H0'}")
    print("Salidas: output/store_differences.csv, output/store_differences.png, "
          "resultados.json")


if __name__ == "__main__":
    main()
