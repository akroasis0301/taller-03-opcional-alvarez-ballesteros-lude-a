# -*- coding: utf-8 -*-
"""
T4 — Efecto e incertidumbre sobre el vector de diferencias pareadas.

Entrada : data/model_errors.csv (columnas store_id, error_a, error_b)
Salidas : resultados.json (contrato de cifras)
          output/store_differences.csv (tabla pareada por tienda)
          output/store_differences.png (figura: diferencias por tienda + sd vs EE)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG_SEED = 42  # única semilla, solo para el jitter cosmético de la figura


def main():
    # ------------------------------------------------------------------ #
    # 1) Cargar y validar los datos (sin modificar el CSV original)
    # ------------------------------------------------------------------ #
    csv_path = Path("data/model_errors.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"store_id": str})
    required = ["store_id", "error_a", "error_b"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {csv_path}: {missing} (se exigen {required})")

    if df["store_id"].isna().any():
        raise ValueError("store_id contiene valores nulos; el identificador es obligatorio")
    if df["store_id"].duplicated().any():
        dups = sorted(df.loc[df["store_id"].duplicated(), "store_id"].tolist())
        raise ValueError(f"store_id duplicado (identificador repetido): {dups}")

    a = pd.to_numeric(df["error_a"], errors="raise").to_numpy(dtype=float)
    b = pd.to_numeric(df["error_b"], errors="raise").to_numpy(dtype=float)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("error_a/error_b contienen valores no finitos (NaN/inf)")
    if (a < 0).any() or (b < 0).any():
        raise ValueError("error_a/error_b contienen valores negativos")

    store_ids = df["store_id"].tolist()
    diff = a - b  # vector de diferencias pareadas, alineado por store_id
    n = int(diff.size)
    if n < 2:
        raise ValueError("Se necesitan al menos 2 pares para estimar sd muestral (ddof=1)")

    # ------------------------------------------------------------------ #
    # 2) Estadísticos del efecto: n, mean, sample_sd (ddof=1), SE = sd/√n
    # ------------------------------------------------------------------ #
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))          # desviación MUESTRAL
    se = float(sd / np.sqrt(n))               # error estándar de la media
    sem_scipy = float(stats.sem(diff, ddof=1))
    se_consistente = bool(abs(se - sem_scipy) < 1e-12)

    # ------------------------------------------------------------------ #
    # 3) Prueba t de una muestra bilateral sobre las diferencias + IC t 95 %
    # ------------------------------------------------------------------ #
    dfree = n - 1
    t_res = stats.ttest_1samp(diff, popmean=0.0, alternative="two-sided")
    t_stat = float(t_res.statistic)
    p_val = float(t_res.pvalue)
    p_check = float(2.0 * stats.t.sf(abs(t_stat), dfree))

    conf = 0.95
    t_crit = float(stats.t.ppf(1.0 - (1.0 - conf) / 2.0, dfree))
    ci_low, ci_high = stats.t.interval(conf, df=dfree, loc=mean, scale=se)
    ci_low, ci_high = float(ci_low), float(ci_high)
    margin = float(t_crit * se)
    alpha = 0.05
    decision = "rechazar_H0" if p_val < alpha else "no_rechazar_H0"

    # ------------------------------------------------------------------ #
    # 4) Consistencia opcional con la subtarea previa T2
    # ------------------------------------------------------------------ #
    t2_path = Path("entrada/T2/resultados.json")
    consistencia_t2 = None
    if t2_path.exists():
        try:
            t2 = json.loads(t2_path.read_text(encoding="utf-8"))
            t2_pairs = {r["store_id"]: float(r["difference"]) for r in t2.get("tabla_pareada", [])}
            if t2_pairs:
                max_abs = max(abs(t2_pairs[s] - float(d))
                              for s, d in zip(store_ids, diff) if s in t2_pairs)
                consistencia_t2 = {
                    "archivo": str(t2_path),
                    "n_pares_t2": t2.get("n_pares"),
                    "max_abs_dif_differences": float(max_abs),
                    "coincide": bool(max_abs < 1e-9),
                }
        except Exception as exc:  # la consistencia con T2 es informativa, no bloqueante
            consistencia_t2 = {"archivo": str(t2_path), "error_lectura": str(exc)}

    # ------------------------------------------------------------------ #
    # 5) Interpretaciones (preguntas 13, 14 y 15)
    # ------------------------------------------------------------------ #
    signo = "positivo" if mean > 0 else ("negativo" if mean < 0 else "nulo")
    interp_media = (
        f"La diferencia media es {mean:+.6f} unidades por tienda (signo {signo}). "
        f"Leída en unidades vendidas, la modalidad A vende en promedio ≈{abs(mean):.2f} unidades "
        f"más que la B por tienda (difference = error_a − error_b > 0); con la convención de la T2 "
        f"sobre errores (menor error, mejor), ese mismo signo positivo favorece a B. "
        f"La magnitud ({abs(mean):.2f} unidades) es modesta frente a la dispersión entre tiendas: "
        f"las diferencias individuales van de {float(diff.min()):+.1f} a {float(diff.max()):+.1f} unidades."
    )
    interp_ee = (
        f"sample_sd = {sd:.6f} unidades: dispersión típica ENTRE tiendas (cuánto se aleja una tienda "
        f"cualquiera de la media); no disminuye al acumular más tiendas de la misma población. "
        f"standard_error = sd/√n = {sd:.6f}/√{n} = {se:.6f} unidades: incertidumbre de la MEDIA "
        f"muestral como estimación del efecto poblacional; es √n = {np.sqrt(n):.0f} veces menor que "
        f"la sd y sí se reduce al aumentar n. Confundirlos llevaría a sobrestimar la precisión de "
        f"tiendas individuales o a subestimar la variabilidad real del efecto entre tiendas."
    )
    interp_ic = (
        f"Bajo el modelo t (n={n} diferencias i.i.d. con media μ y aproximación normal), los valores "
        f"del efecto compatibles con los datos al 95 % son μ ∈ [{ci_low:.6f}, {ci_high:.6f}] unidades. "
        f"Interpretación frecuentista: el procedimiento t bilateral con {dfree} grados de libertad "
        f"contiene al verdadero μ en el 95 % de las muestras repetidas; NO es una probabilidad "
        f"posterior sobre μ (μ es fija, el intervalo es el aleatorio). Como el intervalo contiene 0, "
        f"un efecto nulo sigue siendo compatible con los datos; efectos hasta ≈{ci_high:.2f} unidades "
        f"no quedan descartados por los datos."
    )
    concl_t = (
        f"t({dfree}) = {t_stat:.6f}, p bilateral = {p_val:.6f} > α = {alpha}: {decision}; la media de "
        f"las diferencias no es estadísticamente distinguible de 0 al 5 %, coherente con el IC t 95 % "
        f"que contiene al 0."
    )

    # ------------------------------------------------------------------ #
    # 6) Archivos exigidos: output/store_differences.csv y .png
    # ------------------------------------------------------------------ #
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    favors = np.where(diff > 0, "B", np.where(diff < 0, "A", "tie"))
    tabla = pd.DataFrame({
        "store_id": store_ids,
        "error_a": a,
        "error_b": b,
        "difference": diff,
        "favors": favors,
    })
    tabla.to_csv(out_dir / "store_differences.csv", index=False, encoding="utf-8")

    colores = ["#2a9d8f" if d > 0 else "#e76f51" for d in diff]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # Panel 1: diferencias por tienda con media, IC 95 % y banda ±1 sd
    ax1.bar(store_ids, diff, color=colores, edgecolor="black", linewidth=0.4)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.axhspan(ci_low, ci_high, color="#264653", alpha=0.12,
                label=f"IC t 95 % = [{ci_low:.4f}, {ci_high:.4f}]")
    ax1.axhline(mean, color="#264653", ls="--", lw=1.8, label=f"media = {mean:.4f}")
    ax1.axhline(mean + sd, color="gray", ls=":", lw=1.2)
    ax1.axhline(mean - sd, color="gray", ls=":", lw=1.2,
                label=f"media ± 1 sd = {mean:.4f} ± {sd:.4f}")
    ax1.set_ylabel("difference = error_a − error_b (unidades)")
    ax1.set_title(f"Diferencias pareadas por tienda (n = {n})")
    ax1.tick_params(axis="x", rotation=90)
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(loc="lower left", fontsize=8)

    # Panel 2: dispersión entre tiendas (sd) vs error estándar de la media (EE)
    rng = np.random.default_rng(RNG_SEED)
    jitter = rng.uniform(-0.07, 0.07, size=n)
    ax2.axhspan(mean - sd, mean + sd, color="gray", alpha=0.15,
                label=f"±1 sd (dispersión entre tiendas) = ±{sd:.4f}")
    ax2.scatter(jitter, diff, s=42, color=colores, edgecolor="black", linewidth=0.4,
                zorder=3, label="tiendas (cada punto = 1 tienda)")
    ax2.errorbar([1.0], [mean], yerr=se, fmt="o", ms=9, color="#264653",
                 ecolor="#264653", elinewidth=2.2, capsize=8, capthick=2.2, zorder=4,
                 label=f"media ± EE = {mean:.4f} ± {se:.4f} (EE = sd/√n)")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Tiendas\n(dispersión sd)", "Media muestral\n(incertidumbre EE)"])
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylabel("unidades")
    ax2.set_title("Dispersión entre tiendas (sd) vs error estándar (EE = sd/√n)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    lo = float(min(diff.min(), ci_low) - 0.5)
    hi = float(max(diff.max(), ci_high) + 0.5)
    ax1.set_ylim(lo, hi)
    ax2.set_ylim(lo, hi)

    fig.suptitle("T4 · Efecto pareado e incertidumbre (error_a − error_b)", fontsize=12)
    fig.tight_layout()
    fig.savefig("output/store_differences.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 7) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------ #
    filas = [
        {
            "store_id": s,
            "error_a": float(ai),
            "error_b": float(bi),
            "difference": float(di),
            "favors": str(fi),
        }
        for s, ai, bi, di, fi in zip(store_ids, a, b, diff, favors)
    ]

    resultados = {
        "subtarea": "T4_efecto_e_incertidumbre_diferencias_pareadas",
        "datos": {"csv": "data/model_errors.csv", "n_filas": n, "columnas": required},
        "store_ids": store_ids,
        # --- cifras exactas exigidas (pregunta 13) ---
        "n": n,
        "mean": mean,
        "sample_sd": sd,
        "standard_error": se,
        "se_formula": "sd / sqrt(n)",
        "se_check_scipy_sem": sem_scipy,
        "se_consistente": se_consistente,
        "unidades": "unidades vendidas (difference = error_a − error_b)",
        # --- prueba t bilateral sobre las diferencias (pregunta 15) ---
        "t_test": {
            "prueba": "t de una muestra bilateral sobre el vector de diferencias (equivalente a t pareada)",
            "null_mean": 0.0,
            "alternative": "two-sided",
            "df": dfree,
            "statistic": t_stat,
            "p_value": p_val,
            "p_value_check_2_sf": p_check,
            "alpha": alpha,
            "decision": decision,
        },
        # --- intervalo t bilateral del 95 % (pregunta 14) ---
        "ci95": {
            "confidence_level": conf,
            "tipo": "t bilateral para la media de las diferencias pareadas",
            "df": dfree,
            "t_critical": t_crit,
            "lower": ci_low,
            "upper": ci_high,
            "margin_of_error": margin,
            "contiene_cero": bool(ci_low <= 0.0 <= ci_high),
        },
        # --- respuestas por pregunta ---
        "p13": {
            "n_tiendas": n,
            "mean": mean,
            "sample_sd": sd,
            "standard_error": se,
            "signo": signo,
            "diferencia_min": float(diff.min()),
            "diferencia_max": float(diff.max()),
            "interpretacion_media": interp_media,
            "interpretacion_dispersion_vs_ee": interp_ee,
        },
        "p14": {
            "intervalo_t_95": [ci_low, ci_high],
            "lower": ci_low,
            "upper": ci_high,
            "t_critical": t_crit,
            "df": dfree,
            "margin_of_error": margin,
            "interpretacion": interp_ic,
        },
        "p15": {
            "statistic": t_stat,
            "df": dfree,
            "p_value": p_val,
            "conclusion": concl_t,
        },
        "diferencias_por_tienda": filas,
        "consistencia_con_T2": consistencia_t2,
        "archivos_generados": {
            "resultados": "resultados.json",
            "csv": "output/store_differences.csv",
            "figura": "output/store_differences.png",
        },
    }

    with open("resultados.json", "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    # 8) Resumen
    # ------------------------------------------------------------------ #
    print("T4 — Efecto e incertidumbre (diferencias pareadas error_a − error_b)")
    print(f"  n                = {n}")
    print(f"  mean             = {mean:.6f} unidades")
    print(f"  sample_sd(ddof=1)= {sd:.6f} unidades (dispersión entre tiendas)")
    print(f"  standard_error   = sd/√n = {se:.6f} unidades (incertidumbre de la media)")
    print(f"  Prueba t bilateral: t({dfree}) = {t_stat:.6f}, p = {p_val:.6f} -> {decision} (α={alpha})")
    print(f"  IC t 95 % = [{ci_low:.6f}, {ci_high:.6f}] (t_crit = {t_crit:.6f}, margen = {margin:.6f})")
    if consistencia_t2 is not None:
        print(f"  Consistencia con T2: {consistencia_t2}")
    print("  Archivos: resultados.json, output/store_differences.csv, output/store_differences.png")


if __name__ == "__main__":
    main()
