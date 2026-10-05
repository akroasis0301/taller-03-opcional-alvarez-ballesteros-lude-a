# -*- coding: utf-8 -*-
"""
T6 — Bootstrap pareado con remuestreo de tiendas completas.

- Generador: np.random.default_rng con SEED = 20260829 y BOOTSTRAP_RESAMPLES = 10000.
- Unidad de remuestreo: TIENDA COMPLETA, operando sobre el vector de diferencias
  d_i = error_a_i - error_b_i (nunca se remuestrean los errores de A y B por separado).
- Estadístico por remuestra: media del vector de diferencias remuestreado.
- Se reporta el intervalo percentil del 95 % como análisis de sensibilidad aproximado.
- Reproducibilidad: dos ejecuciones independientes con la misma semilla deben producir
  exactamente el mismo intervalo.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- Parámetros fijos -----------------------------
SEED = 20260829
BOOTSTRAP_RESAMPLES = 10_000
CONF_LEVEL_PCT = 95.0
ALPHA_PCT = (100.0 - CONF_LEVEL_PCT) / 2.0  # 2.5 % en cada cola

DATA_PATH = Path("data/model_errors.csv")
T2_RESULTS = Path("entrada/T2/resultados.json")
OUT_DIR = Path("output")
OUT_CSV = OUT_DIR / "store_differences.csv"   # archivo exigido por el enunciado
OUT_PNG = OUT_DIR / "store_differences.png"   # archivo exigido por el enunciado
BOOT_PNG = Path("bootstrap_percentile_interval.png")
RESULTS_JSON = Path("resultados.json")


def _json_default(o):
    """Convierte tipos de NumPy a tipos nativos para serializar a JSON."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Tipo no serializable: {type(o)}")


def load_paired_table() -> pd.DataFrame:
    """Carga data/model_errors.csv, valida y construye el vector pareado de diferencias."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    required = ["store_id", "error_a", "error_b"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas {missing} en {DATA_PATH}")
    df = df[required].copy()

    df["store_id"] = df["store_id"].astype(str)
    if df["store_id"].duplicated().any():
        dups = df.loc[df["store_id"].duplicated(), "store_id"].tolist()
        raise ValueError(f"store_id duplicado (identificador no único): {dups}")

    for c in ["error_a", "error_b"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
        vals = df[c].to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            raise ValueError(f"Valores no finitos en la columna {c}")
        if (vals < 0).any():
            raise ValueError(f"Valores negativos en la columna {c}")

    # Orden determinista por identificador de tienda
    df = df.sort_values("store_id", kind="mergesort").reset_index(drop=True)

    # Vector de diferencias pareadas (una diferencia por tienda completa)
    df["difference"] = df["error_a"] - df["error_b"]
    df["favors"] = np.where(df["difference"] > 0, "B",
                            np.where(df["difference"] < 0, "A", "tie"))
    return df


def bootstrap_paired(d: np.ndarray, seed: int, n_resamples: int):
    """
    Bootstrap pareado: remuestrea TIENDAS COMPLETAS generando índices de tiendas
    con reemplazo y promediando el vector de diferencias en cada remuestra.
    Nunca se remuestrean por separado los errores de A y B.
    """
    rng = np.random.default_rng(seed)
    n = d.shape[0]
    # Cada fila es una remuestra de tiendas completas (índices con reemplazo)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = d[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, ALPHA_PCT))
    hi = float(np.percentile(boot_means, 100.0 - ALPHA_PCT))
    return boot_means, lo, hi, idx


def main() -> None:
    # ------------------------- Datos y vector pareado -------------------------
    df = load_paired_table()
    d = df["difference"].to_numpy(dtype=float)
    n = int(d.size)
    observed_mean = float(d.mean())

    # --------------------- Bootstrap: dos ejecuciones -------------------------
    # Ejecución 1 y ejecución 2: generadores independientes creados con la MISMA
    # semilla. Deben producir exactamente el mismo intervalo (reproducibilidad).
    boot1, lo1, hi1, idx1 = bootstrap_paired(d, SEED, BOOTSTRAP_RESAMPLES)
    boot2, lo2, hi2, idx2 = bootstrap_paired(d, SEED, BOOTSTRAP_RESAMPLES)

    same_indices = bool(np.array_equal(idx1, idx2))
    same_means = bool(np.array_equal(boot1, boot2))
    same_interval = bool(lo1 == lo2 and hi1 == hi2)
    max_abs_diff_means = float(np.max(np.abs(boot1 - boot2)))

    # ------------------------- Archivos exigidos ------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df[["store_id", "error_a", "error_b", "difference", "favors"]].to_csv(
        OUT_CSV, index=False
    )

    # Figura exigida: diferencias pareadas por tienda
    colors = np.where(df["difference"].to_numpy() > 0, "#1f77b4", "#d62728")
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(df["store_id"], df["difference"], color=colors, edgecolor="black", lw=0.4)
    ax.axhline(0.0, color="black", lw=1.0)
    ax.axhline(observed_mean, color="green", ls="--", lw=1.2,
               label=f"media pareada = {observed_mean:.4f}")
    ax.set_xlabel("store_id")
    ax.set_ylabel("difference = error_a - error_b")
    ax.set_title("Diferencias pareadas por tienda (positivo favorece a B, negativo a A)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)

    # Figura adicional: distribución bootstrap e intervalo percentil del 95 %
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(boot1, bins=60, color="#9ecae1", edgecolor="white")
    ax.axvline(observed_mean, color="black", ls="--", lw=1.4,
               label=f"media observada = {observed_mean:.4f}")
    ax.axvline(lo1, color="red", lw=1.6, label=f"percentil 2.5 % = {lo1:.4f}")
    ax.axvline(hi1, color="red", lw=1.6, label=f"percentil 97.5 % = {hi1:.4f}")
    ax.set_xlabel("media de la diferencia en la remuestra bootstrap")
    ax.set_ylabel("frecuencia")
    ax.set_title(f"Bootstrap pareado ({BOOTSTRAP_RESAMPLES} remuestras de tiendas "
                 f"completas, semilla {SEED}) — IC percentil 95 %")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(BOOT_PNG, dpi=120)
    plt.close(fig)

    # ------------------- Verificación suave contra T2 -------------------------
    t2_check = None
    if T2_RESULTS.exists():
        try:
            t2 = json.loads(T2_RESULTS.read_text(encoding="utf-8"))
            if "mean_difference" in t2:
                t2_check = bool(np.isclose(float(t2["mean_difference"]),
                                           observed_mean, rtol=0.0, atol=1e-9))
        except Exception:
            t2_check = None

    # ----------------------------- resultados.json ----------------------------
    results = {
        "subtask": "T6",
        "description": ("Bootstrap pareado: remuestreo de tiendas completas mediante el "
                        "vector de diferencias; intervalo percentil del 95 % como "
                        "análisis de sensibilidad aproximado y verificación de "
                        "reproducibilidad con dos ejecuciones de la misma semilla"),
        "method": {
            "generator": "np.random.default_rng",
            "resampling_unit": ("tienda completa (par error_a, error_b) vía el vector de "
                                "diferencias d_i = error_a_i - error_b_i; nunca se "
                                "remuestrean los errores de A y B por separado"),
            "statistic_per_resample": "media del vector de diferencias remuestreado",
            "percentile_method": "np.percentile (interpolación lineal por defecto)",
            "indices_drawn_with_replacement": True,
        },
        "seed": SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "confidence_level_pct": CONF_LEVEL_PCT,
        "n_stores": n,
        "observed_mean_difference": observed_mean,
        "bootstrap_mean_of_means": float(boot1.mean()),
        "bootstrap_std_of_means": float(boot1.std(ddof=1)),
        "percentile_interval_95": {"low": lo1, "high": hi1},
        "percentile_2_5": lo1,
        "percentile_97_5": hi1,
        "interval_covers_zero": bool(lo1 <= 0.0 <= hi1),
        "fraction_bootstrap_means_positive": float((boot1 > 0).mean()),
        "fraction_bootstrap_means_negative": float((boot1 < 0).mean()),
        "reproducibility": {
            "two_runs_same_seed": True,
            "run1_percentile_interval_95": [lo1, hi1],
            "run2_percentile_interval_95": [lo2, hi2],
            "intervals_identical": same_interval,
            "bootstrap_means_identical": same_means,
            "resample_index_matrices_identical": same_indices,
            "max_abs_difference_between_runs": max_abs_diff_means,
        },
        "cross_check_with_T2_mean_difference": t2_check,
        "table": df[["store_id", "error_a", "error_b", "difference", "favors"]].to_dict(
            orient="records"
        ),
        "outputs": [
            str(OUT_CSV),
            str(OUT_PNG),
            str(BOOT_PNG),
            str(RESULTS_JSON),
        ],
    }
    with RESULTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)

    # -------------------------------- Resumen ---------------------------------
    print("T6 — Bootstrap pareado (remuestreo de tiendas completas)")
    print(f"  Semilla: {SEED} | Remuestras: {BOOTSTRAP_RESAMPLES} | Tiendas: {n}")
    print(f"  Media pareada observada: {observed_mean:.6f}")
    print(f"  Media bootstrap de las medias: {float(boot1.mean()):.6f} "
          f"(DE bootstrap: {float(boot1.std(ddof=1)):.6f})")
    print(f"  Intervalo percentil 95 %: [{lo1:.6f}, {hi1:.6f}] "
          f"(cubre cero: {lo1 <= 0.0 <= hi1})")
    print("  Reproducibilidad (dos ejecuciones, misma semilla):")
    print(f"    run1: [{lo1:.6f}, {hi1:.6f}]")
    print(f"    run2: [{lo2:.6f}, {hi2:.6f}]")
    print(f"    intervalos idénticos: {same_interval} | "
          f"medias bootstrap idénticas: {same_means} | "
          f"max|Δ| = {max_abs_diff_means:.3e}")
    print(f"  Archivos escritos: {OUT_CSV}, {OUT_PNG}, {BOOT_PNG}, {RESULTS_JSON}")


if __name__ == "__main__":
    main()
