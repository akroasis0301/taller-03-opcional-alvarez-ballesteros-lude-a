#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T6 — Bootstrap pareado: intervalo percentil del 95 %.

Procedimiento (según enunciado):
  * Generador: np.random.default_rng con semilla SEED = 20260829.
  * Remuestras: BOOTSTRAP_RESAMPLES = 10_000.
  * Unidad de remuestreo: la TIENDA COMPLETA. Se remuestrea mediante el vector
    de diferencias pareadas (difference = error_a - error_b); NUNCA se
    remuestrean por separado los errores del modelo A y del modelo B.
  * Estadístico por remuestra: media del vector de diferencias remuestreado.
  * Reporte: intervalo percentil del 95 % (percentiles 2.5 y 97.5), como
    análisis de sensibilidad aproximado.
  * Reproducibilidad: dos ejecuciones con la misma semilla deben producir el
    mismo intervalo (se verifica dentro del script).

Entradas:  data/model_errors.csv (y, si existe, entrada/T2/resultados.json para
           verificación cruzada de las diferencias).
Salidas:   output/store_differences.csv, output/store_differences.png,
           bootstrap_pareado_T6.png, resultados.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------- Parámetros del enunciado ----------------------------
SEED = 20260829
BOOTSTRAP_RESAMPLES = 10_000
NIVEL = 0.95  # intervalo percentil del 95 %

CSV_FUENTE = Path("data/model_errors.csv")
RESULTADOS_T2 = Path("entrada/T2/resultados.json")
RUTA_CSV_DIF = Path("output/store_differences.csv")
RUTA_FIG_DIF = Path("output/store_differences.png")
RUTA_FIG_BOOT = Path("bootstrap_pareado_T6.png")
RUTA_RESULTADOS = Path("resultados.json")


def cargar_y_validar(ruta_csv):
    """Carga el CSV y audita: columnas exactas, store_id único y no nulo,
    errores numéricos, finitos y no negativos (ValueError en caso contrario)."""
    df = pd.read_csv(ruta_csv)
    esperadas = ["store_id", "error_a", "error_b"]
    if list(df.columns) != esperadas:
        raise ValueError(
            "Columnas incorrectas: se esperaban exactamente "
            f"{esperadas}, se encontraron {list(df.columns)}"
        )
    if df["store_id"].isna().any():
        raise ValueError("store_id contiene valores nulos (identificador inválido).")
    if df["store_id"].duplicated().any():
        dups = sorted(df.loc[df["store_id"].duplicated(), "store_id"].astype(str).unique())
        raise ValueError(f"store_id duplicado (identificador no único): {dups}")
    for col in ("error_a", "error_b"):
        valores = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if np.isnan(valores).any():
            raise ValueError(f"La columna '{col}' contiene valores no numéricos o nulos.")
        if not np.isfinite(valores).all():
            raise ValueError(f"La columna '{col}' contiene valores no finitos (NaN/inf).")
        if (valores < 0).any():
            raise ValueError(f"La columna '{col}' contiene valores negativos.")
    return df


def construir_pareado(df):
    """Tabla pareada por tienda: difference = error_a - error_b y favors
    ('B' si difference>0, 'A' si difference<0, 'empate' si ==0).
    Cada fila contiene ambos errores de la misma store_id: el par viaja junto."""
    tabla = df[["store_id", "error_a", "error_b"]].copy()
    tabla["difference"] = tabla["error_a"] - tabla["error_b"]
    tabla["favors"] = np.where(
        tabla["difference"] > 0, "B",
        np.where(tabla["difference"] < 0, "A", "empate"),
    )
    return tabla


def bootstrap_pareado(diferencias, semilla, n_remuestras):
    """Bootstrap percentil PAREADO.

    Remuestrea TIENDAS COMPLETAS: sortea índices de tiendas (con reemplazo) y
    toma, para cada tienda sorteada, su diferencia pareada completa
    (difference = error_a - error_b). Los errores de A y de B NUNCA se
    remuestrean por separado: la unidad aleatorizada es la tienda (el par),
    de modo que cada remuestra conserva la estructura de apareamiento.
    Estadístico por remuestra: media del vector de diferencias remuestreado.
    Devuelve (medias_bootstrap, límite_inferior, límite_superior, índices).
    """
    d = np.asarray(diferencias, dtype=float)
    n_tiendas = d.size
    rng = np.random.default_rng(semilla)  # generador exigido + semilla exigida
    idx = rng.integers(0, n_tiendas, size=(n_remuestras, n_tiendas))
    medias = d[idx].mean(axis=1)
    colas = 100.0 * (1.0 - NIVEL) / 2.0  # 2.5
    lo, hi = np.percentile(medias, [colas, 100.0 - colas])  # percentiles 2.5 y 97.5
    return medias, float(lo), float(hi), idx


def main():
    # ------------------------------------------------------------------ datos
    df = cargar_y_validar(CSV_FUENTE)
    tabla = construir_pareado(df)
    d = tabla["difference"].to_numpy(dtype=float)
    a = tabla["error_a"].to_numpy(dtype=float)
    b = tabla["error_b"].to_numpy(dtype=float)
    n_tiendas = int(d.size)

    # ------------------------------------------- verificación cruzada con T2
    consistencia_t2 = None
    if RESULTADOS_T2.exists():
        try:
            prev = json.loads(RESULTADOS_T2.read_text(encoding="utf-8"))
            dif_prev = np.asarray(prev.get("diferencias", []), dtype=float)
            consistencia_t2 = bool(
                dif_prev.size == d.size and np.allclose(dif_prev, d, rtol=0.0, atol=1e-9)
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            consistencia_t2 = None

    # ------------------------------------------------- bootstrap: ejecución 1
    medias1, lo1, hi1, idx1 = bootstrap_pareado(d, SEED, BOOTSTRAP_RESAMPLES)

    # Integridad del pareo dentro del remuestreo: la diferencia de la tienda
    # sorteada es idéntica a tomar la tienda completa (error_a, error_b) y
    # restar; nunca se rompe el par ni se remuestrea A y B por separado.
    pareo_integro = bool(np.array_equal(d[idx1], a[idx1] - b[idx1]))

    # ------------------------------- bootstrap: ejecución 2 (reproducibilidad)
    medias2, lo2, hi2, _ = bootstrap_pareado(d, SEED, BOOTSTRAP_RESAMPLES)
    intervalos_identicos = bool(lo1 == lo2 and hi1 == hi2)
    medias_identicas = bool(np.array_equal(medias1, medias2))
    if not (intervalos_identicos and medias_identicas):
        raise RuntimeError(
            "Fallo de reproducibilidad: dos ejecuciones con la misma semilla "
            f"dieron intervalos distintos: [{lo1!r}, {hi1!r}] vs [{lo2!r}, {hi2!r}]"
        )

    # ------------------------------------- archivos exigidos por el enunciado
    RUTA_CSV_DIF.parent.mkdir(parents=True, exist_ok=True)
    RUTA_FIG_DIF.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(RUTA_CSV_DIF, index=False)

    # Figura exigida: diferencias pareadas por tienda
    plt.figure(figsize=(9, 4.8))
    colores = np.where(d >= 0, "#1f77b4", "#d62728")  # azul: favors B; rojo: favors A
    plt.bar(tabla["store_id"], d, color=colores, edgecolor="black", linewidth=0.4)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axhline(float(d.mean()), color="green", linestyle="--", linewidth=1.0,
                label=f"media = {d.mean():.4f}")
    plt.xlabel("store_id")
    plt.ylabel("difference = error_a - error_b")
    plt.title("Diferencias pareadas por tienda (positivo favorece a B, negativo a A)")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(RUTA_FIG_DIF), dpi=120)
    plt.close()

    # Figura adicional: histograma de las medias bootstrap e IC percentil 95 %
    plt.figure(figsize=(9, 4.8))
    plt.hist(medias1, bins=50, color="#4c72b0", edgecolor="white", linewidth=0.3)
    plt.axvline(float(d.mean()), color="black", linestyle="--", linewidth=1.2,
                label=f"media observada = {d.mean():.4f}")
    plt.axvline(lo1, color="crimson", linewidth=1.5, label=f"percentil 2.5 = {lo1:.4f}")
    plt.axvline(hi1, color="crimson", linewidth=1.5, label=f"percentil 97.5 = {hi1:.4f}")
    plt.axvline(0.0, color="gray", linestyle=":", linewidth=1.0, label="0 (sin diferencia)")
    plt.xlabel("media de la diferencia en la remuestra")
    plt.ylabel("frecuencia")
    plt.title(f"Bootstrap pareado: {BOOTSTRAP_RESAMPLES} remuestras de tiendas completas "
              f"(semilla {SEED})")
    plt.legend(fontsize=8, loc="upper left")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(RUTA_FIG_BOOT), dpi=120)
    plt.close()

    # --------------------------------------------------------- resultados.json
    resultados = {
        "subtarea": "T6_bootstrap_pareado_intervalo_percentil_95",
        "csv_fuente": "data/model_errors.csv",
        "semilla": SEED,
        "n_remuestras": BOOTSTRAP_RESAMPLES,
        "n_tiendas": n_tiendas,
        "metodo": {
            "tipo": "bootstrap percentil pareado",
            "generador": "np.random.default_rng",
            "unidad_remuestreada": "tienda completa (par (error_a, error_b) -> difference)",
            "detalle_remuestreo": (
                "Se sortean índices de tiendas con reemplazo y se toma el vector de "
                "diferencias pareadas (difference = error_a - error_b); NUNCA se "
                "remuestrean por separado los errores de A y de B."
            ),
            "estadistico_por_remuestra": "media del vector de diferencias remuestreado",
            "intervalo": "percentil del 95 % (percentiles 2.5 y 97.5 de las medias bootstrap)",
            "rol": "análisis de sensibilidad aproximado",
        },
        "pareo_integro_en_remuestreo": pareo_integro,
        "diferencias": d.tolist(),
        "media_diferencia_observada": float(d.mean()),
        "mediana_diferencia_observada": float(np.median(d)),
        "desv_est_diferencias_ddof1": float(d.std(ddof=1)),
        "bootstrap": {
            "media_de_las_medias": float(medias1.mean()),
            "desv_est_de_las_medias": float(medias1.std(ddof=1)),
            "sesgo_aproximado": float(medias1.mean() - d.mean()),
            "percentil_2_5": lo1,
            "percentil_97_5": hi1,
            "intervalo_percentil_95": [lo1, hi1],
            "n_remuestras_media_positiva": int((medias1 > 0).sum()),
            "n_remuestras_media_negativa": int((medias1 < 0).sum()),
            "prop_remuestras_media_positiva": float((medias1 > 0).mean()),
            "prop_remuestras_media_negativa": float((medias1 < 0).mean()),
        },
        "intervalo_percentil_95_reportado": {
            "limite_inferior": lo1,
            "limite_superior": hi1,
            "intervalo_str": f"[{lo1:.6f}, {hi1:.6f}]",
            "nivel": 0.95,
            "semilla": SEED,
            "n_remuestras": BOOTSTRAP_RESAMPLES,
            "generador": f"np.random.default_rng({SEED})",
        },
        "intervalo_contiene_cero": bool(lo1 <= 0.0 <= hi1),
        "reproducibilidad": {
            "protocolo": (
                "Dos ejecuciones independientes del bootstrap con "
                f"np.random.default_rng({SEED}) y {BOOTSTRAP_RESAMPLES} remuestras"
            ),
            "segunda_ejecucion_intervalo_percentil_95": [lo2, hi2],
            "intervalos_identicos": intervalos_identicos,
            "medias_remuestra_identicas": medias_identicas,
            "reproducibilidad_comprobada": bool(intervalos_identicos and medias_identicas),
        },
        "consistencia_con_T2": {
            "archivo": "entrada/T2/resultados.json",
            "disponible": bool(RESULTADOS_T2.exists()),
            "diferencias_coinciden": consistencia_t2,
        },
        "archivos_generados": {
            "csv_diferencias": str(RUTA_CSV_DIF),
            "figura_diferencias": str(RUTA_FIG_DIF),
            "figura_bootstrap": str(RUTA_FIG_BOOT),
            "resultados": str(RUTA_RESULTADOS),
        },
    }
    RUTA_RESULTADOS.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ------------------------------------------------------------- resumen
    print("=" * 68)
    print("T6 — Bootstrap pareado (intervalo percentil del 95 %)")
    print("=" * 68)
    print(f"Datos: {CSV_FUENTE} | tiendas (pares completos): {n_tiendas}")
    print(f"Generador: np.random.default_rng | semilla: {SEED} | remuestras: {BOOTSTRAP_RESAMPLES}")
    print(f"Unidad remuestreada: tienda completa via vector de diferencias "
          f"(pareo íntegro: {pareo_integro})")
    print(f"Media observada de la diferencia: {d.mean():.6f}")
    print(f"IC 95 % percentil (ejecución 1): [{lo1:.6f}, {hi1:.6f}]")
    print(f"IC 95 % percentil (ejecución 2): [{lo2:.6f}, {hi2:.6f}]")
    print(f"Reproducibilidad (misma semilla -> mismo intervalo): {intervalos_identicos} "
          f"| medias idénticas: {medias_identicas}")
    contiene = "contiene" if (lo1 <= 0.0 <= hi1) else "NO contiene"
    print(f"El intervalo {contiene} al 0 (lectura de sensibilidad aproximada).")
    print(f"Archivos: {RUTA_CSV_DIF} | {RUTA_FIG_DIF} | {RUTA_FIG_BOOT} | {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
