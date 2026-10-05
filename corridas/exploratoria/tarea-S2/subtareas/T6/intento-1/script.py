# -*- coding: utf-8 -*-
"""
T6 — Bootstrap pareado (análisis de sensibilidad aproximado).

- Carga data/model_errors.csv (store_id, error_a, error_b) con validaciones
  coherentes con T2 y construye el vector de diferencias pareadas por tienda:
  difference = error_a - error_b.
- Bootstrap con np.random.default_rng(SEED=20260829) y 10_000 remuestras,
  remuestreando TIENDAS COMPLETAS (índices de filas pareadas), nunca los
  errores de A y B por separado.
- Reporta el intervalo percentil del 95 % y verifica reproducibilidad:
  dos ejecuciones con la misma semilla deben dar el mismo intervalo.
- Genera: t6_bootstrap_histograma.png, t6_bootstrap_medias.csv, resultados.json
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
PERCENTILES = (2.5, 97.5)  # intervalo percentil del 95 %

RUTA_DATOS = Path("data/model_errors.csv")
RUTA_T2 = Path("entrada/T2/resultados.json")

FIGURA_PNG = "t6_bootstrap_histograma.png"
CSV_REMUESTRAS = "t6_bootstrap_medias.csv"
RUTA_RESULTADOS = Path("resultados.json")


# ------------------------------- Validaciones -------------------------------
def cargar_y_validar(ruta: Path) -> pd.DataFrame:
    """Carga el CSV y aplica las validaciones de emparejamiento de T2."""
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta}")
    df = pd.read_csv(ruta)

    columnas_esperadas = ["store_id", "error_a", "error_b"]
    if list(df.columns) != columnas_esperadas:
        raise ValueError(
            f"Columnas esperadas {columnas_esperadas}; encontradas {list(df.columns)}"
        )
    if df["store_id"].isna().any():
        raise ValueError("store_id contiene valores nulos")
    duplicados = df.loc[df["store_id"].duplicated(), "store_id"].tolist()
    if duplicados:
        raise ValueError(f"store_id duplicado: {duplicados[0]}")

    for col in ("error_a", "error_b"):
        valores = pd.to_numeric(df[col], errors="coerce")
        if valores.isna().any():
            raise ValueError(f"La columna {col} contiene valores no numéricos")
        arr = valores.to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError(f"La columna {col} contiene valores no finitos")
        if (arr < 0).any():
            raise ValueError(f"La columna {col} contiene valores negativos")
        df[col] = arr
    return df


# ------------------------------ Bootstrap pareado ---------------------------
def bootstrap_pareado(diferencias: np.ndarray, semilla: int, n_remuestras: int):
    """
    Bootstrap pareado: remuestrea TIENDAS COMPLETAS (filas del vector de
    diferencias) con reemplazo. Cada remuestra tiene n_tiendas tiendas y la
    estadística es la media de las diferencias de la remuestra. El par
    (error_a, error_b) de cada tienda entra o sale junto: NUNCA se remuestrean
    los errores de A y B por separado.
    """
    rng = np.random.default_rng(semilla)
    n_tiendas = diferencias.size
    indices = rng.integers(0, n_tiendas, size=(n_remuestras, n_tiendas))
    medias = diferencias[indices].mean(axis=1)
    lo, hi = np.percentile(medias, PERCENTILES)
    return medias, float(lo), float(hi)


# ------------------------------- Carga de datos -----------------------------
df = cargar_y_validar(RUTA_DATOS)
df["difference"] = df["error_a"] - df["error_b"]
diferencias = df["difference"].to_numpy(dtype=float)
n_tiendas = diferencias.size
media_observada = float(diferencias.mean())

# Comprobación suave contra el resultado de T2 (si está disponible)
coincide_t2 = None
if RUTA_T2.exists():
    with RUTA_T2.open(encoding="utf-8") as f:
        t2 = json.load(f)
    diff_t2 = np.array(
        [fila["difference"] for fila in t2.get("tabla_pareada", [])], dtype=float
    )
    if diff_t2.size == diferencias.size:
        coincide_t2 = bool(np.allclose(diff_t2, diferencias, rtol=0.0, atol=1e-9))

# ------------------------- Bootstrap: dos corridas --------------------------
# Corrida 1 y corrida 2: cada una crea su propio generador con la MISMA semilla,
# simulando dos ejecuciones completas e independientes del procedimiento.
medias_1, lo_1, hi_1 = bootstrap_pareado(diferencias, SEED, BOOTSTRAP_RESAMPLES)
medias_2, lo_2, hi_2 = bootstrap_pareado(diferencias, SEED, BOOTSTRAP_RESAMPLES)

intervalos_identicos = bool(lo_1 == lo_2 and hi_1 == hi_2)
medias_identicas = bool(np.array_equal(medias_1, medias_2))
max_diff_intervalo = float(max(abs(lo_1 - lo_2), abs(hi_1 - hi_2)))
max_diff_medias = float(np.max(np.abs(medias_1 - medias_2)))
reproducible = intervalos_identicos and medias_identicas

# ------------------------------ Estadísticos --------------------------------
media_bootstrap = float(medias_1.mean())
mediana_bootstrap = float(np.median(medias_1))
sd_bootstrap = float(medias_1.std(ddof=1))
prop_media_positiva = float((medias_1 > 0).mean())
ic_contiene_cero = bool(lo_1 <= 0.0 <= hi_1)

# ------------------------------- CSV de remuestras --------------------------
pd.DataFrame(
    {
        "remuestra": np.arange(1, BOOTSTRAP_RESAMPLES + 1, dtype=int),
        "media_diferencias": medias_1,
    }
).to_csv(CSV_REMUESTRAS, index=False)

# ---------------------------------- Figura ----------------------------------
fig, ax = plt.subplots(figsize=(8.0, 5.0))
ax.hist(medias_1, bins=40, color="#4C72B0", alpha=0.8, edgecolor="white")
ax.axvline(
    media_observada, color="black", lw=2,
    label=f"Media observada = {media_observada:.4f}",
)
ax.axvline(lo_1, color="crimson", ls="--", lw=2, label=f"Percentil 2.5 = {lo_1:.4f}")
ax.axvline(hi_1, color="crimson", ls="--", lw=2, label=f"Percentil 97.5 = {hi_1:.4f}")
ax.set_title(
    "Bootstrap pareado — IC percentil 95 %\n"
    f"SEED = {SEED} · {BOOTSTRAP_RESAMPLES} remuestras de tiendas completas"
)
ax.set_xlabel("Media de diferencias por remuestra (error_a − error_b)")
ax.set_ylabel("Frecuencia")
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
fig.savefig(FIGURA_PNG, dpi=120)
plt.close(fig)

# ------------------------------ resultados.json -----------------------------
resultados = {
    "subtarea": "T6",
    "titulo": (
        "Bootstrap pareado — intervalo percentil del 95 % "
        "(análisis de sensibilidad aproximado)"
    ),
    "semilla": SEED,
    "bootstrap_remuestras": BOOTSTRAP_RESAMPLES,
    "n_tiendas": int(n_tiendas),
    "generador_aleatorio": f"np.random.default_rng({SEED})",
    "remuestreo": (
        "tiendas completas mediante el vector de diferencias "
        "(difference = error_a - error_b); los errores de A y B "
        "NO se remuestrean por separado"
    ),
    "papel_del_intervalo": "análisis de sensibilidad aproximado",
    "percentiles_usados": list(PERCENTILES),
    "media_diferencias_observada": media_observada,
    "mediana_diferencias_observada": float(np.median(diferencias)),
    "desviacion_estandar_diferencias_muestral": float(diferencias.std(ddof=1)),
    "media_bootstrap": media_bootstrap,
    "mediana_bootstrap": mediana_bootstrap,
    "desviacion_estandar_bootstrap": sd_bootstrap,
    "error_estandar_bootstrap": sd_bootstrap,
    "ic_percentil_95": [lo_1, hi_1],
    "ic_percentil_95_inf": lo_1,
    "ic_percentil_95_sup": hi_1,
    "amplitud_ic_percentil_95": float(hi_1 - lo_1),
    "ic_contiene_cero": ic_contiene_cero,
    "proporcion_remuestras_con_media_mayor_que_cero": prop_media_positiva,
    "reproducibilidad": {
        "corridas_con_misma_semilla": 2,
        "corrida_1_ic_percentil_95": [lo_1, hi_1],
        "corrida_2_ic_percentil_95": [lo_2, hi_2],
        "intervalos_identicos": intervalos_identicos,
        "medias_bootstrap_identicas": medias_identicas,
        "max_diferencia_absoluta_en_intervalo": max_diff_intervalo,
        "max_diferencia_absoluta_en_medias_bootstrap": max_diff_medias,
        "reproducible": reproducible,
    },
    "coincide_con_T2": coincide_t2,
    "vector_diferencias": diferencias.tolist(),
    "tabla_pareada": [
        {
            "store_id": str(r.store_id),
            "error_a": float(r.error_a),
            "error_b": float(r.error_b),
            "difference": float(r.difference),
        }
        for r in df.itertuples(index=False)
    ],
    "archivos_generados": {
        "figura": FIGURA_PNG,
        "csv_remuestras_bootstrap": CSV_REMUESTRAS,
        "resultados": "resultados.json",
    },
}

with RUTA_RESULTADOS.open("w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# --------------------------------- Resumen ----------------------------------
print("=== T6 · Bootstrap pareado (análisis de sensibilidad aproximado) ===")
print(f"Semilla: {SEED} | Remuestras: {BOOTSTRAP_RESAMPLES} | Tiendas: {n_tiendas}")
print(f"Media observada de diferencias (A - B): {media_observada:.6f}")
print(f"IC percentil 95 %: [{lo_1:.6f}, {hi_1:.6f}] "
      f"(amplitud {hi_1 - lo_1:.6f})")
print(f"¿El IC percentil 95 % contiene 0?: {ic_contiene_cero}")
print(f"Reproducibilidad (2 corridas, misma semilla): {reproducible} "
      f"| max diff intervalo = {max_diff_intervalo} "
      f"| max diff medias bootstrap = {max_diff_medias}")
if coincide_t2 is not None:
    print(f"Vector de diferencias coincide con T2: {coincide_t2}")
print(f"Archivos generados: {FIGURA_PNG}, {CSV_REMUESTRAS}, resultados.json")
