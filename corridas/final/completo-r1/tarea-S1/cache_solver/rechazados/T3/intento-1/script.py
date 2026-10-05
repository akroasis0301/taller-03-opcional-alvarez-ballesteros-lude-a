# T3: Copia del DataFrame original + columnas load_score (de T2) y requires_review (bool),
# calculadas con operaciones vectorizadas. El original nunca se modifica.

from pathlib import Path
import json

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ rutas
RUTA_CSV = Path("data/server_measurements.csv")
RUTA_T2_JSON = Path("entrada/T2/resultados.json")
RUTA_PARQUET = Path("output/server_analysis.parquet")
RUTA_RESULTADOS = Path("resultados.json")

FEATURES = ["gpu_utilization", "cpu_utilization", "memory_gb"]
PESOS_DEFAULT = np.array([0.50, 0.30, 0.20])

# ------------------------------------------------- 1) DataFrame original
df_original = pd.read_csv(RUTA_CSV, parse_dates=["timestamp"])
df_original.columns = [c.strip().lower().replace(" ", "_") for c in df_original.columns]
cols_originales = list(df_original.columns)
snapshot_original = df_original.copy(deep=True)  # respaldo para verificar integridad

# ------------------------------------ 2) parámetros de T2 (medias/std/pesos)
fuente_params = None
t2 = None
if RUTA_T2_JSON.exists():
    with open(RUTA_T2_JSON, encoding="utf-8") as f:
        t2 = json.load(f)
    means = np.asarray(t2["means"], dtype=float)
    stds = np.asarray(t2["stds"], dtype=float)  # ddof=0 (z-scores con std=1 en T2)
    pesos = np.asarray(t2["pesos"], dtype=float)
    fuente_params = str(RUTA_T2_JSON)
else:
    X0 = df_original[FEATURES].to_numpy(dtype=float)
    means = X0.mean(axis=0)
    stds = X0.std(axis=0, ddof=0)
    pesos = PESOS_DEFAULT
    fuente_params = "recalculado in situ (ddof=0)"

# --------------------------- 3) copia del original + columnas vectorizadas
df_new = df_original.copy(deep=True)  # el original queda intacto

X = df_new[FEATURES].to_numpy(dtype=float)
std_safe = np.where(stds == 0.0, 1.0, stds)  # sin división por cero
Z = (X - means) / std_safe
Z = np.where(stds == 0.0, 0.0, Z)            # si std=0 -> z=0
df_new["load_score"] = Z @ pesos             # vectorizado

df_new["requires_review"] = (df_new["load_score"] > 1.5) | (df_new["temperature_c"] > 80)
df_new["requires_review"] = df_new["requires_review"].astype(bool)

# ------------------------------------------------- 4) verificación fila 23
fila = df_new.iloc[23]
ts_fila = pd.Timestamp(fila["timestamp"])
load_fila = float(fila["load_score"])
temp_fila = float(fila["temperature_c"])
review_fila = bool(fila["requires_review"])

ver_fila23 = {
    "server": str(fila["server"]),
    "server_ok": str(fila["server"]) == "AI-SRV-01",
    "timestamp": str(ts_fila),
    "timestamp_ok": ts_fila == pd.Timestamp("2026-08-24 09:55:00"),
    "load_score": load_fila,
    "load_score_ok": abs(load_fila - 3.136218) <= 1e-6,
    "temperature_c": temp_fila,
    "temperature_c_ok": abs(temp_fila - 66.97) <= 1e-9,
    "requires_review": review_fila,
    "requires_review_ok": review_fila is True,
}

# --------------------------------------- 5) integridad del original y reglas
original_intacto = bool(
    list(df_original.columns) == cols_originales
    and "load_score" not in df_original.columns
    and "requires_review" not in df_original.columns
    and df_original.equals(snapshot_original)
)
copia_fiel = bool(df_new[cols_originales].equals(df_original))
forma_ok = df_new.shape == (df_original.shape[0], len(cols_originales) + 2)
regla_ok = bool(
    (
        df_new["requires_review"]
        == ((df_new["load_score"] > 1.5) | (df_new["temperature_c"] > 80))
    ).all()
)
es_bool = bool(pd.api.types.is_bool_dtype(df_new["requires_review"]))
tiene_cols = bool("load_score" in df_new.columns and "requires_review" in df_new.columns)

max_load = float(df_new["load_score"].max())
coincide_t2_max = None
if t2 is not None:
    coincide_t2_max = bool(abs(max_load - float(t2["load_score_resumen"]["maximo"])) <= 1e-12)

# ------------------------------------------------------------- 6) conteos
mask_load = df_new["load_score"] > 1.5
mask_temp = df_new["temperature_c"] > 80
mask_rev = df_new["requires_review"]
conteos = {
    "n_filas": int(len(df_new)),
    "n_requires_review_true": int(mask_rev.sum()),
    "n_requires_review_false": int((~mask_rev).sum()),
    "proporcion_requires_review_true": float(mask_rev.mean()),
    "n_load_score_mayor_1_5": int(mask_load.sum()),
    "n_temperature_mayor_80": int(mask_temp.sum()),
    "n_ambas_condiciones": int((mask_load & mask_temp).sum()),
    "n_solo_load_score": int((mask_load & ~mask_temp).sum()),
    "n_solo_temperature": int((~mask_load & mask_temp).sum()),
}

# ------------------------------------------------------- 7) parquet exigido
RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df_new.to_parquet(RUTA_PARQUET, index=False)
df_parquet = pd.read_parquet(RUTA_PARQUET)
parquet_ok = bool(
    len(df_parquet) == len(df_new)
    and list(df_parquet.columns) == list(df_new.columns)
    and bool((df_parquet["requires_review"] == df_new["requires_review"]).all())
)

# ------------------------------------------------------- 8) resultados.json
fila_23_completa = {}
for col in df_new.columns:
    v = fila[col]
    if isinstance(v, pd.Timestamp):
        v = str(v)
    elif isinstance(v, (bool, np.bool_)):
        v = bool(v)
    elif isinstance(v, np.integer):
        v = int(v)
    elif isinstance(v, np.floating):
        v = float(v)
    fila_23_completa[col] = v

criterio = bool(
    tiene_cols
    and es_bool
    and all(v for k, v in ver_fila23.items() if k.endswith("_ok"))
    and original_intacto
    and copia_fiel
    and forma_ok
    and regla_ok
    and parquet_ok
)

resultados = {
    "fuente_parametros_T2": fuente_params,
    "means_usados": means.tolist(),
    "stds_usados": stds.tolist(),
    "pesos_usados": pesos.tolist(),
    "forma_original": list(df_original.shape),
    "forma_copia": list(df_new.shape),
    "columnas_originales": cols_originales,
    "columnas_finales": list(df_new.columns),
    "columnas_agregadas": ["load_score", "requires_review"],
    "load_score_dtype": str(df_new["load_score"].dtype),
    "requires_review_dtype": str(df_new["requires_review"].dtype),
    "requires_review_es_bool": es_bool,
    "fila_23": fila_23_completa,
    "fila_23_esperada": {
        "server": "AI-SRV-01",
        "timestamp": "2026-08-24 09:55:00",
        "load_score": 3.136218,
        "temperature_c": 66.97,
        "requires_review": True,
    },
    "verificacion_fila_23": ver_fila23,
    "load_score_resumen": {
        "media": float(df_new["load_score"].mean()),
        "desviacion_estandar": float(df_new["load_score"].std(ddof=0)),
        "minimo": float(df_new["load_score"].min()),
        "maximo": max_load,
    },
    "max_load_score_coincide_T2": coincide_t2_max,
    "conteos": conteos,
    "indices_requires_review_true": [int(i) for i in df_new.index[mask_rev]],
    "original_intacto": original_intacto,
    "copia_coincide_con_original_en_columnas_originales": copia_fiel,
    "forma_copia_correcta": forma_ok,
    "regla_review_verificada_vectorizada": regla_ok,
    "parquet_escrito": str(RUTA_PARQUET),
    "parquet_n_filas": int(len(df_parquet)),
    "parquet_columnas": list(df_parquet.columns),
    "parquet_verificado": parquet_ok,
    "criterio_exito_cumplido": criterio,
}

with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------ resumen
print("=== T3: copia del DataFrame con load_score y requires_review ===")
print(f"Original: {df_original.shape[0]} filas x {df_original.shape[1]} cols | intacto: {original_intacto}")
print(f"Copia:    {df_new.shape[0]} filas x {df_new.shape[1]} cols | agregadas: load_score, requires_review")
print(
    f"Fila 23 -> server={ver_fila23['server']}, timestamp={ver_fila23['timestamp']}, "
    f"load_score={ver_fila23['load_score']:.6f}, temperature_c={ver_fila23['temperature_c']}, "
    f"requires_review={ver_fila23['requires_review']}"
)
print(
    f"requires_review=True: {conteos['n_requires_review_true']}/{conteos['n_filas']} "
    f"(load>1.5: {conteos['n_load_score_mayor_1_5']}, temp>80: {conteos['n_temperature_mayor_80']}, "
    f"ambas: {conteos['n_ambas_condiciones']})"
)
print(f"Parquet: {RUTA_PARQUET} (filas={len(df_parquet)}, verificado={parquet_ok})")
print(f"Criterio de éxito cumplido: {criterio}")
