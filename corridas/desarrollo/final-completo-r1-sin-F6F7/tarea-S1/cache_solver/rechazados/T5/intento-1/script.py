```python
# -*- coding: utf-8 -*-
"""
T5 — Guardar y verificar el resultado.

1) Reconstruye el DataFrame analizado de la Tarea 3
   (data/server_measurements.csv + load_score de la Tarea 2 + requires_review).
2) Lo guarda en output/server_analysis.parquet con PyArrow, sin el índice.
3) Vuelve a leerlo y verifica el round trip:
   filas antes == después, columnas antes == después,
   esquema antes == después y valores iguales.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RUTA_CSV = Path("data/server_measurements.csv")
RUTA_PARQUET = Path("output/server_analysis.parquet")
RUTA_PARQUET_STR = RUTA_PARQUET.as_posix()
RUTA_RESULTADOS = Path("resultados.json")
UMBRAL_LOAD_SCORE = 1.5
UMBRAL_TEMPERATURA = 80.0

# ----------------------------------------------------------------------
# 1) Datos base y referencias de subtareas previas (T2/T3)
# ----------------------------------------------------------------------
df_base = pd.read_csv(RUTA_CSV, parse_dates=["timestamp"], encoding="utf-8-sig")
df_base.columns = [str(c).strip().replace(" ", "_") for c in df_base.columns]
n = len(df_base)


def cargar_json(ruta):
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


t3 = cargar_json(Path("entrada/T3/resultados.json"))
t2 = cargar_json(Path("entrada/T2/resultados.json"))

conocidos = {}  # load_score conocidos fila a fila (registrados por T3)
if isinstance(t3, dict):
    for fila in t3.get("filas_requires_review") or []:
        try:
            conocidos[int(fila["indice"])] = float(fila["load_score"])
        except Exception:
            pass


def validar_load_score(val):
    """Comprueba tamaño y coincidencia con los load_score conocidos de T3."""
    try:
        arr = np.asarray(val, dtype=float)
    except Exception:
        return False, np.inf
    if arr.ndim != 1 or arr.shape[0] != n:
        return False, np.inf
    if not conocidos:
        return True, 0.0
    diffs = [abs(float(arr[i]) - v) for i, v in conocidos.items() if 0 <= i < arr.shape[0]]
    if not diffs:
        return False, np.inf
    dmax = max(diffs)
    return dmax < 1e-9, dmax


def buscar_listas_numericas(obj, largo):
    """Todas las listas de `largo` números anidadas en una estructura JSON."""
    if isinstance(obj, list):
        if len(obj) == largo and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj
        ):
            yield obj
        else:
            for x in obj:
                yield from buscar_listas_numericas(x, largo)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from buscar_listas_numericas(v, largo)


def desenvolver(val):
    if isinstance(val, dict):
        for k in ("valores", "values", "datos", "load_score", "serie"):
            if k in val:
                return val[k]
    return val


# ----------------------------------------------------------------------
# 2) load_score (fuente principal: entrada/T2/resultados.json, clave
#    'load_score', tal como lo registró la Tarea 3; luego fallbacks)
# ----------------------------------------------------------------------
load_score = None
fuente_load_score = None

candidatos = []
for etiqueta, data in (("entrada/T2/resultados.json", t2), ("entrada/T3/resultados.json", t3)):
    if isinstance(data, dict):
        for clave in ("load_score", "load_scores", "valores_load_score"):
            if clave in data:
                candidatos.append((f"{etiqueta} (clave '{clave}')", desenvolver(data[clave])))
        for i, lista in enumerate(buscar_listas_numericas(data, n)):
            candidatos.append((f"{etiqueta} (lista anidada #{i + 1})", lista))

for etiqueta, candidato in candidatos:
    ok, _ = validar_load_score(candidato)
    if ok:
        load_score = np.asarray(candidato, dtype=float)
        fuente_load_score = etiqueta
        break

if load_score is None:
    rutas = []
    for carpeta in ("entrada/T2", "entrada/T3"):
        d = Path(carpeta)
        if d.is_dir():
            rutas.extend(sorted(d.iterdir()))
    for f in rutas:
        if f.name == "resultados.json" or f.suffix not in (".parquet", ".csv", ".npy"):
            continue
        try:
            if f.suffix == ".npy":
                arr = np.ravel(np.load(f, allow_pickle=False))
                ok, _ = validar_load_score(arr)
                if ok:
                    load_score, fuente_load_score = arr.astype(float), str(f)
                    break
                continue
            tmp = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
        except Exception:
            continue
        if isinstance(tmp, pd.DataFrame) and "load_score" in tmp.columns and len(tmp) == n:
            ok, _ = validar_load_score(tmp["load_score"].to_numpy())
            if ok:
                load_score = tmp["load_score"].to_numpy(dtype=float)
                fuente_load_score = str(f)
                break

if load_score is None:
    # Último recurso: recalcular candidatos típicos de load_score (T2)
    # y validarlos contra los valores conocidos registrados por T3.
    gpu = df_base["gpu_utilization"].to_numpy(dtype=float)
    cpu = df_base["cpu_utilization"].to_numpy(dtype=float)
    mem = df_base["memory_gb"].to_numpy(dtype=float)
    pot = df_base["power_w"].to_numpy(dtype=float)

    def zscore(x, ddof):
        x = np.asarray(x, dtype=float)
        return (x - x.mean()) / x.std(ddof=ddof)

    z_g0, z_g1 = zscore(gpu, 0), zscore(gpu, 1)
    z_c0, z_c1 = zscore(cpu, 0), zscore(cpu, 1)
    z_m0, z_m1 = zscore(mem, 0), zscore(mem, 1)

    propuestos = {}
    for ddof, tag in ((0, "ddof=0"), (1, "ddof=1")):
        propuestos[f"z(gpu_utilization, {tag})"] = zscore(gpu, ddof)
        propuestos[f"z(cpu_utilization, {tag})"] = zscore(cpu, ddof)
        propuestos[f"z(memory_gb, {tag})"] = zscore(mem, ddof)
        propuestos[f"z(power_w, {tag})"] = zscore(pot, ddof)
        propuestos[f"z(gpu+cpu, {tag})"] = zscore(gpu + cpu, ddof)
        propuestos[f"z(gpu+cpu+mem, {tag})"] = zscore(gpu + cpu + mem, ddof)
        propuestos[f"z((gpu+cpu+mem)/3, {tag})"] = zscore((gpu + cpu + mem) / 3.0, ddof)
        propuestos[f"z(0.4gpu+0.3cpu+0.3mem, {tag})"] = zscore(
            0.4 * gpu + 0.3 * cpu + 0.3 * mem, ddof
        )
        propuestos[f"z(gpu+cpu+mem+power, {tag})"] = zscore(gpu + cpu + mem + pot, ddof)
    propuestos["z_gpu+z_cpu+z_mem (ddof=0)"] = z_g0 + z_c0 + z_m0
    propuestos["z_gpu+z_cpu+z_mem (ddof=1)"] = z_g1 + z_c1 + z_m1
    propuestos["(z_gpu+z_cpu+z_mem)/3 (ddof=0)"] = (z_g0 + z_c0 + z_m0) / 3.0
    propuestos["(z_gpu+z_cpu+z_mem)/3 (ddof=1)"] = (z_g1 + z_c1 + z_m1) / 3.0
    propuestos["max(z_gpu,z_cpu,z_mem) (ddof=0)"] = np.maximum(np.maximum(z_g0, z_c0), z_m0)
    propuestos["max(z_gpu,z_cpu,z_mem) (ddof=1)"] = np.maximum(np.maximum(z_g1, z_c1), z_m1)

    mejor_arr, mejor_d, mejor_nombre = None, np.inf, None
    for nombre, cand in propuestos.items():
        ok, d = validar_load_score(cand)
        if ok:
            load_score = np.asarray(cand, dtype=float)
            fuente_load_score = f"recalculado: {nombre}"
            break
        if d < mejor_d:
            mejor_arr, mejor_d, mejor_nombre = np.asarray(cand, dtype=float), d, nombre
    if load_score is None and mejor_arr is not None:
        load_score = mejor_arr
        fuente_load_score = (
            f"recalculado (mejor aproximacion: {mejor_nombre}; dif. max. vs T3 = {mejor_d:.6g})"
        )
        print(f"AVISO: load_score reconstruido por aproximacion -> {fuente_load_score}")

if load_score is None:
    raise RuntimeError("No fue posible obtener el load_score de las subtareas previas.")

# ----------------------------------------------------------------------
# 3) DataFrame analizado (T3): copia + load_score + requires_review
# ----------------------------------------------------------------------
df = df_base.copy()
df["load_score"] = load_score
df["requires_review"] = (df["load_score"] > UMBRAL_LOAD_SCORE) | (
    df["temperature_c"] > UMBRAL_TEMPERATURA
)
df["requires_review"] = df["requires_review"].astype(bool)

if isinstance(t3, dict) and t3.get("columnas_df_analizado"):
    orden = [c for c in t3["columnas_df_analizado"] if c in df.columns]
    if len(orden) == df.shape[1]:
        df = df[orden]

coherencia_t3 = {}
if isinstance(t3, dict):
    if t3.get("n_observaciones") is not None:
        coherencia_t3["n_observaciones"] = bool(len(df) == t3["n_observaciones"])
    if t3.get("columnas_df_analizado"):
        coherencia_t3["columnas"] = bool(list(df.columns) == list(t3["columnas_df_analizado"]))
    if t3.get("requires_review_dtype"):
        coherencia_t3["requires_review_dtype"] = bool(
            str(df["requires_review"].dtype) == t3["requires_review
