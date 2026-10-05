# -*- coding: utf-8 -*-
"""
T5 - Guardar y verificar el resultado.

1) Reconstruye el DataFrame analizado de la Tarea 3
   (data/server_measurements.csv + load_score de la Tarea 2 + requires_review).
2) Lo guarda en output/server_analysis.parquet con PyArrow, sin el indice.
3) Vuelve a leerlo y verifica el round trip:
   filas antes == despues, columnas antes == despues,
   esquema antes == despues y valores iguales.
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


def cargar_json(ruta):
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def desenvolver(val):
    """Desenvuelve un valor tipo {'valores': [...]} si aplica."""
    if isinstance(val, dict):
        for k in ("valores", "values", "datos", "load_score", "serie"):
            if k in val:
                return val[k]
    return val


def buscar_listas_numericas(obj, largo):
    """Genera las listas de longitud `largo` con numeros anidadas en obj."""
    if isinstance(obj, list):
        if largo > 0 and len(obj) == largo and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj
        ):
            yield obj
        else:
            for x in obj:
                for hallazgo in buscar_listas_numericas(x, largo):
                    yield hallazgo
    elif isinstance(obj, dict):
        for v in obj.values():
            for hallazgo in buscar_listas_numericas(v, largo):
                yield hallazgo


# ----------------------------------------------------------------------
# 1) Datos base y referencias de subtareas previas (T2/T3)
# ----------------------------------------------------------------------
df_base = pd.read_csv(RUTA_CSV, parse_dates=["timestamp"], encoding="utf-8-sig")
df_base.columns = [str(c).strip().replace(" ", "_") for c in df_base.columns]
n = len(df_base)

t3 = cargar_json(Path("entrada/T3/resultados.json"))
t2 = cargar_json(Path("entrada/T2/resultados.json"))

conocidos = {}  # load_score por fila registrados por la T3
if isinstance(t3, dict):
    for fila in t3.get("filas_requires_review") or []:
        try:
            conocidos[int(fila["indice"])] = float(fila["load_score"])
        except Exception:
            pass


def validar_load_score(val):
    """True si val es un vector de longitud n que coincide con los valores de T3."""
    try:
        arr = np.asarray(val, dtype=float)
    except Exception:
        return False
    if arr.ndim != 1 or arr.shape[0] != n:
        return False
    for i, v in conocidos.items():
        if 0 <= i < n and abs(float(arr[i]) - v) > 1e-6:
            return False
    return True


# ----------------------------------------------------------------------
# 2) load_score (fuente principal: entrada/T2/resultados.json, clave 'load_score')
# ----------------------------------------------------------------------
load_score = None
fuente_load_score = None

candidatos = []
for etiqueta, data in (("entrada/T2/resultados.json", t2), ("entrada/T3/resultados.json", t3)):
    if isinstance(data, dict):
        for clave in ("load_score", "load_scores", "valores_load_score"):
            if clave in data:
                candidatos.append((etiqueta + " (clave '" + clave + "')", desenvolver(data[clave])))
        for i, lista in enumerate(buscar_listas_numericas(data, n)):
            candidatos.append((etiqueta + " (lista anidada " + str(i + 1) + ")", lista))

for etiqueta, candidato in candidatos:
    if validar_load_score(candidato):
        load_score = np.asarray(candidato, dtype=float)
        fuente_load_score = etiqueta
        break

if load_score is None:
    for carpeta in ("entrada/T2", "entrada/T3"):
        d = Path(carpeta)
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix not in (".parquet", ".csv", ".npy"):
                continue
            try:
                if f.suffix == ".npy":
                    arr = np.ravel(np.load(f, allow_pickle=False))
                    if validar_load_score(arr):
                        load_score = arr.astype(float)
                        fuente_load_score = str(f)
                        break
                    continue
                tmp = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
            except Exception:
                continue
            if isinstance(tmp, pd.DataFrame) and "load_score" in tmp.columns and len(tmp) == n:
                if validar_load_score(tmp["load_score"].to_numpy()):
                    load_score = tmp["load_score"].to_numpy(dtype=float)
                    fuente_load_score = str(f)
                    break
        if load_score is not None:
            break

if load_score is None:
    # Ultimo recurso: recalcular candidatos tipicos de load_score (T2)
    gpu = df_base["gpu_utilization"].to_numpy(dtype=float)
    cpu = df_base["cpu_utilization"].to_numpy(dtype=float)
    mem = df_base["memory_gb"].to_numpy(dtype=float)
    pot = df_base["power_w"].to_numpy(dtype=float)

    def zscore(x, ddof):
        x = np.asarray(x, dtype=float)
        return (x - x.mean()) / x.std(ddof=ddof)

    propuestos = {}
    for ddof in (0, 1):
        tag = "ddof=" + str(ddof)
        zg = zscore(gpu, ddof)
        zc = zscore(cpu, ddof)
        zm = zscore(mem, ddof)
        propuestos["z_gpu (" + tag + ")"] = zg
        propuestos["z_cpu (" + tag + ")"] = zc
        propuestos["z_mem (" + tag + ")"] = zm
        propuestos["z_power (" + tag + ")"] = zscore(pot, ddof)
        propuestos["z_gpu+z_cpu+z_mem (" + tag + ")"] = zg + zc + zm
        propuestos["(z_gpu+z_cpu+z_mem)/3 (" + tag + ")"] = (zg + zc + zm) / 3.0
        propuestos["z(gpu+cpu+mem) (" + tag + ")"] = zscore(gpu + cpu + mem, ddof)
        propuestos["z((gpu+cpu+mem)/3) (" + tag + ")"] = zscore((gpu + cpu + mem) / 3.0, ddof)

    for nombre, cand in propuestos.items():
        if validar_load_score(cand):
            load_score = np.asarray(cand, dtype=float)
            fuente_load_score = "recalculado: " + nombre
            break

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
        coherencia_t3["n_observaciones"] = bool(len(df) == int(t3["n_observaciones"]))
    if t3.get("columnas_df_analizado"):
        coherencia_t3["columnas"] = bool(list(df.columns) == list(t3["columnas_df_analizado"]))
    if t3.get("requires_review_dtype"):
        coherencia_t3["requires_review_dtype"] = bool(
            str(df["requires_review"].dtype) == str(t3["requires_review_dtype"])
        )
    if t3.get("n_requires_review") is not None:
        coherencia_t3["n_requires_review"] = bool(
            int(df["requires_review"].sum()) == int(t3["n_requires_review"])
        )

# ----------------------------------------------------------------------
# 4) Guardar con PyArrow en output/server_analysis.parquet, sin el indice
# ----------------------------------------------------------------------
RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
tabla = pa.Table.from_pandas(df, preserve_index=False)
pq.write_table(tabla, RUTA_PARQUET)

# ----------------------------------------------------------------------
# 5) Volver a leer y verificar el round trip
# ----------------------------------------------------------------------
df_back = pd.read_parquet(RUTA_PARQUET)
tabla_back = pq.read_table(RUTA_PARQUET)

filas_antes = int(df.shape[0])
filas_despues = int(df_back.shape[0])
cols_antes = list(df.columns)
cols_despues = list(df_back.columns)

filas_ok = bool(filas_antes == filas_despues)
columnas_ok = bool(cols_antes == cols_despues)

esquema_antes = tabla.schema
esquema_despues = tabla_back.schema
dtypes_antes = [str(t) for t in df.dtypes]
dtypes_despues = [str(t) for t in df_back.dtypes]
esquema_ok = bool(esquema_antes.equals(esquema_despues) and dtypes_antes == dtypes_despues)

a = df.reset_index(drop=True)
b = df_back.reset_index(drop=True)
valores_ok = bool(a.equals(b))
if not valores_ok:
    try:
        valores_ok = True
        for col in a.columns:
            x = a[col]
            y = b[col]
            if pd.api.types.is_float_dtype(x) and pd.api.types.is_float_dtype(y):
                if not np.allclose(
                    x.to_numpy(dtype=float),
                    y.to_numpy(dtype=float),
                    equal_nan=True,
                ):
                    valores_ok = False
                    break
            elif x.astype(str).tolist() != y.astype(str).tolist():
                valores_ok = False
                break
    except Exception:
        valores_ok = False

print("Round trip verificado: " + RUTA_PARQUET_STR)
print("Filas: " + str(filas_ok))
print("Columnas: " + str(columnas_ok))
print("Esquema: " + str(esquema_ok))
print("Valores: " + str(valores_ok))

# ----------------------------------------------------------------------
# 6) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T5",
    "archivo_parquet": RUTA_PARQUET_STR,
    "archivo_existe": bool(RUTA_PARQUET.exists()),
    "motor": "pyarrow",
    "index_incluido": False,
    "filas_antes": filas_antes,
    "filas_despues": filas_despues,
    "columnas_antes": cols_antes,
    "columnas_despues": cols_despues,
    "n_columnas_antes": len(cols_antes),
    "n_columnas_despues": len(cols_despues),
    "esquema_antes": str(esquema_antes),
    "esquema_despues": str(esquema_despues),
    "dtypes_antes": dtypes_antes,
    "dtypes_despues": dtypes_despues,
    "verificacion_filas": filas_ok,
    "verificacion_columnas": columnas_ok,
    "verificacion_esquema": esquema_ok,
    "verificacion_valores": valores_ok,
    "round_trip_verificado": bool(filas_ok and columnas_ok and esquema_ok and valores_ok),
    "load_score_fuente": fuente_load_score,
    "n_requires_review": int(df["requires_review"].sum()),
    "coherencia_con_T3": coherencia_t3,
    "mensaje": (
        "Round trip verificado: " + RUTA_PARQUET_STR
        + " | Filas: " + str(filas_ok)
        + ", Columnas: " + str(columnas_ok)
        + ", Esquema: " + str(esquema_ok)
        + ", Valores: " + str(valores_ok)
    ),
}

with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print(
    "Resumen: " + str(filas_antes) + " filas y " + str(len(cols_antes))
    + " columnas escritas y releidas sin perdidas; requires_review = "
    + str(int(df["requires_review"].sum())) + " filas."
)
print("resultados.json actualizado.")
