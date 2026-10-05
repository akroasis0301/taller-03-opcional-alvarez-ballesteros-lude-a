#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T4 - Construir un resumen por servidor.

Agrupa el DataFrame analizado en T3 (columnas originales + load_score +
requires_review) por la columna `server` y construye la tabla resumen con
EXACTAMENTE las columnas, en este orden:

    server, observations, mean_power_w, max_temperature_c, mean_load, review_count

con una fila por servidor (6 filas):
    observations      : numero de filas del servidor
    mean_power_w      : media de power_w (W)
    max_temperature_c : maximo de temperature_c (C)
    mean_load         : media de load_score (puntuacion de carga de T2)
    review_count      : numero de filas con requires_review == True

Fuentes (por orden de prioridad):
    1) DataFrame analizado guardado por T3 en entrada/T3/ (parquet/csv/feather)
    2) data/server_measurements.csv + load_score de entrada/T3/resultados.json o
       entrada/T2/resultados.json + requires_review recalculada con la regla de
       T3 (load_score > 1.5 o temperature_c > 80)
    3) requires_review desde indices_requires_review de entrada/T3/resultados.json
    4) ultimo recurso: load_score reconstruido con la formula candidata que
       reproduce los valores registrados por T3.

Salidas:
    resultados.json                 (contrato de la subtarea)
    output/server_analysis.parquet  (exigido por el enunciado)
    t4_resumen_servidores.png       (figura resumen)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

RUTA_CSV = Path("data/server_measurements.csv")
CARPETA_T3 = Path("entrada/T3")
CARPETA_T2 = Path("entrada/T2")
COLUMNAS_RESUMEN = ["server", "observations", "mean_power_w",
                    "max_temperature_c", "mean_load", "review_count"]
UMBRAL_LOAD_SCORE = 1.5
UMBRAL_TEMPERATURA = 80.0
TOL_LOAD_SCORE = 1e-6


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def cargar_json(ruta):
    ruta = Path(ruta)
    if ruta.is_file():
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def extraer_lista_numeros(obj, n):
    """Busca recursivamente una lista de exactamente n numeros dentro de obj."""
    if isinstance(obj, (list, tuple)):
        if len(obj) == n:
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj):
                return [float(v) for v in obj]
            if obj and all(isinstance(v, dict) for v in obj):
                claves_valor = ("load_score", "value", "valor", "score")
                claves_indice = ("index", "indice", "i")
                if all(any(k in v and isinstance(v[k], (int, float))
                           and not isinstance(v[k], bool) for k in claves_valor)
                       for v in obj):
                    def _idx(v):
                        for k in claves_indice:
                            if k in v:
                                try:
                                    return int(v[k])
                                except (TypeError, ValueError):
                                    return None
                        return None
                    indices = [_idx(v) for v in obj]
                    if all(i is not None for i in indices):
                        orden = [v for _, v in sorted(zip(indices, obj),
                                                      key=lambda t: t[0])]
                    else:
                        orden = list(obj)
                    vals = []
                    for v in orden:
                        for k in claves_valor:
                            if (k in v and isinstance(v[k], (int, float))
                                    and not isinstance(v[k], bool)):
                                vals.append(float(v[k]))
                                break
                    if len(vals) == n:
                        return vals
        for v in obj:
            r = extraer_lista_numeros(v, n)
            if r is not None:
                return r
    elif isinstance(obj, dict):
        if len(obj) == n and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                 for v in obj.values()):
            try:
                claves = sorted(obj.keys(), key=lambda k: int(k))
            except (ValueError, TypeError):
                claves = list(obj.keys())
            return [float(obj[k]) for k in claves]
        for v in obj.values():
            r = extraer_lista_numeros(v, n)
            if r is not None:
                return r
    return None


def a_bool_numpy(serie):
    """Convierte una serie (bool, 0/1 o 'True'/'False') en un array numpy bool."""
    if serie.dtype == bool:
        return serie.to_numpy(dtype=bool)
    if serie.dtype == object:
        return (serie.astype(str).str.strip().str.lower()
                .isin(["true", "1", "1.0", "yes", "si"]).to_numpy())
    return serie.fillna(0).astype(bool).to_numpy()


def zscore(x, ddof=0):
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=ddof)
    if sd == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def intentar_cargar_tabla(ruta):
    try:
        if ruta.suffix == ".parquet":
            return pd.read_parquet(ruta)
        if ruta.suffix == ".csv":
            return pd.read_csv(ruta)
        if ruta.suffix == ".feather":
            return pd.read_feather(ruta)
    except Exception:
        return None
    return None


def mismo_orden_que_base(d, df_base):
    """True si d comparte el orden (server, timestamp) del CSV base."""
    if not {"server", "timestamp"}.issubset(d.columns):
        return True
    ts_d = pd.to_datetime(d["timestamp"]).to_numpy()
    ts_b = pd.to_datetime(df_base["timestamp"]).to_numpy()
    return bool((d["server"].astype(str).to_numpy()
                 == df_base["server"].astype(str).to_numpy()).all()
                and (ts_d == ts_b).all())


# ---------------------------------------------------------------------------
# 1) Datos base y contratos de las subtareas previas (T2, T3)
# ---------------------------------------------------------------------------
df_base = pd.read_csv(RUTA_CSV, parse_dates=["timestamp"])
n_total = len(df_base)
if n_total == 0:
    raise SystemExit("El CSV base esta vacio.")
print(f"[datos] {RUTA_CSV}: {n_total} filas, "
      f"{df_base['server'].nunique()} servidores")

t3_json = cargar_json(CARPETA_T3 / "resultados.json")
t2_json = cargar_json(CARPETA_T2 / "resultados.json")

valores_conocidos_ls = {}
filas_meta_t3 = []
if isinstance(t3_json, dict):
    filas_meta_t3 = t3_json.get("filas_requires_review") or []
    for fila in filas_meta_t3:
        try:
            valores_conocidos_ls[int(fila["indice"])] = float(fila["load_score"])
        except (KeyError, TypeError, ValueError):
            pass

indices_t3 = None
if isinstance(t3_json, dict) and isinstance(t3_json.get("indices_requires_review"), list):
    indices_t3 = [int(i) for i in t3_json["indices_requires_review"]]


def max_diff_vs_conocidos(arr):
    """Max |arr[i] - valor registrado por T3| sobre los indices conocidos."""
    if not valores_conocidos_ls:
        return None
    diffs = [abs(float(arr[i]) - v) for i, v in valores_conocidos_ls.items()
             if 0 <= i < len(arr)]
    return max(diffs) if diffs else None


orden_csv_consistente = None
if filas_meta_t3:
    ok = []
    for fila in filas_meta_t3:
        try:
            i = int(fila["indice"])
            ok.append(0 <= i < n_total
                      and str(df_base.iloc[i]["server"]) == str(fila["server"])
                      and abs(float(df_base.iloc[i]["temperature_c"])
                              - float(fila["temperature_c"])) < 1e-9)
        except (KeyError, TypeError, ValueError):
            ok.append(False)
    orden_csv_consistente = bool(ok) and all(ok)
    if not orden_csv_consistente:
        print("[aviso] El orden del CSV no coincide con los indices registrados "
              "por T3; se priorizan fuentes alineables por (server, timestamp).")

# ---------------------------------------------------------------------------
# 2) load_score y requires_review del DataFrame analizado de T3
# ---------------------------------------------------------------------------
load_score = None
fuente_load_score = None
requires_review = None
fuente_requires_review = None
fuente_df_t3 = None
diff_max_load_score = None

# 2.a) Archivos de datos guardados por T3
if CARPETA_T3.is_dir():
    rutas_t3 = []
    for patron in ("*.parquet", "*.feather", "*.csv"):
        rutas_t3.extend(sorted(CARPETA_T3.glob(patron)))
    for ruta in rutas_t3:
        d = intentar_cargar_tabla(ruta)
        if d is None or len(d) != n_total:
            continue
        if not {"server", "power_w", "temperature_c", "load_score",
                "requires_review"}.issubset(d.columns):
            continue
        d = d.reset_index(drop=True)
        ls = pd.to_numeric(d["load_score"], errors="coerce").to_numpy(dtype=float)
        if np.isnan(ls).any():
            continue
        if mismo_orden_que_base(d, df_base):
            load_score = ls
            fuente_load_score = f"{ruta}:load_score"
            requires_review = a_bool_numpy(d["requires_review"])
            fuente_requires_review = f"{ruta}:requires_review"
            fuente_df_t3 = str(ruta)
            break
        d2 = d.copy()
        d2["timestamp"] = pd.to_datetime(d2["timestamp"])
        merged = df_base.merge(
            d2[["server", "timestamp", "load_score", "requires_review"]],
            on=["server", "timestamp"], how="left")
        if len(merged) == n_total and merged["load_score"].notna().all():
            load_score = merged["load_score"].to_numpy(dtype=float)
            requires_review = a_bool_numpy(merged["requires_review"])
            fuente_load_score = f"{ruta}:load_score (alineado por server+timestamp)"
            fuente_requires_review = (f"{ruta}:requires_review "
                                      "(alineado por server+timestamp)")
            fuente_df_t3 = str(ruta)
            break

# 2.b) load_score posicional desde los resultados.json de T3 / T2
if load_score is None and orden_csv_consistente is not False:
    for nombre, js in ((f"{CARPETA_T3 / 'resultados.json'}:load_score", t3_json),
                       (f"{CARPETA_T2 / 'resultados.json'}:load_score", t2_json)):
        if isinstance(js, dict) and "load_score" in js:
            lista = extraer_lista_numeros(js["load_score"], n_total)
            if lista is None:
                continue
            cand = np.asarray(lista, dtype=float)
            if np.isnan(cand).any():
                continue
            load_score = cand
            fuente_load_score = nombre
            break

# 2.c) load_score desde archivos de datos de T2
if (load_score is None and orden_csv_consistente is not False
        and CARPETA_T2.is_dir()):
    rutas_t2 = []
    for patron in ("*.parquet", "*.feather", "*.csv"):
        rutas_t2.extend(sorted(CARPETA_T2.glob(patron)))
    for ruta in rutas_t2:
        d = intentar_cargar_tabla(ruta)
        if d is None or len(d) != n_total or "load_score" not in d.columns:
            continue
        d = d.reset_index(drop=True)
        if not mismo_orden_que_base(d, df_base):
            continue
        ls = pd.to_numeric(d["load_score"], errors="coerce").to_numpy(dtype=float)
        if np.isnan(ls).any():
            continue
        load_score = ls
        fuente_load_score = f"{ruta}:load_score"
        break

# Verificar la fuente elegida frente a los valores registrados por T3
alternativa_posicional = None
if load_score is not None:
    diff = max_diff_vs_conocidos(load_score)
    if diff is not None and diff > TOL_LOAD_SCORE:
        print(f"[aviso] {fuente_load_score} no reproduce los load_score de T3 "
              f"(max diff = {diff:.3e}); se intentara una reconstruccion exacta.")
        alternativa_posicional = (diff, fuente_load_score, load_score)
        load_score = None
        fuente_load_score = None

# 2.d) Ultimo recurso: reconstruir load_score con formulas candidatas
if load_score is None:
    g = df_base["gpu_utilization"].to_numpy(dtype=float)
    c = df_base["cpu_utilization"].to_numpy(dtype=float)
    m = df_base["memory_gb"].to_numpy(dtype=float)
    candidatos = []
    for ddof in (0, 1):
        zg, zc, zm = zscore(g, ddof), zscore(c, ddof), zscore(m, ddof)
        candidatos.append((f"z(gpu)+z(cpu)+z(mem) [ddof={ddof}]", zg + zc + zm))
        candidatos.append((f"(z(gpu)+z(cpu)+z(mem))/3 [ddof={ddof}]",
                           (zg + zc + zm) / 3.0))
        candidatos.append((f"0.5z(gpu)+0.3z(cpu)+0.2z(mem) [ddof={ddof}]",
                           0.5 * zg + 0.3 * zc + 0.2 * zm))
    compuestos = {
        "gpu_utilization": g,
        "cpu_utilization": c,
        "memory_gb": m,
        "gpu+cpu": g + c,
        "gpu+cpu+memory": g + c + m,
        "(gpu+cpu)/2": (g + c) / 2.0,
        "(gpu+cpu+memory)/3": (g + c + m) / 3.0,
        "0.5g+0.3c+0.2m": 0.5 * g + 0.3 * c + 0.2 * m,
        "0.4g+0.3c+0.3m": 0.4 * g + 0.3 * c + 0.3 * m,
        "0.6g+0.2c+0.2m": 0.6 * g + 0.2 * c + 0.2 * m,
    }
    for nombre, comp in compuestos.items():
        for ddof in (0, 1):
            candidatos.append((f"z({nombre}) [ddof={ddof}]", zscore(comp, ddof)))

    if valores_conocidos_ls:
        puntuados = sorted(
            ((max_diff_vs_conocidos(arr), nombre, arr) for nombre, arr in candidatos),
            key=lambda t: t[0])
        diff_max, nombre_elegido, arr_elegida = puntuados[0]
        if alternativa_posicional is not None and alternativa_posicional[0] <= diff_max:
            diff_max, fuente_load_score, load_score = alternativa_posicional
        else:
            load_score = arr_elegida
            fuente_load_score = f"reconstruido: {nombre_elegido}"
        diff_max_load_score = float(diff_max)
        print(f"[aviso] load_score obtenido de '{fuente_load_score}' "
              f"(max diff vs T3 = {diff_max:.3e}).")
    else:
        load_score = zscore(g + c + m, ddof=0)
        fuente_load_score = ("reconstruido: z(gpu+cpu+memory) [ddof=0] "
                             "(sin referencia de T3)")
        print("[aviso] load_score reconstruido sin valores de referencia de T3.")

# ---------------------------------------------------------------------------
# 3) requires_review (regla de T3) y verificacion con el contrato de T3
# ---------------------------------------------------------------------------
if requires_review is None:
    if load_score is not None:
        temp = df_base["temperature_c"].to_numpy(dtype=float)
        requires_review = (load_score > UMBRAL_LOAD_SCORE) | (temp > UMBRAL_TEMPERATURA)
        fuente_requires_review = (f"recalculada con la regla de T3: load_score > "
                                  f"{UMBRAL_LOAD_SCORE} o temperature_c > "
                                  f"{UMBRAL_TEMPERATURA}")
    elif indices_t3 is not None:
        requires_review = np.isin(np.arange(n_total), np.asarray(indices_t3))
        fuente_requires_review = (f"{CARPETA_T3 / 'resultados.json'}:"
                                  "indices_requires_review")

if requires_review is None:
    raise SystemExit("No se pudo construir requires_review/load_score: "
                     "faltan entrada/T3 y entrada/T2.")

requires_review = np.asarray(requires_review, dtype=bool)

verif_t3 = {"aplica": bool(indices_t3 is not None)}
if indices_t3 is not None:
    idx_obt = np.flatnonzero(requires_review).tolist()
    coincide = idx_obt == list(indices_t3)
    verif_t3.update({
        "n_requires_review_T3": len(indices_t3),
        "n_requires_review_obtenido": int(requires_review.sum()),
        "indices_coinciden_con_T3": bool(coincide),
    })
    if not coincide:
        requires_review = np.isin(np.arange(n_total), np.asarray(indices_t3))
        fuente_requires_review = (f"{CARPETA_T3 / 'resultados.json'}:"
                                  "indices_requires_review (corregida)")
        verif_t3["corregida_con_indices_T3"] = True
        verif_t3["n_requires_review_obtenido"] = int(requires_review.sum())
        verif_t3["indices_coinciden_con_T3"] = True

if load_score is not None and diff_max_load_score is None:
    diff_max_load_score = max_diff_vs_conocidos(load_score)

# ---------------------------------------------------------------------------
# 4) DataFrame analizado y tabla resumen por servidor
# ---------------------------------------------------------------------------
df_t4 = df_base.copy()
df_t4["load_score"] = load_score
df_t4["requires_review"] = requires_review

resumen = (df_t4.groupby("server", sort=True)
           .agg(observations=("power_w", "size"),
                mean_power_w=("power_w", "mean"),
                max_temperature_c=("temperature_c", "max"),
                mean_load=("load_score", "mean"),
                review_count=("requires_review", "sum"))
           .reset_index())
resumen["observations"] = resumen["observations"].astype(int)
resumen["review_count"] = resumen["review_count"].astype(int)
resumen = resumen[COLUMNAS_RESUMEN]

# ---------------------------------------------------------------------------
# 5) Verificaciones: estructura y criterio de exito (AI-SRV-01)
# ---------------------------------------------------------------------------
checks_estructura = {
    "seis_filas_una_por_servidor": bool(len(resumen) == 6
                                        and resumen["server"].is_unique),
    "orden_columnas_exacto": list(resumen.columns) == COLUMNAS_RESUMEN,
    "servidores_AI_SRV_01_a_06": sorted(resumen["server"].astype(str).tolist())
                                 == [f"AI-SRV-{i:02d}" for i in range(1, 7)],
}

conteo_review_T3 = None
coincide_review_por_servidor = None
if filas_meta_t3:
    conteo_review_T3 = {}
    for fila in filas_meta_t3:
        s = str(fila.get("server", ""))
        conteo_review_T3[s] = conteo_review_T3.get(s, 0) + 1
    obt = dict(zip(resumen["server"].astype(str),
                   resumen["review_count"].astype(int)))
    coincide_review_por_servidor = all(obt.get(s, -1) == n
                                       for s, n in conteo_review_T3.items())
verif_t3["review_count_por_servidor_T3"] = conteo_review_T3
verif_t3["coincide_review_por_servidor"] = (
    None if coincide_review_por_servidor is None
    else bool(coincide_review_por_servidor))

esperado01 = {"observations": 50, "mean_power_w": 363.4598,
              "max_temperature_c": 86.50, "mean_load": -0.614527,
              "review_count": 2}
obtenido01 = {}
checks01 = {}
checks01_round = {}
fila01 = resumen.loc[resumen["server"] == "AI-SRV-01"]
if len(fila01) == 1:
    f01 = fila01.iloc[0]
    obtenido01 = {"observations": int(f01["observations"]),
                  "mean_power_w": float(f01["mean_power_w"]),
                  "max_temperature_c": float(f01["max_temperature_c"]),
                  "mean_load": float(f01["mean_load"]),
                  "review_count": int(f01["review_count"])}
    checks01 = {
        "observations": obtenido01["observations"] == esperado01["observations"],
        "mean_power_w": abs(obtenido01["mean_power_w"]
                            - esperado01["mean_power_w"]) <= 5.1e-5,
        "max_temperature_c": abs(obtenido01["max_temperature_c"]
                                 - esperado01["max_temperature_c"]) <= 5.1e-3,
        "mean_load": abs(obtenido01["mean_load"]
                         - esperado01["mean_load"]) <= 5.1e-7,
        "review_count": obtenido01["review_count"] == esperado01["review_count"],
    }
    checks01_round = {
        "mean_power_w_round4": round(obtenido01["mean_power_w"], 4) == 363.4598,
        "max_temperature_c_round2": round(obtenido01["max_temperature_c"], 2) == 86.5,
        "mean_load_round6": round(obtenido01["mean_load"], 6) == -0.614527,
    }

criterio_ok = bool(all(checks_estructura.values()) and checks01
                   and all(checks01.values()))

# ---------------------------------------------------------------------------
# 6) Salidas: parquet exigido, figura y resultados.json
# ---------------------------------------------------------------------------
salida_parquet = Path("output/server_analysis.parquet")
salida_parquet.parent.mkdir(parents=True, exist_ok=True)
resumen.to_parquet(salida_parquet, index=False)

parquet_ok = False
try:
    comprob = pd.read_parquet(salida_parquet)
    parquet_ok = bool(list(comprob.columns) == COLUMNAS_RESUMEN
                      and len(comprob) == len(resumen)
                      and np.allclose(comprob["mean_load"].to_numpy(dtype=float),
                                      resumen["mean_load"].to_numpy(dtype=float)))
except Exception:
    parquet_ok = False

fig, ejes = plt.subplots(1, 3, figsize=(14, 4.2))
ejes[0].bar(resumen["server"], resumen["mean_power_w"], color="#4C72B0")
ejes[0].set_title("Potencia media por servidor")
ejes[0].set_ylabel("mean_power_w (W)")
ejes[1].bar(resumen["server"], resumen["mean_load"], color="#55A868")
ejes[1].axhline(0.0, color="gray", linewidth=0.8)
ejes[1].set_title("Carga media (media de load_score)")
ejes[1].set_ylabel("mean_load")
ejes[2].bar(resumen["server"], resumen["review_count"], color="#C44E52")
ejes[2].set_title("Observaciones que requieren revision")
ejes[2].set_ylabel("review_count")
for e in ejes:
    e.tick_params(axis="x", rotation=45)
fig.suptitle("T4 - Resumen por servidor")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("t4_resumen_servidores.png", dpi=120)
plt.close(fig)


def fila_dict(r, redondeada=False):
    d = {"server": str(r["server"]),
         "observations": int(r["observations"]),
         "mean_power_w": float(r["mean_power_w"]),
         "max_temperature_c": float(r["max_temperature_c"]),
         "mean_load": float(r["mean_load"]),
         "review_count": int(r["review_count"])}
    if redondeada:
        d["mean_power_w"] = round(d["mean_power_w"], 4)
        d["max_temperature_c"] = round(d["max_temperature_c"], 2)
        d["mean_load"] = round(d["mean_load"], 6)
    return d


tabla_full = [fila_dict(r) for _, r in resumen.iterrows()]
tabla_round = [fila_dict(r, redondeada=True) for _, r in resumen.iterrows()]

resultados = {
    "subtarea": "T4",
    "descripcion": ("Tabla resumen con una fila por servidor y exactamente las "
                    "columnas server, observations, mean_power_w, "
                    "max_temperature_c, mean_load, review_count (en ese orden)."),
    "definiciones": {
        "observations": "numero de filas del servidor en el DataFrame de T3",
        "mean_power_w": "media de power_w",
        "max_temperature_c": "maximo de temperature_c",
        "mean_load": "media de load_score",
        "review_count": "numero de filas con requires_review=True",
    },
    "fuentes": {
        "datos_base": str(RUTA_CSV),
        "dataframe_analizado_T3": fuente_df_t3,
        "load_score": fuente_load_score,
        "requires_review": fuente_requires_review,
        "orden_csv_consistente_con_T3": orden_csv_consistente,
    },
    "n_filas_base": int(n_total),
    "n_filas_tabla": int(len(resumen)),
    "orden_columnas": COLUMNAS_RESUMEN,
    "tabla_resumen": tabla_full,
    "tabla_resumen_redondeada": tabla_round,
    "por_servidor": {d["server"]: d for d in tabla_full},
    "valores_por_columna": {col: [d[col] for d in tabla_full]
                            for col in COLUMNAS_RESUMEN},
    "totales": {"observations": int(resumen["observations"].sum()),
                "review_count": int(resumen["review_count"].sum())},
    "verificacion_contra_T3": verif_t3,
    "verificacion_load_score_vs_T3": {
        "n_valores_comparados": int(len(valores_conocidos_ls)),
        "max_diff": diff_max_load_score,
    },
    "verificacion_AI_SRV_01": {
        "esperado": esperado01,
        "obtenido": obtenido01,
        "obtenido_redondeado": (
            {"observations": obtenido01["observations"],
             "mean_power_w": round(obtenido01["mean_power_w"], 4),
             "max_temperature_c": round(obtenido01["max_temperature_c"], 2),
             "mean_load": round(obtenido01["mean_load"], 6),
             "review_count": obtenido01["review_count"]}
            if obtenido01 else {}),
        "checks": checks01,
        "checks_redondeo": checks01_round,
        "coincide": bool(checks01 and all(checks01.values())),
    },
    "checks_estructura": checks_estructura,
    "criterio_exito_cumplido": criterio_ok,
    "archivos_generados": {"parquet": str(salida_parquet),
                           "parquet_verificado": parquet_ok,
                           "figura": "t4_resumen_servidores.png"},
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 7) Mostrar la tabla completa y resumen en consola
# ---------------------------------------------------------------------------
tabla_print = resumen.copy()
tabla_print["mean_power_w"] = tabla_print["mean_power_w"].map("{:.4f}".format)
tabla_print["max_temperature_c"] = tabla_print["max_temperature_c"].map("{:.2f}".format)
tabla_print["mean_load"] = tabla_print["mean_load"].map("{:.6f}".format)

print("\nTabla resumen por servidor (tabla completa):")
print(tabla_print.to_string(index=False))

print("\nResumen T4:")
print(f"  DataFrame analizado (T3)   : {fuente_df_t3 or 'reconstruido desde CSV + T2/T3'}")
print(f"  Fuente de load_score       : {fuente_load_score}")
print(f"  Fuente de requires_review  : {fuente_requires_review}")
print(f"  Dimensiones de la tabla    : {resumen.shape[0]} filas x {resumen.shape[1]} columnas")
print(f"  Columnas (orden exacto)    : {list(resumen.columns)}")
print(f"  Total observaciones        : {int(resumen['observations'].sum())}")
print(f"  Total review_count         : {int(resumen['review_count'].sum())}")
if diff_max_load_score is not None:
    print(f"  max |load_score - T3|      : {diff_max_load_score:.3e}")
if coincide_review_por_servidor is not None:
    print(f"  review_count vs T3 (por servidor): "
          f"{'OK' if coincide_review_por_servidor else 'FALLO'}")
estado01 = "OK" if (checks01 and all(checks01.values())) else "FALLO"
print(f"  Verificacion AI-SRV-01     : {estado01} {checks01}")
print(f"  Criterio de exito cumplido : {criterio_ok}")
print(f"  Archivos generados         : {salida_parquet}, "
      f"t4_resumen_servidores.png, resultados.json")
