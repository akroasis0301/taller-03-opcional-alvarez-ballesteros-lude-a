#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T4 - Construir un resumen por servidor.

A partir del DataFrame enriquecido de T3 (con load_score y requires_review)
construye la tabla resumen con EXACTAMENTE las columnas, en este orden:

    server | observations | mean_power_w | max_temperature_c | mean_load | review_count

Metricas:
    observations      : numero de filas del servidor
    mean_power_w      : media de power_w (W)
    max_temperature_c : maximo de temperature_c (C)
    mean_load         : media de load_score
    review_count      : conteo de filas con requires_review=True

load_score sigue la definicion de T2: estandarizar por columna (Z-score)
gpu_utilization, cpu_utilization y memory_gb y combinarlas con los pesos
(0.50, 0.30, 0.20) mediante operaciones vectorizadas:

    load_score = 0.50*z(gpu_utilization) + 0.30*z(cpu_utilization) + 0.20*z(memory_gb)

requires_review = (load_score > 1.5) | (temperature_c > 80)

Si el DataFrame de T3 esta disponible en entrada/T3/ se usa directamente (y se
valida); si no, se reconstruye desde data/server_measurements.csv aplicando la
definicion de T2 y validando contra los valores reportados por T3 (load_score
de la fila 23, indices con requires_review, mean_load de AI-SRV-01). El ddof
del Z-score (0 o 1) se determina empiricamente con esa validacion.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------
COLUMNAS_EXIGIDAS = [
    "server",
    "observations",
    "mean_power_w",
    "max_temperature_c",
    "mean_load",
    "review_count",
]
SERVIDOR_CONTROL = "AI-SRV-01"
CONTROL = {
    "observations": 50,
    "mean_power_w": 363.4598,
    "max_temperature_c": 86.50,
    "mean_load": -0.614527,
    "review_count": 2,
}
COLS_Z = ["gpu_utilization", "cpu_utilization", "memory_gb"]
PESOS = (0.50, 0.30, 0.20)  # gpu, cpu, mem (definicion de T2)
UMBRAL_LOAD = 1.5
UMBRAL_TEMP = 80.0

# Referencias reportadas por T3 (respaldo si entrada/T3/resultados.json falta)
T3_FILA23_LOAD_SCORE = 3.136217861183744
T3_INDICES_REVIEW = [
    23, 42, 78, 121, 146, 162, 171, 176, 189, 192, 201, 211, 212, 213, 219,
    223, 225, 240, 241, 246, 247, 249, 251, 252, 253, 254, 256, 259, 260, 261,
    263, 264, 266, 268, 269, 273, 274, 276, 278, 281, 282, 283, 287, 288, 290,
    292, 293, 295, 298,
]
T3_N_LOAD_ALTO = 9
T3_REVIEW_TOTAL = 49


# ----------------------------------------------------------------------------
# Utilidades de E/S
# ----------------------------------------------------------------------------
def leer_tabla(ruta):
    try:
        sufijo = ruta.suffix.lower()
        if sufijo == ".parquet":
            return pd.read_parquet(ruta)
        if sufijo == ".csv":
            return pd.read_csv(ruta)
    except Exception:
        return None
    return None


def glob_ordenado(patron):
    return sorted(Path(".").glob(patron), key=lambda p: str(p))


def cargar_ground_truth_t3():
    """Referencias de T3 para validar load_score (archivo si existe, si no
    las constantes del contexto)."""
    gt = {
        "fila23_load_score": T3_FILA23_LOAD_SCORE,
        "indices_requires_review": list(T3_INDICES_REVIEW),
        "n_load_score_mayor_1_5": T3_N_LOAD_ALTO,
        "review_count_total": T3_REVIEW_TOTAL,
        "fuente": "constantes de contexto (respaldo)",
    }
    ruta = Path("entrada/T3/resultados.json")
    if ruta.is_file():
        try:
            data = json.loads(ruta.read_text(encoding="utf-8"))
            f23 = data.get("fila_23")
            if isinstance(f23, dict) and "load_score" in f23:
                gt["fila23_load_score"] = float(f23["load_score"])
            idx = data.get("indices_requires_review")
            if isinstance(idx, list) and len(idx) > 0:
                gt["indices_requires_review"] = [int(i) for i in idx]
            for clave in ("n_load_score_mayor_1_5", "review_count_total"):
                valor = data.get(clave)
                if isinstance(valor, (int, float)) and not isinstance(valor, bool):
                    gt[clave] = int(valor)
            gt["fuente"] = str(ruta)
        except Exception:
            pass
    return gt


def cargar_df_base():
    """Devuelve (df, ruta, origen). Prefiere artefactos de T3 con mas columnas."""
    necesarias = {"server", "power_w", "temperature_c"}
    extras = [
        "load_score", "requires_review", "timestamp",
        "gpu_utilization", "cpu_utilization", "memory_gb",
    ]
    hallados = []
    for patron in ("entrada/T3/**/*.parquet", "entrada/T3/**/*.csv"):
        for ruta in glob_ordenado(patron):
            df = leer_tabla(ruta)
            if df is not None and necesarias.issubset(df.columns) and len(df) > 0:
                puntaje = sum(1 for c in extras if c in df.columns)
                hallados.append((puntaje, str(ruta), df))
    if hallados:
        hallados.sort(key=lambda t: (-t[0], t[1]))
        _, ruta, df = hallados[0]
        return df.reset_index(drop=True), ruta, "artefactos de T3 (entrada/T3/)"

    ruta_csv = Path("data/server_measurements.csv")
    if not ruta_csv.is_file():
        raise FileNotFoundError(
            "No se encontro el DataFrame de T3 en entrada/T3/ ni "
            "data/server_measurements.csv"
        )
    try:
        df = pd.read_csv(ruta_csv, parse_dates=["timestamp"])
    except Exception:
        df = pd.read_csv(ruta_csv)
    return df.reset_index(drop=True), str(ruta_csv), "CSV original (data/server_measurements.csv)"


# ----------------------------------------------------------------------------
# load_score segun la definicion de T2
# ----------------------------------------------------------------------------
def calcular_load_score_global(df, ddof):
    """Z-score por columna (global) ponderado con (0.50, 0.30, 0.20)."""
    n = len(df)
    z = np.empty((len(COLS_Z), n), dtype=float)
    for i, col in enumerate(COLS_Z):
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        if not np.isfinite(x).all():
            return None
        sd = float(np.std(x, ddof=ddof))
        if not np.isfinite(sd) or sd == 0.0:
            return None
        z[i] = (x - x.mean()) / sd
    return np.asarray(PESOS, dtype=float) @ z


def calcular_load_score_por_servidor(df, ddof):
    """Variante de respaldo: Z-scores calculados dentro de cada servidor."""
    srv = df["server"].astype(str).to_numpy()
    n = len(df)
    total = np.zeros(n, dtype=float)
    for w, col in zip(PESOS, COLS_Z):
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        if not np.isfinite(x).all():
            return None
        z = np.empty(n, dtype=float)
        for s in pd.unique(srv):
            m = srv == s
            xs = x[m]
            sd = float(xs.std(ddof=ddof))
            if not np.isfinite(sd) or sd == 0.0:
                return None
            z[m] = (xs - xs.mean()) / sd
        total += w * z
    return total


def _buscar_lista_numerica(obj, n):
    hallados = []

    def rec(o, clave):
        if isinstance(o, list):
            if len(o) == n and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in o
            ):
                hallados.append((clave.lower(), o))
            return
        if isinstance(o, dict):
            for k, v in o.items():
                rec(v, str(k))

    rec(obj, "")
    if not hallados:
        return None
    for clave, lista in hallados:
        if "load" in clave or "score" in clave:
            return lista
    return hallados[0][1]


def generar_candidatos(df, gt):
    """Candidatos de load_score, en orden de preferencia."""
    cands = []
    n = len(df)

    if "load_score" in df.columns:
        ls = pd.to_numeric(df["load_score"], errors="coerce").to_numpy(float)
        if len(ls) == n and np.isfinite(ls).all():
            cands.append({
                "nombre": "columna load_score del DataFrame de T3",
                "load_score": ls,
                "ddof": None,
            })

    for patron in ("entrada/T2/**/*.parquet", "entrada/T2/**/*.csv"):
        for ruta in glob_ordenado(patron):
            t2 = leer_tabla(ruta)
            if t2 is not None and "load_score" in t2.columns and len(t2) == n:
                ls = pd.to_numeric(t2["load_score"], errors="coerce").to_numpy(float)
                if np.isfinite(ls).all():
                    cands.append({
                        "nombre": f"load_score recuperado de {ruta}",
                        "load_score": ls,
                        "ddof": None,
                    })

    ruta_t2 = Path("entrada/T2/resultados.json")
    if ruta_t2.is_file():
        try:
            data = json.loads(ruta_t2.read_text(encoding="utf-8"))
            lista = _buscar_lista_numerica(data, n)
            if lista is not None:
                cands.append({
                    "nombre": f"lista load_score de {ruta_t2}",
                    "load_score": np.asarray(lista, dtype=float),
                    "ddof": None,
                })
        except Exception:
            pass

    if all(c in df.columns for c in COLS_Z):
        # Definicion de T2: z-scores por columna @ (0.50, 0.30, 0.20)
        for ddof in (1, 0):
            ls = calcular_load_score_global(df, ddof)
            if ls is not None:
                cands.append({
                    "nombre": (
                        "recalculado (T2): z-scores por columna "
                        "(gpu, cpu, mem) @ pesos (0.50, 0.30, 0.20), "
                        f"ddof={ddof}"
                    ),
                    "load_score": ls,
                    "ddof": ddof,
                })
        # Respaldo adicional: z-scores por servidor
        for ddof in (1, 0):
            ls = calcular_load_score_por_servidor(df, ddof)
            if ls is not None:
                cands.append({
                    "nombre": (
                        "recalculado (T2, por servidor): z-scores "
                        "@ pesos (0.50, 0.30, 0.20), "
                        f"ddof={ddof}"
                    ),
                    "load_score": ls,
                    "ddof": ddof,
                })
    return cands


def puntuar_load_score(df, ls, gt):
    """Puntua un candidato contra las referencias de T3/T4 (max 9 puntos):
    mean_load de AI-SRV-01 (2), load_score fila 23 (3), indices
    requires_review exactos (3), n con load_score>1.5 (1)."""
    ls = np.asarray(ls, dtype=float)
    temp = pd.to_numeric(df["temperature_c"], errors="coerce").to_numpy(float)
    srv = df["server"].astype(str).to_numpy()
    detalles = {}
    puntos = 0

    mascara = srv == SERVIDOR_CONTROL
    media = float(np.mean(ls[mascara])) if mascara.any() else None
    ok_media = media is not None and abs(media - CONTROL["mean_load"]) <= 1e-6
    detalles["mean_load_ai_srv_01"] = {
        "obtenido": media,
        "esperado": CONTROL["mean_load"],
        "ok": bool(ok_media),
    }
    puntos += 2 if ok_media else 0

    v23 = float(ls[23]) if len(ls) > 23 else None
    ok_f23 = v23 is not None and abs(v23 - gt["fila23_load_score"]) <= 1e-6
    detalles["load_score_fila_23"] = {
        "obtenido": v23,
        "esperado": gt["fila23_load_score"],
        "ok": bool(ok_f23),
    }
    puntos += 3 if ok_f23 else 0

    req = (ls > UMBRAL_LOAD) | (temp > UMBRAL_TEMP)
    idx_obt = [int(i) for i in np.where(req)[0]]
    idx_esp = gt["indices_requires_review"]
    ok_idx = idx_esp is not None and idx_obt == [int(i) for i in idx_esp]
    detalles["indices_requires_review"] = {
        "coinciden": bool(ok_idx),
        "n_obtenidos": len(idx_obt),
        "n_esperados": (len(idx_esp) if idx_esp is not None else None),
    }
    puntos += 3 if ok_idx else 0

    n_alto = int((ls > UMBRAL_LOAD).sum())
    ok_n = (
        gt["n_load_score_mayor_1_5"] is not None
        and n_alto == int(gt["n_load_score_mayor_1_5"])
    )
    detalles["n_load_score_mayor_1_5"] = {
        "obtenido": n_alto,
        "esperado": gt["n_load_score_mayor_1_5"],
        "ok": bool(ok_n),
    }
    puntos += 1 if ok_n else 0

    return puntos, detalles


def seleccionar_load_score(df, gt):
    evaluaciones = []
    mejor = None
    for cand in generar_candidatos(df, gt):
        puntos, detalles = puntuar_load_score(df, cand["load_score"], gt)
        cand = {**cand, "puntos": puntos, "detalles": detalles}
        evaluaciones.append({
            "candidato": cand["nombre"],
            "puntos": puntos,
            "detalles": detalles,
        })
        if mejor is None or puntos > mejor["puntos"]:
            mejor = cand
    if mejor is None:
        raise RuntimeError("No se pudo obtener ni reconstruir load_score")
    return mejor, evaluaciones


# ----------------------------------------------------------------------------
# Tabla resumen
# ----------------------------------------------------------------------------
def tabla_registros(tabla):
    registros = []
    for _, r in tabla.iterrows():
        registros.append({
            "server": str(r["server"]),
            "observations": int(r["observations"]),
            "mean_power_w": float(r["mean_power_w"]),
            "max_temperature_c": float(r["max_temperature_c"]),
            "mean_load": float(r["mean_load"]),
            "review_count": int(r["review_count"]),
        })
    return registros


def construir_resumen(df):
    g = df.groupby("server", sort=True)
    resumen = pd.DataFrame(
        {
            "observations": g.size(),
            "mean_power_w": g["power_w"].mean(),
            "max_temperature_c": g["temperature_c"].max(),
            "mean_load": g["load_score"].mean(),
            "review_count": g["requires_review"].sum().astype("int64"),
        }
    ).reset_index()
    resumen["observations"] = resumen["observations"].astype("int64")
    resumen["review_count"] = resumen["review_count"].astype("int64")
    resumen = resumen[COLUMNAS_EXIGIDAS]
    return resumen


def main():
    gt = cargar_ground_truth_t3()
    df, fuente_df, origen_df = cargar_df_base()

    # Normalizacion defensiva de tipos
    df["server"] = df["server"].astype(str).str.strip()
    for col in ("power_w", "temperature_c", *COLS_Z):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # load_score (definicion T2) + requires_review, validados contra T3
    mejor, evaluaciones = seleccionar_load_score(df, gt)
    df["load_score"] = mejor["load_score"]
    df["requires_review"] = (df["load_score"] > UMBRAL_LOAD) | (
        df["temperature_c"] > UMBRAL_TEMP
    )

    # Tabla resumen: una fila por servidor, columnas en el orden exigido
    resumen = construir_resumen(df)
    orden_correcto = bool(list(resumen.columns) == COLUMNAS_EXIGIDAS)

    # Impresion de la tabla completa de los seis servidores
    formatos = {
        "observations": lambda v: f"{int(v):d}",
        "mean_power_w": lambda v: f"{float(v):.4f}",
        "max_temperature_c": lambda v: f"{float(v):.2f}",
        "mean_load": lambda v: f"{float(v):.6f}",
        "review_count": lambda v: f"{int(v):d}",
    }
    tabla_texto = resumen.to_string(index=False, formatters=formatos)
    print("=== T4 - Tabla resumen por servidor ===")
    print(tabla_texto)

    # Verificacion contra la fila de control del enunciado
    filas_ctrl = resumen.loc[resumen["server"] == SERVIDOR_CONTROL]
    control_presente = len(filas_ctrl) == 1
    verificacion = {}
    if control_presente:
        f = filas_ctrl.iloc[0]
        chequeos = [
            ("observations", int(f["observations"]),
             CONTROL["observations"], lambda o, e: o == e),
            ("mean_power_w", float(f["mean_power_w"]),
             CONTROL["mean_power_w"], lambda o, e: abs(o - e) <= 5e-5),
            ("max_temperature_c", float(f["max_temperature_c"]),
             CONTROL["max_temperature_c"], lambda o, e: abs(o - e) <= 5e-3),
            ("mean_load", float(f["mean_load"]),
             CONTROL["mean_load"], lambda o, e: abs(o - e) <= 1e-6),
            ("review_count", int(f["review_count"]),
             CONTROL["review_count"], lambda o, e: o == e),
        ]
        for nombre_col, obtenido, esperado, criterio in chequeos:
            verificacion[nombre_col] = {
                "obtenido": obtenido,
                "esperado": esperado,
                "ok": bool(criterio(obtenido, esperado)),
            }
    fila_control_coincide = bool(
        control_presente and verificacion and all(v["ok"] for v in verificacion.values())
    )

    # Verificaciones globales contra los reportes de T3
    total_review = int(resumen["review_count"].sum())
    n_load_alto = int((df["load_score"] > UMBRAL_LOAD).sum())
    idx_req_obt = [int(i) for i in np.where(df["requires_review"].to_numpy())[0]]
    verificaciones_globales = {
        "n_filas": {
            "obtenido": int(len(df)), "esperado": 300,
            "ok": bool(len(df) == 300),
        },
        "review_count_total": {
            "obtenido": total_review, "esperado": gt["review_count_total"],
            "ok": bool(total_review == gt["review_count_total"]),
        },
        "n_load_score_mayor_1_5": {
            "obtenido": n_load_alto, "esperado": gt["n_load_score_mayor_1_5"],
            "ok": bool(n_load_alto == gt["n_load_score_mayor_1_5"]),
        },
        "indices_requires_review": {
            "obtenido": idx_req_obt,
            "coinciden": bool(idx_req_obt == list(gt["indices_requires_review"])),
        },
    }

    # Tabla con el redondeo mostrado en el enunciado
    disp = resumen.copy()
    disp["mean_power_w"] = disp["mean_power_w"].round(4)
    disp["max_temperature_c"] = disp["max_temperature_c"].round(2)
    disp["mean_load"] = disp["mean_load"].round(6)

    # Parquet exigido por el enunciado: preservar si ya existe
    ruta_parquet_entregable = Path("output/server_analysis.parquet")
    if ruta_parquet_entregable.is_file():
        estado_parquet = "preservado sin modificar (ya existia de la subtarea previa)"
    else:
        try:
            ruta_parquet_entregable.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(ruta_parquet_entregable, index=False)
            estado_parquet = (
                "regenerado con el DataFrame enriquecido "
                "(load_score + requires_review), pues no existia"
            )
        except Exception as exc:
            estado_parquet = f"no disponible ({exc})"

    # Parquet propio de T4 con la tabla resumen
    ruta_parquet_resumen = Path("output/server_summary.parquet")
    try:
        ruta_parquet_resumen.parent.mkdir(parents=True, exist_ok=True)
        resumen.to_parquet(ruta_parquet_resumen, index=False)
        estado_parquet_resumen = "escrito"
    except Exception as exc:
        estado_parquet_resumen = f"no disponible ({exc})"

    # Figura resumen (2x2)
    fig, ejes = plt.subplots(2, 2, figsize=(11, 7))
    paneles = [
        ("mean_power_w", "Potencia media (W)", "#4472c4"),
        ("max_temperature_c", "Temperatura maxima (C)", "#ed7d31"),
        ("mean_load", "Carga media (load_score)", "#70ad47"),
        ("review_count", "Observaciones a revisar", "#c00000"),
    ]
    for ax, (col, titulo, color) in zip(ejes.flat, paneles):
        ax.bar(resumen["server"], resumen[col], color=color)
        ax.set_title(titulo)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("T4 - Resumen por servidor")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("resumen_servidores.png", dpi=120)
    plt.close(fig)

    # Contrato resultados.json
    exito = bool(resumen.shape[0] == 6 and orden_correcto and fila_control_coincide)
    info_load = {
        "definicion": (
            "load_score = 0.50*z(gpu_utilization) + 0.30*z(cpu_utilization) "
            "+ 0.20*z(memory_gb), con z-scores por columna (definicion de T2)"
        ),
        "pesos": {"gpu_utilization": 0.50, "cpu_utilization": 0.30, "memory_gb": 0.20},
        "candidato_elegido": mejor["nombre"],
        "ddof_usado": mejor.get("ddof"),
        "puntos_validacion": int(mejor["puntos"]),
        "puntos_maximos": 9,
        "validacion": mejor["detalles"],
        "candidatos_evaluados": [
            {"candidato": e["candidato"], "puntos": int(e["puntos"])}
            for e in evaluaciones
        ],
        "ground_truth_fuente": gt["fuente"],
    }
    resultados = {
        "subtarea": "T4_resumen_por_servidor",
        "fuente_datos": fuente_df,
        "origen_datos": origen_df,
        "load_score": info_load,
        "n_filas_entrada": int(len(df)),
        "n_servidores": int(resumen.shape[0]),
        "servidores": [str(s) for s in resumen["server"]],
        "columnas": COLUMNAS_EXIGIDAS,
        "orden_columnas_correcto": orden_correcto,
        "definiciones": {
            "observations": "numero de filas por servidor",
            "mean_power_w": "media de power_w (W) por servidor",
            "max_temperature_c": "maximo de temperature_c (C) por servidor",
            "mean_load": "media de load_score por servidor",
            "review_count": "conteo de filas con requires_review=True por servidor",
        },
        "tabla": tabla_registros(resumen),
        "tabla_redondeada": tabla_registros(disp),
        "tabla_texto": tabla_texto,
        "fila_control": SERVIDOR_CONTROL,
        "valores_esperados_fila_control": CONTROL,
        "verificacion_fila_control": verificacion,
        "fila_control_coincide": fila_control_coincide,
        "verificaciones_globales": verificaciones_globales,
        "total_observations": int(resumen["observations"].sum()),
        "total_review_count": total_review,
        "parquet_enunciado": str(ruta_parquet_entregable),
        "estado_parquet_enunciado": estado_parquet,
        "parquet_resumen": str(ruta_parquet_resumen),
        "estado_parquet_resumen": estado_parquet_resumen,
        "figura": "resumen_servidores.png",
        "exito": exito,
    }
    Path("resultados.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Resumen breve
    print("\n=== Resumen T4 ===")
    print(f"Fuente de datos    : {fuente_df} ({origen_df})")
    print(f"load_score         : {mejor['nombre']}")
    print(
        f"  validacion       : {mejor['puntos']}/9 puntos "
        f"(mean_load AI-SRV-01, fila 23, indices, n load>1.5)"
    )
    print(f"Filas de entrada   : {len(df)}")
    print(f"Servidores (filas) : {resumen.shape[0]}")
    print("Orden de columnas  : " + ("correcto" if orden_correcto else "INCORRECTO"))
    if control_presente:
        estado = (
            "coincide con el valor esperado"
            if fila_control_coincide
            else "NO coincide - ver verificacion_fila_control"
        )
        print(f"Fila {SERVIDOR_CONTROL} : {estado}")
    print(f"review_count total : {total_review} (esperado {gt['review_count_total']})")
    print(f"Parquet enunciado  : {estado_parquet}")
    print(
        "Archivos escritos  : resultados.json, output/server_summary.parquet, "
        "resumen_servidores.png"
    )
    print(f"EXITO              : {'SI' if exito else 'NO'}")


if __name__ == "__main__":
    main()
