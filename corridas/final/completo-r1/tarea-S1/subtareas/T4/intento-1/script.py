#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T4 — Construir la tabla resumen por servidor.

Parte del DataFrame de T3 (300 filas con load_score y requires_review), agrupa
por `server` y construye una tabla con UNA fila por servidor y EXACTAMENTE las
columnas, en este orden:

    server, observations, mean_power_w, max_temperature_c, mean_load, review_count

donde:
    observations      -> número de filas del servidor
    mean_power_w      -> media de power_w
    max_temperature_c -> máximo de temperature_c
    mean_load         -> media de load_score
    review_count      -> número de filas con requires_review == True

Escribe resultados.json (contrato de la subtarea) y garantiza la entrega
output/server_analysis.parquet exigida por el enunciado.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNAS_ESPERADAS = [
    "server",
    "observations",
    "mean_power_w",
    "max_temperature_c",
    "mean_load",
    "review_count",
]

ESPERADO_SRV01 = {
    "observations": 50,
    "mean_power_w": 363.4598,
    "max_temperature_c": 86.50,
    "mean_load": -0.614527,
    "review_count": 2,
}

COLS_Z = ["gpu_utilization", "cpu_utilization", "memory_gb"]
REQUERIDAS_T3 = {"server", "power_w", "temperature_c", "load_score", "requires_review"}


def cargar_dataframe_t3():
    """Devuelve (df, descripcion_fuente, reconstruido_desde_csv)."""
    # 1) Parquet de T3 (salida oficial o copia en entrada/T3/)
    candidatos = [Path("output/server_analysis.parquet")]
    t3_dir = Path("entrada/T3")
    if t3_dir.is_dir():
        candidatos.extend(sorted(t3_dir.glob("*.parquet")))

    for ruta in candidatos:
        if ruta.is_file():
            try:
                df = pd.read_parquet(ruta)
            except Exception:
                continue
            if REQUERIDAS_T3.issubset(set(df.columns)):
                return df, f"parquet leído de {ruta.as_posix()}", False

    # 2) Respaldo: reconstruir el DataFrame de T3 desde el CSV original
    csv_path = Path("data/server_measurements.csv")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    params_path = Path("entrada/T3/resultados.json")
    if params_path.is_file():
        with open(params_path, "r", encoding="utf-8") as fh:
            t3 = json.load(fh)
        medias = np.asarray(t3["means_usados"], dtype=float)
        stds = np.asarray(t3["stds_usados"], dtype=float)
        pesos = np.asarray(t3["pesos_usados"], dtype=float)
        fuente_params = params_path.as_posix()
    else:
        X = df[COLS_Z].to_numpy(dtype=float)
        medias = X.mean(axis=0)
        stds = X.std(axis=0)  # ddof=0, solo respaldo
        pesos = np.array([0.50, 0.30, 0.20])
        fuente_params = "calculados del CSV (respaldo)"

    Z = (df[COLS_Z].to_numpy(dtype=float) - medias) / stds
    df["load_score"] = Z @ pesos
    df["requires_review"] = (df["load_score"] > 1.5) | (df["temperature_c"] > 80)
    return (
        df,
        f"reconstruido desde {csv_path.as_posix()} (parámetros: {fuente_params})",
        True,
    )


def main():
    df, fuente, reconstruido = cargar_dataframe_t3()

    # Normalización defensiva de tipos
    df["load_score"] = df["load_score"].astype(float)
    if not pd.api.types.is_bool_dtype(df["requires_review"]):
        df["requires_review"] = (
            df["requires_review"].astype(str).str.lower().isin(["true", "1", "1.0", "yes"])
        )

    # Entrega exigida por el enunciado: output/server_analysis.parquet
    parquet_path = Path("output/server_analysis.parquet")
    parquet_escrito_por_t4 = False
    if not parquet_path.is_file():
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        parquet_escrito_por_t4 = True

    # ---------- Tabla resumen por servidor ----------
    g = df.groupby("server", sort=True)
    resumen = pd.DataFrame(
        {
            "observations": g.size(),
            "mean_power_w": g["power_w"].mean(),
            "max_temperature_c": g["temperature_c"].max(),
            "mean_load": g["load_score"].mean(),
            "review_count": g["requires_review"].sum(),
        }
    ).reset_index()
    resumen["observations"] = resumen["observations"].astype(int)
    resumen["review_count"] = resumen["review_count"].astype(int)
    resumen = resumen[COLUMNAS_ESPERADAS]  # orden exacto exigido

    orden_ok = list(resumen.columns) == COLUMNAS_ESPERADAS
    n_filas = int(len(resumen))
    servidores = sorted(str(s) for s in df["server"].unique())
    una_fila_por_servidor = bool(resumen["server"].is_unique)

    # ---------- Verificación contra el criterio de éxito (AI-SRV-01) ----------
    verif = {"servidor_presente": False}
    filas1 = resumen.loc[resumen["server"] == "AI-SRV-01"]
    if len(filas1) == 1:
        f = filas1.iloc[0]
        verif = {
            "servidor_presente": True,
            "observations": int(f["observations"]),
            "observations_ok": int(f["observations"]) == ESPERADO_SRV01["observations"],
            "mean_power_w": float(f["mean_power_w"]),
            "mean_power_w_redondeado_4": round(float(f["mean_power_w"]), 4),
            "mean_power_w_ok": abs(float(f["mean_power_w"]) - ESPERADO_SRV01["mean_power_w"]) <= 5e-5,
            "max_temperature_c": float(f["max_temperature_c"]),
            "max_temperature_c_redondeado_2": round(float(f["max_temperature_c"]), 2),
            "max_temperature_c_ok": abs(float(f["max_temperature_c"]) - ESPERADO_SRV01["max_temperature_c"]) <= 5e-3,
            "mean_load": float(f["mean_load"]),
            "mean_load_redondeado_6": round(float(f["mean_load"]), 6),
            "mean_load_ok": abs(float(f["mean_load"]) - ESPERADO_SRV01["mean_load"]) <= 5e-7,
            "review_count": int(f["review_count"]),
            "review_count_ok": int(f["review_count"]) == ESPERADO_SRV01["review_count"],
        }

    checks_ok = [v for k, v in verif.items() if k.endswith("_ok")]
    total_obs = int(resumen["observations"].sum())
    total_review = int(resumen["review_count"].sum())
    n_true_df = int(df["requires_review"].sum())

    # Contraste opcional con los conteos reportados por T3
    coincide_t3 = None
    t3_json = Path("entrada/T3/resultados.json")
    if t3_json.is_file():
        try:
            with open(t3_json, "r", encoding="utf-8") as fh:
                t3 = json.load(fh)
            n_true_t3 = int(t3.get("conteos", {}).get("n_requires_review_true", -1))
            if n_true_t3 >= 0:
                coincide_t3 = bool(total_review == n_true_t3)
        except Exception:
            coincide_t3 = None

    criterio = bool(
        n_filas == 6
        and orden_ok
        and una_fila_por_servidor
        and verif.get("servidor_presente", False)
        and all(checks_ok)
    )

    # ---------- Serialización (contrato resultados.json) ----------
    tabla = []
    for _, r in resumen.iterrows():
        tabla.append(
            {
                "server": str(r["server"]),
                "observations": int(r["observations"]),
                "mean_power_w": float(r["mean_power_w"]),
                "max_temperature_c": float(r["max_temperature_c"]),
                "mean_load": float(r["mean_load"]),
                "review_count": int(r["review_count"]),
            }
        )
    tabla_redondeada = [
        {
            "server": t["server"],
            "observations": t["observations"],
            "mean_power_w": round(t["mean_power_w"], 4),
            "max_temperature_c": round(t["max_temperature_c"], 2),
            "mean_load": round(t["mean_load"], 6),
            "review_count": t["review_count"],
        }
        for t in tabla
    ]

    resultados = {
        "subtarea": "T4",
        "descripcion": "Tabla resumen por servidor: server, observations, mean_power_w, "
                       "max_temperature_c, mean_load, review_count",
        "fuente_datos": fuente,
        "df_reconstruido_desde_csv": bool(reconstruido),
        "n_filas_df_entrada": int(len(df)),
        "servidores": servidores,
        "n_filas_tabla": n_filas,
        "columnas": list(resumen.columns),
        "orden_columnas_correcto": bool(orden_ok),
        "una_fila_por_servidor": una_fila_por_servidor,
        "tabla_resumen": tabla,
        "tabla_resumen_redondeada": tabla_redondeada,
        "todas_las_observaciones_son_50": bool((resumen["observations"] == 50).all()),
        "totales": {
            "observations": total_obs,
            "review_count": total_review,
            "requires_review_true_en_df": n_true_df,
            "review_count_consistente_con_df": bool(total_review == n_true_df),
            "review_count_coincide_con_T3": coincide_t3,
        },
        "verificacion_AI-SRV-01": verif,
        "valores_esperados_AI-SRV-01": ESPERADO_SRV01,
        "entrega_parquet": {
            "ruta": parquet_path.as_posix(),
            "existia_previamente": bool(not parquet_escrito_por_t4),
            "escrito_por_T4": bool(parquet_escrito_por_t4),
        },
        "criterio_exito_cumplido": criterio,
    }
    with open("resultados.json", "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, indent=2, ensure_ascii=False)

    # ---------- Resumen en consola ----------
    print("T4 — Tabla resumen por servidor")
    print(f"Fuente de datos: {fuente}")
    print(resumen.to_string(index=False))
    print(f"Filas: {n_filas} | Columnas: {list(resumen.columns)}")
    print(f"Totales: observations={total_obs}, review_count={total_review}")
    print(f"Verificación AI-SRV-01: {json.dumps(verif, ensure_ascii=False)}")
    print(f"Criterio de éxito cumplido: {criterio}")
    print("resultados.json escrito.")


if __name__ == "__main__":
    main()
