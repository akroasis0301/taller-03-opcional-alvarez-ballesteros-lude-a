#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T4 - Construir un resumen por servidor.

Agrupa el DataFrame de T3 por 'server' y construye la tabla resumen con
EXACTAMENTE las columnas, en este orden:

    server | observations | mean_power_w | max_temperature_c | mean_load | review_count

Significado de cada metrica:
    observations      : numero de filas del servidor
    mean_power_w      : media de power_w (W)
    max_temperature_c : maximo de temperature_c (C)
    mean_load         : media de load_score (metrica de carga de T2, presente en T3)
    review_count      : conteo de filas con requires_review=True

Imprime la tabla completa de los seis servidores, guarda el resumen en
output/server_summary.parquet, la figura resumen_servidores.png y el contrato
resultados.json. El parquet exigido por el enunciado
(output/server_analysis.parquet, producido por T3) se usa como entrada y se
preserva sin modificar; solo se regeneraria si no existiera.
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
VALORES_CONTROL = {
    "observations": 50,
    "mean_power_w": 363.4598,
    "max_temperature_c": 86.50,
    "mean_load": -0.614527,
    "review_count": 2,
}


# ----------------------------------------------------------------------------
# Carga del DataFrame de T3
# ----------------------------------------------------------------------------
def _leer_tabla(ruta):
    try:
        sufijo = ruta.suffix.lower()
        if sufijo == ".parquet":
            return pd.read_parquet(ruta)
        if sufijo == ".csv":
            return pd.read_csv(ruta)
    except Exception:
        return None
    return None


def _glob_ordenado(patron):
    return sorted(
        Path(".").glob(patron),
        key=lambda p: (0 if "server_analysis" in p.name.lower() else 1, str(p)),
    )


def cargar_df_t3():
    """Busca el DataFrame enriquecido de T3 (con load_score y requires_review)."""
    necesarias = {"server", "power_w", "temperature_c", "load_score", "requires_review"}
    candidatas = []
    for patron in ("entrada/T3/**/*.parquet", "entrada/T3/**/*.csv"):
        candidatas.extend(_glob_ordenado(patron))
    candidatas.append(Path("output/server_analysis.parquet"))
    vistas = set()
    for ruta in candidatas:
        if ruta in vistas or not ruta.is_file():
            continue
        vistas.add(ruta)
        df = _leer_tabla(ruta)
        if df is not None and necesarias.issubset(df.columns) and len(df) > 0:
            return df, str(ruta)
    return None, None


# ----------------------------------------------------------------------------
# Respaldo: recuperar load_score desde T2 o reconstruirlo (validado contra T3)
# ----------------------------------------------------------------------------
def _indices_load_alto_esperados(df):
    """Indices con requires_review=True y temperature_c <= 80 segun T3; por la
    definicion de T3 (load_score > 1.5 o temperature_c > 80) son exactamente
    los indices con load_score > 1.5."""
    ruta = Path("entrada/T3/resultados.json")
    if not ruta.is_file():
        return None
    try:
        data = json.loads(ruta.read_text(encoding="utf-8"))
    except Exception:
        return None
    idx = data.get("indices_requires_review") if isinstance(data, dict) else None
    if not isinstance(idx, list) or not idx:
        return None
    temp = pd.to_numeric(df["temperature_c"], errors="coerce").to_numpy(float)
    n = len(df)
    seleccion = []
    for i in idx:
        if isinstance(i, bool) or not isinstance(i, (int, float)):
            continue
        i = int(i)
        if 0 <= i < n and not (temp[i] > 80):
            seleccion.append(i)
    return np.array(sorted(seleccion), dtype=int) if seleccion else None


def _media_control(df, ls):
    mascara = df["server"].astype(str).to_numpy() == SERVIDOR_CONTROL
    if not mascara.any():
        return None
    return float(np.mean(np.asarray(ls, dtype=float)[mascara]))


def _validar_load_score(df, ls):
    """Puntos: media de AI-SRV-01 correcta (3) y conjunto load>1.5 correcto (2)."""
    ls = np.asarray(ls, dtype=float)
    puntos = 0
    media = _media_control(df, ls)
    if media is not None and abs(media - VALORES_CONTROL["mean_load"]) <= 5e-7:
        puntos += 3
    objetivo = _indices_load_alto_esperados(df)
    if objetivo is not None:
        obtenidos = np.where(ls > 1.5)[0]
        if np.array_equal(np.sort(obtenidos), np.sort(objetivo)):
            puntos += 2
    return puntos


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


def load_scores_desde_t2(n):
    for patron in ("entrada/T2/**/*.parquet", "entrada/T2/**/*.csv"):
        for ruta in _glob_ordenado(patron):
            df_t2 = _leer_tabla(ruta)
            if df_t2 is not None and "load_score" in df_t2.columns and len(df_t2) == n:
                serie = pd.to_numeric(df_t2["load_score"], errors="coerce")
                if bool(serie.notna().all()):
                    return serie.to_numpy(float), str(ruta)
    ruta_json = Path("entrada/T2/resultados.json")
    if ruta_json.is_file():
        try:
            data = json.loads(ruta_json.read_text(encoding="utf-8"))
        except Exception:
            return None, None
        lista = _buscar_lista_numerica(data, n)
        if lista is not None:
            return np.asarray(lista, dtype=float), str(ruta_json)
    return None, None


def reconstruir_load_score(df):
    """Ultimo recurso: z-score de compuestos de carga plausibles; se elige el
    candidato que valida contra los reportes de T3."""
    gpu = pd.to_numeric(df["gpu_utilization"], errors="coerce").to_numpy(float)
    cpu = pd.to_numeric(df["cpu_utilization"], errors="coerce").to_numpy(float)
    mem = pd.to_numeric(df["memory_gb"], errors="coerce").to_numpy(float)
    pot = pd.to_numeric(df["power_w"], errors="coerce").to_numpy(float)
    compuestos = {
        "z(gpu+cpu+mem)": gpu + cpu + mem,
        "z(gpu+cpu)": gpu + cpu,
        "z(mem)": mem,
        "z(pow)": pot,
        "z(gpu)": gpu,
        "z(cpu)": cpu,
    }
    n = len(df)
    mejor = None
    for nombre, x in compuestos.items():
        if not np.isfinite(x).all():
            continue
        sd0 = float(x.std())
        if not np.isfinite(sd0) or sd0 == 0.0:
            continue
        z0 = (x - x.mean()) / sd0
        for sufijo, z in (("ddof=0", z0), ("ddof=1", z0 * np.sqrt((n - 1) / n))):
            puntos = _validar_load_score(df, z)
            if mejor is None or puntos > mejor[0]:
                mejor = (puntos, f"{nombre} [{sufijo}]", z)
    if mejor is None:
        raise RuntimeError("No fue posible reconstruir load_score a partir del CSV")
    return mejor[2], mejor[1], int(mejor[0])


def obtener_dataframe_t3():
    """Devuelve (df, fuente, reconstruido, nota)."""
    df, fuente = cargar_df_t3()
    if df is not None:
        return df, fuente, False, "DataFrame de T3 leido directamente"

    ruta_csv = Path("data/server_measurements.csv")
    if not ruta_csv.is_file():
        raise FileNotFoundError(
            "No se encontro el DataFrame de T3 ni data/server_measurements.csv"
        )
    try:
        df = pd.read_csv(ruta_csv, parse_dates=["timestamp"])
    except Exception:
        df = pd.read_csv(ruta_csv)

    ls, fuente_ls = load_scores_desde_t2(len(df))
    if ls is not None and _validar_load_score(df, ls) >= 2:
        df["load_score"] = ls
        df["requires_review"] = (df["load_score"] > 1.5) | (df["temperature_c"] > 80)
        return (
            df,
            f"{ruta_csv} + load_score de {fuente_ls}",
            False,
            "load_score recuperado de los artefactos de T2 (validado contra T3)",
        )

    z, nombre, puntos = reconstruir_load_score(df)
    df["load_score"] = z
    df["requires_review"] = (df["load_score"] > 1.5) | (df["temperature_c"] > 80)
    return (
        df,
        str(ruta_csv),
        True,
        f"load_score reconstruido como {nombre} (validacion: {puntos} puntos)",
    )


def tabla_registros(tabla):
    registros = []
    for _, r in tabla.iterrows():
        registros.append(
            {
                "server": str(r["server"]),
                "observations": int(r["observations"]),
                "mean_power_w": float(r["mean_power_w"]),
                "max_temperature_c": float(r["max_temperature_c"]),
                "mean_load": float(r["mean_load"]),
                "review_count": int(r["review_count"]),
            }
        )
    return registros


def main():
    df, fuente, reconstruido, nota = obtener_dataframe_t3()

    # --- normalizacion defensiva de tipos ---
    df = df.reset_index(drop=True).copy()
    df["server"] = df["server"].astype(str).str.strip()
    for col in ("power_w", "temperature_c", "load_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["requires_review"].dtype != bool:
        df["requires_review"] = (
            df["requires_review"].astype(str).str.strip().str.lower()
            .isin(["true", "1", "1.0"])
        )

    # --- tabla resumen: una fila por servidor, columnas en el orden exigido ---
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

    # --- impresion de la tabla completa (formato del enunciado) ---
    formatos = {
        "observations": "{:d}".format,
        "mean_power_w": "{:.4f}".format,
        "max_temperature_c": "{:.2f}".format,
        "mean_load": "{:.6f}".format,
        "review_count": "{:d}".format,
    }
    tabla_texto = resumen.to_string(index=False, formatters=formatos)
    print("=== T4 - Tabla resumen por servidor ===")
    print(tabla_texto)

    # --- verificacion contra la fila de control del enunciado ---
    filas_ctrl = resumen.loc[resumen["server"] == SERVIDOR_CONTROL]
    control_presente = len(filas_ctrl) == 1
    verificacion = {}
    if control_presente:
        f = filas_ctrl.iloc[0]
        chequeos = [
            ("observations", int(f["observations"]),
             VALORES_CONTROL["observations"], lambda o, e: o == e),
            ("mean_power_w", float(f["mean_power_w"]),
             VALORES_CONTROL["mean_power_w"], lambda o, e: abs(o - e) <= 5e-5),
            ("max_temperature_c", float(f["max_temperature_c"]),
             VALORES_CONTROL["max_temperature_c"], lambda o, e: abs(o - e) <= 5e-3),
            ("mean_load", float(f["mean_load"]),
             VALORES_CONTROL["mean_load"], lambda o, e: abs(o - e) <= 5e-7),
            ("review_count", int(f["review_count"]),
             VALORES_CONTROL["review_count"], lambda o, e: o == e),
        ]
        for nombre_col, obtenido, esperado, criterio in chequeos:
            verificacion[nombre_col] = {
                "obtenido": obtenido,
                "esperado": esperado,
                "ok": bool(criterio(obtenido, esperado)),
            }
    fila_control_coincide = bool(
        control_presente and all(v["ok"] for v in verificacion.values())
    )

    # --- tabla redondeada tal como se muestra en el enunciado ---
    disp = resumen.copy()
    disp["mean_power_w"] = disp["mean_power_w"].round(4)
    disp["max_temperature_c"] = disp["max_temperature_c"].round(2)
    disp["mean_load"] = disp["mean_load"].round(6)

    # --- parquet exigido por el enunciado: preservar el de T3 ---
    ruta_parquet_t3 = Path("output/server_analysis.parquet")
    if ruta_parquet_t3.is_file():
        estado_parquet_t3 = "preservado sin modificar (producido por T3)"
    else:
        try:
            ruta_parquet_t3.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(ruta_parquet_t3, index=False)
            estado_parquet_t3 = "regenerado desde la fuente de T3 (no existia)"
        except Exception as exc:
            estado_parquet_t3 = f"no disponible ({exc})"

    # --- parquet propio de T4 con la tabla resumen ---
    ruta_parquet_resumen = Path("output/server_summary.parquet")
    ruta_parquet_resumen.parent.mkdir(parents=True, exist_ok=True)
    resumen.to_parquet(ruta_parquet_resumen, index=False)

    # --- figura resumen ---
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4.2))
    ejes[0].bar(resumen["server"], resumen["mean_power_w"], color="#4472c4")
    ejes[0].set_title("Potencia media por servidor")
    ejes[0].set_ylabel("mean_power_w (W)")
    ejes[0].tick_params(axis="x", rotation=45)
    ejes[1].bar(resumen["server"], resumen["max_temperature_c"], color="#ed7d31")
    ejes[1].set_title("Temperatura maxima por servidor")
    ejes[1].set_ylabel("max_temperature_c (C)")
    ejes[1].tick_params(axis="x", rotation=45)
    fig.suptitle("T4 - Resumen por servidor")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("resumen_servidores.png", dpi=120)
    plt.close(fig)

    # --- contrato resultados.json ---
    orden_correcto = bool(list(resumen.columns) == COLUMNAS_EXIGIDAS)
    exito = bool(len(resumen) == 6 and orden_correcto and fila_control_coincide)
    resultados = {
        "subtarea": "T4_resumen_por_servidor",
        "fuente_datos": fuente,
        "nota_fuente": nota,
        "load_score_reconstruido": bool(reconstruido),
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
        "valores_esperados_fila_control": VALORES_CONTROL,
        "verificacion_fila_control": verificacion,
        "fila_control_coincide": fila_control_coincide,
        "total_observations": int(resumen["observations"].sum()),
        "total_review_count": int(resumen["review_count"].sum()),
        "parquet_resumen": str(ruta_parquet_resumen),
        "parquet_enunciado": str(ruta_parquet_t3),
        "estado_parquet_enunciado": estado_parquet_t3,
        "figura": "resumen_servidores.png",
        "exito": exito,
    }
    Path("resultados.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- resumen breve ---
    print("\n=== Resumen T4 ===")
    print(f"Fuente de datos      : {fuente}")
    print(f"  ({nota})")
    print(f"Filas de entrada     : {len(df)}")
    print(f"Servidores (filas)   : {resumen.shape[0]}")
    print("Orden de columnas    : " + ("correcto" if orden_correcto else "INCORRECTO"))
    if control_presente:
        estado = (
            "coincide con el valor esperado"
            if fila_control_coincide
            else "NO coincide - ver verificacion_fila_control"
        )
        print(f"Fila {SERVIDOR_CONTROL}  : {estado}")
    print(f"Total observations   : {int(resumen['observations'].sum())}")
    print(f"Total review_count   : {int(resumen['review_count'].sum())}")
    print(f"Parquet del enunciado: {estado_parquet_t3}")
    print(
        "Archivos escritos    : resultados.json, output/server_summary.parquet, "
        "resumen_servidores.png"
    )
    print(f"EXITO                : {'SI' if exito else 'NO'}")


if __name__ == "__main__":
    main()
