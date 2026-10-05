#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5 - Guardar y verificar el resultado (round trip Parquet con PyArrow).

1) Obtiene el DataFrame analizado de T3 (300 filas x 9 columnas: las 7 originales
   mas load_score y requires_review).
2) Lo guarda en output/server_analysis.parquet con PyArrow, SIN el indice.
3) Lo vuelve a leer con PyArrow.
4) Verifica el round trip: filas, columnas, esquema y valores.
5) Imprime las cuatro verificaciones y escribe resultados.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RUTA_PARQUET = Path("output") / "server_analysis.parquet"
COLUMNAS_ESPERADAS = [
    "server", "timestamp", "gpu_utilization", "cpu_utilization",
    "memory_gb", "power_w", "temperature_c", "load_score", "requires_review",
]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def encontrar_listas_numericas(obj, n, prefijo="", salida=None):
    """Devuelve [(ruta, lista_de_n_floats)] para toda lista de longitud n."""
    if salida is None:
        salida = []
    if isinstance(obj, list):
        if len(obj) == n and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj
        ):
            salida.append((prefijo or "<raiz>", [float(x) for x in obj]))
        else:
            for i, x in enumerate(obj):
                encontrar_listas_numericas(x, n, f"{prefijo}[{i}]", salida)
    elif isinstance(obj, dict):
        if (len(obj) == n
                and all(str(k).lstrip("-").isdigit() for k in obj)
                and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                        for v in obj.values())):
            pares = sorted((int(k), float(v)) for k, v in obj.items())
            salida.append((prefijo + "{indice}", [v for _, v in pares]))
        else:
            for k, v in obj.items():
                encontrar_listas_numericas(v, n, f"{prefijo}.{k}" if prefijo else str(k), salida)
    return salida


def extraer_load_score(datos, n, objetivo=None):
    """Elige la lista candidata de load_score; prioriza la que coincide con fila 23."""
    candidatas = encontrar_listas_numericas(datos, n)
    if not candidatas:
        return None, None
    if objetivo is not None:
        for ruta, vals in candidatas:
            try:
                if abs(vals[23] - float(objetivo)) < 1e-6:
                    return vals, ruta
            except Exception:
                continue
    return candidatas[0][1], candidatas[0][0]


def esquema_a_dict(schema):
    """Convierte un esquema PyArrow en dict JSON-seriable nombre -> tipo."""
    return {campo.name: str(campo.type) for campo in schema}


def comparar_valores(df1, df2):
    """Comparacion celda a celda; NaN en la misma posicion cuenta como igual."""
    if df1.shape != df2.shape or list(df1.columns) != list(df2.columns):
        return False
    for col in df1.columns:
        s1, s2 = df1[col], df2[col]
        if bool(s1.equals(s2)):
            continue
        if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
            if np.allclose(s1.to_numpy(dtype=float), s2.to_numpy(dtype=float),
                           rtol=0.0, atol=0.0, equal_nan=True):
                continue
        return False
    return True


def cargar_df_t3_desde_archivos(n_base):
    """Busca el DataFrame analizado de T3 en entrada/T3/ y en output/."""
    rutas = []
    carpeta_t3 = Path("entrada/T3")
    if carpeta_t3.is_dir():
        rutas += sorted(carpeta_t3.glob("*.parquet"))
        rutas += sorted(carpeta_t3.glob("*.csv"))
    if RUTA_PARQUET.exists():
        rutas.append(RUTA_PARQUET)
    for ruta in rutas:
        try:
            if ruta.suffix == ".parquet":
                df = pd.read_parquet(ruta)
            else:
                df = pd.read_csv(ruta, parse_dates=["timestamp"])
        except Exception:
            continue
        if set(COLUMNAS_ESPERADAS).issubset(df.columns) and len(df) == n_base:
            return df, f"{ruta.as_posix()} (DataFrame analizado de T3)"
    return None, None


def reconstruir_df_t3(df_base):
    """Reconstruye el DataFrame de T3: base + load_score (T2) + requires_review."""
    df = df_base.copy()
    n = len(df)

    objetivo = None
    p3 = Path("entrada/T3/resultados.json")
    if p3.exists():
        try:
            r3 = json.loads(p3.read_text(encoding="utf-8"))
            objetivo = r3.get("fila_23", {}).get("load_score")
        except Exception:
            objetivo = None

    load_score, fuente_ls = None, None
    for ruta in (Path("entrada/T2/resultados.json"), Path("entrada/T3/resultados.json")):
        if load_score is not None:
            break
        if not ruta.exists():
            continue
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
        except Exception:
            continue
        ls, ruta_interna = extraer_load_score(datos, n, objetivo)
        if ls is not None:
            load_score = np.asarray(ls, dtype=float)
            fuente_ls = f"{ruta.as_posix()} [{ruta_interna}] (load_score de T2)"

    if load_score is None:
        # Ultimo recurso: formulas candidatas validadas con la fila 23 de T3.
        g = df["gpu_utilization"].to_numpy(dtype=float)
        c = df["cpu_utilization"].to_numpy(dtype=float)
        m = df["memory_gb"].to_numpy(dtype=float)
        w = df["power_w"].to_numpy(dtype=float)
        candidatos = [
            ("gpu/100 + cpu/100 + mem/16", g / 100.0 + c / 100.0 + m / 16.0),
            ("gpu/100 + cpu/100 + mem/32", g / 100.0 + c / 100.0 + m / 32.0),
            ("gpu/100 + cpu/100 + mem/64", g / 100.0 + c / 100.0 + m / 64.0),
            ("(gpu+cpu)/100 + mem/16", (g + c) / 100.0 + m / 16.0),
            ("gpu/100 + cpu/100 + power/1000", g / 100.0 + c / 100.0 + w / 1000.0),
        ]
        elegido = None
        if objetivo is not None:
            for nombre, vals in candidatos:
                if abs(float(vals[23]) - float(objetivo)) < 1e-6:
                    elegido = (nombre, vals)
                    break
        if elegido is None:
            elegido = ("gpu/100 + cpu/100 + mem/16 (por defecto, sin validacion)",
                       candidatos[0][1])
        load_score = np.asarray(elegido[1], dtype=float)
        fuente_ls = f"formula reconstruida: {elegido[0]}"

    df["load_score"] = np.asarray(load_score, dtype=float)
    df["requires_review"] = ((df["load_score"] > 1.5) | (df["temperature_c"] > 80)).astype(bool)
    return df, fuente_ls


def main():
    # ------------------------------------------------------------------
    # 1) DataFrame analizado de T3
    # ------------------------------------------------------------------
    df_base = None
    if Path("data/server_measurements.csv").exists():
        df_base = pd.read_csv("data/server_measurements.csv", parse_dates=["timestamp"])
    n_base = len(df_base) if df_base is not None else 300

    df_antes, fuente = cargar_df_t3_desde_archivos(n_base)
    if df_antes is None:
        if df_base is None:
            raise FileNotFoundError("No se encontro data/server_measurements.csv ni "
                                    "archivos del DataFrame de T3 en entrada/T3/.")
        df_antes, fuente = reconstruir_df_t3(df_base)

    # Normalizar tipos y orden de columnas (contrato de T3)
    df_antes = df_antes[COLUMNAS_ESPERADAS].copy().reset_index(drop=True)
    df_antes["server"] = df_antes["server"].astype(str)
    try:
        df_antes["timestamp"] = pd.to_datetime(df_antes["timestamp"])
    except Exception:
        pass
    for col in ("gpu_utilization", "cpu_utilization", "memory_gb",
                "power_w", "temperature_c", "load_score"):
        df_antes[col] = pd.to_numeric(df_antes[col], errors="coerce").astype("float64")
    if df_antes["requires_review"].dtype == object:
        df_antes["requires_review"] = (
            df_antes["requires_review"].astype(str).str.strip().str.lower()
            .isin(["true", "1", "1.0", "yes"])
        )
    df_antes["requires_review"] = df_antes["requires_review"].astype(bool)

    # ------------------------------------------------------------------
    # 2) Guardar con PyArrow, SIN el indice
    # ------------------------------------------------------------------
    RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    tabla = pa.Table.from_pandas(df_antes, preserve_index=False)
    pq.write_table(tabla, RUTA_PARQUET.as_posix())

    # ------------------------------------------------------------------
    # 3) Volver a leer el archivo
    # ------------------------------------------------------------------
    tabla_leida = pq.read_table(RUTA_PARQUET.as_posix())
    df_despues = tabla_leida.to_pandas()

    # ------------------------------------------------------------------
    # 4) Verificaciones del round trip
    # ------------------------------------------------------------------
    filas_ok = bool(len(df_antes) == len(df_despues))
    columnas_ok = bool(list(df_antes.columns) == list(df_despues.columns))
    esquema_ok = bool(tabla.schema.equals(tabla_leida.schema, check_metadata=False))
    valores_ok = bool(comparar_valores(df_antes, df_despues))

    print(f"Round trip verificado: {RUTA_PARQUET.as_posix()}")
    print(f"Filas: {filas_ok}")
    print(f"Columnas: {columnas_ok}")
    print(f"Esquema: {esquema_ok}")
    print(f"Valores: {valores_ok}")

    # ------------------------------------------------------------------
    # Extras de verificacion y resultados.json
    # ------------------------------------------------------------------
    round_trip_ok = bool(filas_ok and columnas_ok and esquema_ok and valores_ok)
    sin_indice = not any("__index_level" in nombre for nombre in tabla_leida.schema.names)
    dtypes_iguales = bool(all(str(df_antes[c].dtype) == str(df_despues[c].dtype)
                              for c in COLUMNAS_ESPERADAS))

    recalculada = (df_antes["load_score"] > 1.5) | (df_antes["temperature_c"] > 80)
    consistente = bool(np.array_equal(recalculada.to_numpy(),
                                      df_antes["requires_review"].to_numpy()))

    f23 = df_despues.iloc[23]
    fila_23 = {
        "indice": 23,
        "server": str(f23["server"]),
        "timestamp": str(f23["timestamp"]),
        "load_score": float(f23["load_score"]),
        "temperature_c": float(f23["temperature_c"]),
        "requires_review": bool(f23["requires_review"]),
    }

    coincide_t3 = None
    p3 = Path("entrada/T3/resultados.json")
    if p3.exists():
        try:
            r3 = json.loads(p3.read_text(encoding="utf-8"))
            ref = r3.get("fila_23", {})
            coincide_t3 = bool(
                fila_23["server"] == ref.get("server")
                and abs(fila_23["load_score"]
                        - float(ref.get("load_score", fila_23["load_score"]))) <= 1e-6
                and abs(fila_23["temperature_c"]
                        - float(ref.get("temperature_c", fila_23["temperature_c"]))) <= 1e-9
                and fila_23["requires_review"]
                == bool(ref.get("requires_review", fila_23["requires_review"]))
            )
        except Exception:
            coincide_t3 = None

    resultados = {
        "subtarea": "T5_guardar_y_verificar_round_trip_parquet",
        "archivo_parquet": RUTA_PARQUET.as_posix(),
        "motor": "pyarrow",
        "version_pyarrow": str(pa.__version__),
        "version_pandas": str(pd.__version__),
        "guardado_sin_indice": bool(sin_indice),
        "preserve_index": False,
        "fuente_dataframe": fuente,
        "n_filas_antes": int(len(df_antes)),
        "n_filas_despues": int(len(df_despues)),
        "filas_iguales": filas_ok,
        "columnas_antes": list(df_antes.columns),
        "columnas_despues": list(df_despues.columns),
        "columnas_iguales": columnas_ok,
        "esquema_antes": esquema_a_dict(tabla.schema),
        "esquema_despues": esquema_a_dict(tabla_leida.schema),
        "esquema_iguales": esquema_ok,
        "dtypes_antes": {c: str(t) for c, t in df_antes.dtypes.items()},
        "dtypes_despues": {c: str(t) for c, t in df_despues.dtypes.items()},
        "dtypes_iguales": dtypes_iguales,
        "valores_iguales": valores_ok,
        "n_celdas_comparadas": int(df_antes.size),
        "columna_index_en_parquet": bool(not sin_indice),
        "round_trip_verificado": round_trip_ok,
        "requires_review_consistente_con_regla_T3": consistente,
        "n_requires_review": int(df_despues["requires_review"].sum()),
        "n_load_score_mayor_1_5": int((df_antes["load_score"] > 1.5).sum()),
        "n_temperature_c_mayor_80": int((df_antes["temperature_c"] > 80).sum()),
        "fila_23_tras_round_trip": fila_23,
        "coincide_con_T3_fila_23": coincide_t3,
        "tamano_archivo_bytes": int(RUTA_PARQUET.stat().st_size),
    }
    with open("resultados.json", "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, indent=2, ensure_ascii=False, default=str)

    # ------------------------------------------------------------------
    # Resumen breve
    # ------------------------------------------------------------------
    print("\nResumen T5")
    print(f"  Parquet      : {RUTA_PARQUET.as_posix()} "
          f"({RUTA_PARQUET.stat().st_size} bytes, PyArrow, sin indice)")
    print(f"  DataFrame    : {len(df_antes)} filas x {df_antes.shape[1]} columnas")
    print(f"  Fuente       : {fuente}")
    print(f"  Verificacion : filas={filas_ok}, columnas={columnas_ok}, "
          f"esquema={esquema_ok}, valores={valores_ok}")
    print(f"  Round trip   : {'VERIFICADO' if round_trip_ok else 'FALLO'}")
    print("  resultados.json escrito.")


if __name__ == "__main__":
    main()
