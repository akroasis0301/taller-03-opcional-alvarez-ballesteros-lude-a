#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subtarea T2 — Auditar los datos y construir los pares (data/model_errors.csv).

Pasos:
1) Cargar data/model_errors.csv (SOLO lectura; el CSV original nunca se modifica).
2) Validar:
   - que existan EXACTAMENTE las columnas store_id, error_a, error_b;
   - que cada store_id sea único y no nulo (ValueError que menciona el identificador);
   - que los errores sean numéricos, finitos y no negativos (ValueError si no).
3) Construir la tabla pareada conservando las tres columnas originales y añadiendo
   difference = error_a - error_b y favors ('B' si difference > 0, 'A' si < 0),
   verificando el emparejamiento mediante store_id (no solo longitudes iguales).
4) Contar tiendas que favorecen a A y a B, detectar empates e identificar las
   diferencias de mayor magnitud en cada dirección con su store_id.
5) Escribir resultados.json, output/store_differences.csv y
   output/store_differences.png (copia local: store_differences.png).
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CSV_PATH = Path("data/model_errors.csv")
OUT_DIR = Path("output")
CSV_OUT = OUT_DIR / "store_differences.csv"
PNG_OUT = OUT_DIR / "store_differences.png"
PNG_LOCAL = Path("store_differences.png")
RESULTS_PATH = Path("resultados.json")

EXPECTED_COLS = ["store_id", "error_a", "error_b"]


# --------------------------------------------------------------------------- #
# Validaciones del enunciado
# --------------------------------------------------------------------------- #
def validar_csv(df: pd.DataFrame) -> dict:
    """Valida columnas exactas, store_id único/no nulo y errores numéricos,
    finitos y no negativos. Lanza ValueError mencionando lo problemático."""
    encontradas = [str(c) for c in df.columns]
    if len(encontradas) != len(EXPECTED_COLS) or set(encontradas) != set(EXPECTED_COLS):
        faltan = sorted(set(EXPECTED_COLS) - set(encontradas))
        sobran = sorted(set(encontradas) - set(EXPECTED_COLS))
        raise ValueError(
            f"El CSV debe tener EXACTAMENTE las columnas {EXPECTED_COLS}; "
            f"encontradas={encontradas}; faltan={faltan}; sobran={sobran}"
        )

    rep = {
        "columnas_exactas": True,
        "columnas_encontradas": encontradas,
        "n_filas_csv": int(len(df)),
        "store_id_nulos": [],
        "store_id_duplicados": [],
        "error_a_no_numerico": [],
        "error_b_no_numerico": [],
        "error_a_no_finito": [],
        "error_b_no_finito": [],
        "error_a_negativos": [],
        "error_b_negativos": [],
    }

    # -- store_id: no nulo y único ------------------------------------------
    sid = df["store_id"]
    filas_nulas = df.index[sid.isna()].tolist()
    if filas_nulas:
        raise ValueError(
            f"store_id contiene valores nulos en las filas {filas_nulas}: "
            "cada tienda debe tener un identificador no nulo"
        )
    dupes = sorted(sid[sid.duplicated(keep=False)].unique().tolist())
    if dupes:
        raise ValueError(
            f"store_id duplicado: {dupes}: cada identificador de tienda debe ser único"
        )

    # -- errores: numéricos, finitos y no negativos --------------------------
    for col in ("error_a", "error_b"):
        raw = df[col]
        if pd.api.types.is_numeric_dtype(raw):
            vals = pd.to_numeric(raw, errors="coerce").astype(float)
        else:
            vals = pd.to_numeric(raw, errors="coerce")
            malas = df.index[raw.notna() & vals.isna()].tolist()
            if malas:
                ejemplos = {str(sid.iloc[i]): repr(raw.iloc[i]) for i in malas[:10]}
                raise ValueError(
                    f"'{col}' contiene valores NO numéricos (filas {malas}, "
                    f"ejemplos por store_id: {ejemplos})"
                )
            vals = vals.astype(float)
        arr = vals.to_numpy(dtype=float)
        no_fin = df.index[~np.isfinite(arr)].tolist()
        if no_fin:
            ejemplos = {str(sid.iloc[i]): repr(raw.iloc[i]) for i in no_fin[:10]}
            raise ValueError(
                f"'{col}' contiene valores NO finitos (NaN/inf) "
                f"(filas {no_fin}, ejemplos por store_id: {ejemplos})"
            )
        negs = df.index[arr < 0].tolist()
        if negs:
            detalle = {str(sid.iloc[i]): float(arr[i]) for i in negs[:10]}
            raise ValueError(
                f"'{col}' contiene valores NEGATIVOS, no permitidos: {detalle}"
            )
    return rep


# --------------------------------------------------------------------------- #
# Tabla pareada + verificación del cruce por store_id
# --------------------------------------------------------------------------- #
def construir_tabla_pareada(df: pd.DataFrame):
    """Construye la tabla pareada y verifica el emparejamiento por store_id."""
    df = df.loc[:, EXPECTED_COLS]  # orden canónico, solo en memoria
    sid = df["store_id"].astype(str)

    idx = pd.Index(sid.tolist(), name="store_id")
    a = pd.Series(df["error_a"].to_numpy(dtype=float), index=idx)
    b = pd.Series(df["error_b"].to_numpy(dtype=float), index=idx)

    # Cruce por store_id: ambas columnas deben estar indexadas por las MISMAS
    # tiendas (mismo conjunto, sin duplicados); la diferencia se calcula sobre
    # claves alineadas por store_id, NO por posición.
    if not a.index.equals(b.index):
        raise ValueError("Los pares no corresponden a las mismas tiendas (store_id)")
    if a.index.has_duplicates:
        raise ValueError("store_id duplicado al alinear los pares")
    diff = a - b  # alineado por store_id

    tabla = pd.DataFrame(
        {
            "store_id": a.index.tolist(),
            "error_a": a.to_numpy(),
            "error_b": b.to_numpy(),
            "difference": diff.to_numpy(),
        }
    )
    tabla["favors"] = np.select(
        [tabla["difference"].to_numpy() > 0, tabla["difference"].to_numpy() < 0],
        ["B", "A"],
        default=None,
    )

    # Integridad: cada fila de la tabla debe corresponder al mismo par del CSV
    mapa = {s: (float(x), float(y)) for s, x, y in zip(sid, df["error_a"], df["error_b"])}
    for _, fila in tabla.iterrows():
        a0, b0 = mapa[str(fila["store_id"])]
        if not (
            math.isclose(a0, float(fila["error_a"]), rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(b0, float(fila["error_b"]), rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(f"El par de {fila['store_id']} no coincide con el CSV original")
    if not np.allclose(
        (tabla["error_a"] - tabla["error_b"]).to_numpy(),
        tabla["difference"].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Inconsistencia al recalcular difference = error_a - error_b")

    verif = {
        "validado_por_store_id": True,
        "metodo": (
            "error_a y error_b se indexaron por store_id; se comprobó que ambos índices "
            "contienen exactamente las mismas tiendas (sin duplicados ni nulos) y "
            "difference se calculó sobre claves alineadas por store_id, no por posición"
        ),
        "mismas_tiendas_en_ambas_columnas": True,
        "n_pares": int(len(tabla)),
        "store_ids": a.index.tolist(),
        "difference_recalculada_ok": True,
    }
    return tabla, verif


# --------------------------------------------------------------------------- #
# Cifras de la Parte 2 (pregunta 12)
# --------------------------------------------------------------------------- #
def extremo_direccion(tabla: pd.DataFrame, direccion: str):
    """Mayor magnitud en cada dirección: 'positiva' (favorece a B) o
    'negativa' (favorece a A). Devuelve None si no hay diferencias de ese signo."""
    diff = tabla["difference"]
    sub = tabla[diff > 0] if direccion == "positiva" else tabla[diff < 0]
    if sub.empty:
        return None
    pos = sub["difference"].idxmax() if direccion == "positiva" else sub["difference"].idxmin()
    fila = sub.loc[pos]
    fav = fila["favors"]
    return {
        "store_id": str(fila["store_id"]),
        "difference": float(fila["difference"]),
        "magnitud": float(abs(fila["difference"])),
        "error_a": float(fila["error_a"]),
        "error_b": float(fila["error_b"]),
        "favors": None if fav is None else str(fav),
        "direccion": direccion,
    }


def demo_longitud_insuficiente(tabla: pd.DataFrame, semilla: int = 0) -> dict:
    """Evidencia de que comprobar solo longitudes no valida el emparejamiento:
    permutar error_b mantiene 16==16 pero cambia los pares y las conclusiones."""
    rng = np.random.default_rng(semilla)
    b_perm = rng.permutation(tabla["error_b"].to_numpy(dtype=float))
    diff_perm = tabla["error_a"].to_numpy(dtype=float) - b_perm
    return {
        "descripcion": (
            "Comprobar solo que len(error_a) == len(error_b) no valida el emparejamiento: "
            "al permutar error_b (semilla 0) las longitudes siguen siendo iguales pero cada "
            "tienda quedaría comparada con el error_b de OTRA tienda. La garantía la da el "
            "cruce por store_id."
        ),
        "longitudes_iguales_tras_permutar": True,
        "posiciones_con_error_b_distinto_tras_permutar": int(
            (b_perm != tabla["error_b"].to_numpy(dtype=float)).sum()
        ),
        "cuentas_con_emparejamiento_por_posicion": {
            "favorecen_A": int((diff_perm < 0).sum()),
            "favorecen_B": int((diff_perm > 0).sum()),
            "empates": int((diff_perm == 0).sum()),
        },
        "cuentas_correctas_cruzando_por_store_id": {
            "favorecen_A": int((tabla["favors"] == "A").sum()),
            "favorecen_B": int((tabla["favors"] == "B").sum()),
            "empates": int((tabla["difference"] == 0).sum()),
        },
    }


# --------------------------------------------------------------------------- #
# Figura
# --------------------------------------------------------------------------- #
def crear_figura(tabla, n_A, n_B, n_emp, max_pos, max_neg, ruta_out, ruta_local):
    vals = tabla["difference"].to_numpy(dtype=float)
    ids = tabla["store_id"].tolist()

    def color(v):
        if v > 0:
            return "#2a9d8f"  # favorece a B
        if v < 0:
            return "#e76f51"  # favorece a A
        return "#9e9e9e"      # empate

    colores = [color(v) for v in vals]
    y = np.arange(len(tabla))

    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.barh(y, vals, color=colores, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(ids)
    ax.invert_yaxis()
    ax.axvline(0.0, color="black", linewidth=1)
    margen = max(0.5, float(np.max(np.abs(vals))) * 0.18) if len(vals) else 0.5
    ax.set_xlim(float(np.min(vals)) - margen - 0.3, float(np.max(vals)) + margen + 0.3)
    ax.set_xlabel("difference = error_a − error_b", fontsize=10)
    ax.set_title(
        "Tarea 2 · Diferencia de error por tienda (error_a − error_b)\n"
        f"Favorece a A: {n_A} tiendas · Favorece a B: {n_B} tiendas · Empates: {n_emp}"
    )
    for i, v in enumerate(vals):
        ax.text(
            v + (0.04 if v >= 0 else -0.04), i, f"{v:+.2f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=7.5, color="#333333",
        )
    if max_pos is not None:
        i = ids.index(max_pos["store_id"])
        ax.annotate(
            f"máx +: {max_pos['store_id']} ({max_pos['difference']:+.2f})",
            xy=(vals[i], i), xytext=(0, 14), textcoords="offset points",
            ha="center", fontsize=8.5, fontweight="bold", color="#1d6f66",
        )
    if max_neg is not None:
        i = ids.index(max_neg["store_id"])
        ax.annotate(
            f"máx −: {max_neg['store_id']} ({max_neg['difference']:+.2f})",
            xy=(vals[i], i), xytext=(0, -16), textcoords="offset points",
            ha="center", fontsize=8.5, fontweight="bold", color="#a2402c",
        )
    ax.legend(
        handles=[
            Patch(color="#e76f51", label="Favorece a A (difference < 0)"),
            Patch(color="#2a9d8f", label="Favorece a B (difference > 0)"),
        ],
        loc="lower right", fontsize=8, framealpha=0.9,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    ruta_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_out, dpi=120, bbox_inches="tight")
    fig.savefig(ruta_local, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def jsonable(obj):
    """Convierte tipos de numpy/pandas a tipos nativos para JSON."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# --------------------------------------------------------------------------- #
# Principal
# --------------------------------------------------------------------------- #
def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {CSV_PATH}")

    sha_inicio = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()

    # 1) Carga (solo lectura; el CSV original no se modifica en ningún punto)
    df = pd.read_csv(CSV_PATH)

    # 2) Validaciones del enunciado
    rep_validacion = validar_csv(df)

    # 3) Tabla pareada + verificación del cruce por store_id
    tabla, verif = construir_tabla_pareada(df)

    # 4) Cifras de la Parte 2 (preguntas 11 y 12)
    ids_A = tabla.loc[tabla["favors"] == "A", "store_id"].tolist()
    ids_B = tabla.loc[tabla["favors"] == "B", "store_id"].tolist()
    mask_empate = tabla["difference"] == 0
    ids_empate = tabla.loc[mask_empate, "store_id"].tolist()
    n_A, n_B, n_emp = len(ids_A), len(ids_B), int(mask_empate.sum())

    max_pos = extremo_direccion(tabla, "positiva")   # mayor magnitud que favorece a B
    max_neg = extremo_direccion(tabla, "negativa")   # mayor magnitud que favorece a A

    demo = demo_longitud_insuficiente(tabla, semilla=0)

    # 5) Archivos exigidos por el enunciado
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(CSV_OUT, index=False)
    crear_figura(tabla, n_A, n_B, n_emp, max_pos, max_neg, PNG_OUT, PNG_LOCAL)

    # Evidencia de que el CSV original no fue modificado
    sha_fin = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    if sha_fin != sha_inicio:
        raise RuntimeError("El CSV original fue modificado durante la ejecución")

    explicacion_p11 = (
        "Los 16 pares fueron validados cruzando por store_id: error_a y error_b se "
        "indexaron por store_id, se comprobó que ambos índices contienen exactamente las "
        "mismas tiendas sin duplicados ni nulos, y difference se calculó sobre claves "
        "alineadas por store_id. Comprobar únicamente que ambas columnas tienen la misma "
        "longitud (16 == 16) es insuficiente porque solo verifica la cantidad de valores: "
        "no detecta que un valor de error_b pueda pertenecer a otra tienda (orden distinto, "
        "desplazamiento, fila omitida o duplicada). En 'demo_longitud_insuficiente' se "
        "muestra que permutar error_b mantiene 16==16 pero cambia el emparejamiento y las "
        "conclusiones por tienda."
    )

    resultados = {
        "subtarea": "T2_auditar_datos_y_construir_pares",
        "csv_entrada": str(CSV_PATH),
        "csv_sha256": sha_inicio,
        "csv_original_no_modificado": True,
        "validacion": rep_validacion,
        "emparejamiento": verif,
        "p11": {
            "pares_validados_por_store_id": True,
            "n_pares_validados": int(len(tabla)),
            "explicacion_por_que_longitud_no_basta": explicacion_p11,
        },
        "p12": {
            "tiendas_que_favorecen_A": n_A,
            "tiendas_que_favorecen_B": n_B,
            "store_ids_que_favorecen_A": ids_A,
            "store_ids_que_favorecen_B": ids_B,
            "empates": n_emp,
            "hay_empates": bool(n_emp > 0),
            "store_ids_empate": ids_empate,
            "diferencia_maxima_positiva_favorece_B": max_pos,
            "diferencia_maxima_negativa_favorece_A": max_neg,
        },
        # claves planas de conveniencia
        "n_favors_A": n_A,
        "n_favors_B": n_B,
        "n_ties": n_emp,
        "hay_empates": bool(n_emp > 0),
        "max_positive_difference": max_pos,
        "max_negative_difference": max_neg,
        "columnas_tabla_pareada": ["store_id", "error_a", "error_b", "difference", "favors"],
        "tabla_pareada": jsonable(tabla.to_dict(orient="records")),
        "estadisticas": {
            "n_filas": int(len(tabla)),
            "media_difference": float(tabla["difference"].mean()),
            "suma_difference": float(tabla["difference"].sum()),
            "min_difference": float(tabla["difference"].min()),
            "max_difference": float(tabla["difference"].max()),
        },
        "demo_longitud_insuficiente": demo,
        "archivos_generados": [str(CSV_OUT), str(PNG_OUT), str(PNG_LOCAL), str(RESULTS_PATH)],
    }

    RESULTS_PATH.write_text(
        json.dumps(jsonable(resultados), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Resumen breve
    print("=" * 64)
    print("T2 · Auditoría de datos y construcción de pares")
    print("=" * 64)
    print(f"CSV leído (solo lectura): {CSV_PATH} | sha256: {sha_inicio[:12]}...")
    print(f"Validación OK: columnas exactas {EXPECTED_COLS}; store_id único y no nulo;")
    print("               errores numéricos, finitos y no negativos.")
    print(f"Pares verificados por store_id: {verif['n_pares']}")
    print(f"Favorecen a A ({n_A}): {ids_A}")
    print(f"Favorecen a B ({n_B}): {ids_B}")
    print(f"Empates ({n_emp}): {ids_empate}")
    if max_pos is not None:
        print(f"Mayor diferencia que favorece a B: {max_pos['store_id']} ({max_pos['difference']:+.4f})")
    if max_neg is not None:
        print(f"Mayor diferencia que favorece a A: {max_neg['store_id']} ({max_neg['difference']:+.4f})")
    print(f"Archivos escritos: {CSV_OUT}, {PNG_OUT}, {PNG_LOCAL}, {RESULTS_PATH}")


if __name__ == "__main__":
    main()
