#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2 — Auditar data/model_errors.csv y construir la tabla pareada.

1) Cargar el CSV (solo lectura; el archivo original NUNCA se modifica).
2) Auditar:
   - columnas exactas: store_id, error_a, error_b (ValueError si no).
   - store_id único y no nulo (ValueError que menciona el identificador).
   - errores numéricos, finitos y no negativos (ValueError).
3) Verificar el emparejamiento POR store_id (no solo longitudes iguales).
4) Construir tabla pareada: 3 columnas originales + difference (= error_a - error_b)
   + favors ('B' si difference>0, 'A' si difference<0, 'empate' si difference==0).
5) Contar diferencias positivas / negativas / nulas (empates) y extraer las de
   mayor magnitud en cada dirección.
6) Escribir output/store_differences.csv, output/store_differences.png y
   resultados.json (vector de diferencias, conteos por modelo y extremos).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CSV_PATH = Path("data/model_errors.csv")
OUT_DIR = Path("output")
EXPECTED_COLS = ["store_id", "error_a", "error_b"]


# ---------------------------------------------------------------- validaciones
def auditar_columnas(df: pd.DataFrame) -> None:
    """El CSV debe tener EXACTAMENTE las columnas store_id, error_a, error_b."""
    cols = list(df.columns)
    if len(cols) != len(EXPECTED_COLS) or set(cols) != set(EXPECTED_COLS):
        raise ValueError(
            f"El CSV debe tener exactamente las columnas {EXPECTED_COLS}; "
            f"se encontraron {cols}"
        )


def auditar_store_id(df: pd.DataFrame) -> None:
    """Cada store_id debe ser único y no nulo (ValueError que menciona el id)."""
    sid = df["store_id"]
    nulos = sid.isna() | sid.astype(str).str.strip().eq("")
    if nulos.any():
        filas = [int(i) for i in df.index[nulos]]
        raise ValueError(f"store_id contiene identificadores nulos en filas {filas}")
    dup_mask = sid.duplicated(keep=False)
    if dup_mask.any():
        dups = sorted(sid[dup_mask].astype(str).unique().tolist())
        raise ValueError(f"store_id debe ser único; identificadores duplicados: {dups}")


def auditar_errores(df: pd.DataFrame) -> pd.DataFrame:
    """error_a y error_b: numéricos, finitos y no negativos (ValueError si no)."""
    df = df.copy()
    for col in ("error_a", "error_b"):
        coerced = pd.to_numeric(df[col], errors="coerce")
        no_num = coerced.isna() & df[col].notna()
        if no_num.any():
            malos = [(str(df.loc[i, "store_id"]), str(df.loc[i, col]))
                     for i in df.index[no_num]]
            raise ValueError(
                f"La columna '{col}' debe ser numérica; valores no numéricos "
                f"(tienda, valor): {malos}"
            )
        vals = coerced.to_numpy(dtype=float)
        finitos = np.isfinite(vals)
        if not finitos.all():
            malos = [str(df.loc[i, "store_id"]) for i in df.index[~finitos]]
            raise ValueError(
                f"La columna '{col}' debe ser finita; valores no finitos "
                f"(NaN/inf) en tiendas: {malos}"
            )
        if (vals < 0).any():
            malos = [str(df.loc[i, "store_id"]) for i in df.index[vals < 0]]
            raise ValueError(
                f"La columna '{col}' debe ser no negativa; valores negativos "
                f"en tiendas: {malos}"
            )
        df[col] = coerced.astype(float)
    return df


def verificar_pareo_por_store_id(df: pd.DataFrame) -> int:
    """
    Verifica el emparejamiento POR store_id: cada tienda aparece exactamente una
    vez y con ambos errores en su fila. Comprobar solo longitudes iguales de dos
    vectores NO demuestra que correspondan a las mismas tiendas.
    """
    sid = df["store_id"]
    if sid.isna().any():
        raise ValueError("Pareo inválido: hay filas con store_id nulo")
    if sid.duplicated().any():
        dups = sorted(sid[sid.duplicated()].astype(str).unique().tolist())
        raise ValueError(f"Pareo inválido: store_id duplicados {dups}")
    if len(df) != int(sid.nunique()):
        raise ValueError(
            "Pareo inválido: el número de filas no coincide con el número de "
            "tiendas únicas (el pareo debe hacerse por store_id)"
        )
    if df[["error_a", "error_b"]].isna().any().any():
        raise ValueError("Pareo inválido: alguna tienda carece de error_a o error_b")
    return int(len(df))


def _df_con_valor(df: pd.DataFrame, col: str, idx: int, valor) -> pd.DataFrame:
    """
    Devuelve una copia de df con df.loc[idx, col] = valor, forzando dtype object
    en la columna para que pandas nunca lance TypeError por upcast al insertar
    un valor de otro tipo (p. ej. una cadena en una columna float64).
    """
    out = df.copy()
    out[col] = out[col].astype(object)
    out.loc[idx, col] = valor
    return out


def self_test_validaciones() -> dict:
    """Comprueba (con DataFrames sintéticos en memoria) que cada validación
    lanza ValueError en los casos descritos. No toca archivos."""
    base = pd.DataFrame({
        "store_id": ["S01", "S02", "S03"],
        "error_a": [1.0, 2.0, 3.0],
        "error_b": [1.5, 2.5, 3.5],
    })
    pruebas = {}

    def espera_value_error(nombre, fn):
        try:
            fn()
            pruebas[nombre] = False
        except ValueError:
            pruebas[nombre] = True
        except Exception:
            pruebas[nombre] = False

    extra = base.copy(); extra["extra"] = 1.0
    espera_value_error("columnas_incorrectas", lambda: auditar_columnas(extra))

    falta = base.drop(columns=["error_b"])
    espera_value_error("columna_faltante", lambda: auditar_columnas(falta))

    dup = pd.concat([base, base.iloc[[0]]], ignore_index=True)
    espera_value_error("store_id_duplicado", lambda: auditar_store_id(dup))

    nulo = _df_con_valor(base, "store_id", 1, None)
    espera_value_error("store_id_nulo", lambda: auditar_store_id(nulo))

    no_num = _df_con_valor(base, "error_a", 2, "x")
    espera_value_error("error_no_numerico", lambda: auditar_errores(no_num))

    infi = _df_con_valor(base, "error_b", 0, np.inf)
    espera_value_error("error_no_finito", lambda: auditar_errores(infi))

    neg = _df_con_valor(base, "error_a", 2, -0.1)
    espera_value_error("error_negativo", lambda: auditar_errores(neg))

    return pruebas


# ------------------------------------------------------------------------ main
def main() -> None:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {CSV_PATH}")

    # 1) Carga (solo lectura; el CSV original no se modifica)
    df_raw = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    # 2) Auditoría
    auditar_columnas(df_raw)
    df = df_raw[EXPECTED_COLS].copy()
    auditar_store_id(df)
    df = auditar_errores(df)

    # 3) Verificación del pareo por store_id
    n_pares = verificar_pareo_por_store_id(df)

    # 4) Tabla pareada
    df["difference"] = df["error_a"] - df["error_b"]
    df["favors"] = np.where(df["difference"] > 0, "B",
                    np.where(df["difference"] < 0, "A", "empate"))

    diff = df["difference"].to_numpy(dtype=float)
    n_pos = int((diff > 0).sum())
    n_neg = int((diff < 0).sum())
    n_zero = int((diff == 0).sum())

    # 5) Diferencias de mayor magnitud en cada dirección
    extremos = {}
    sub_pos = df.loc[df["difference"] > 0]
    if not sub_pos.empty:
        r = sub_pos.loc[sub_pos["difference"].idxmax()]
        extremos["mayor_positiva_favors_B"] = {
            "store_id": str(r["store_id"]),
            "error_a": float(r["error_a"]),
            "error_b": float(r["error_b"]),
            "difference": float(r["difference"]),
        }
    else:
        extremos["mayor_positiva_favors_B"] = None

    sub_neg = df.loc[df["difference"] < 0]
    if not sub_neg.empty:
        r = sub_neg.loc[sub_neg["difference"].idxmin()]
        extremos["mayor_negativa_favors_A"] = {
            "store_id": str(r["store_id"]),
            "error_a": float(r["error_a"]),
            "error_b": float(r["error_b"]),
            "difference": float(r["difference"]),
        }
    else:
        extremos["mayor_negativa_favors_A"] = None

    # 6) Auto-test de que las validaciones lanzan ValueError
    self_test = self_test_validaciones()
    if not all(self_test.values()):
        fallidas = [k for k, v in self_test.items() if not v]
        raise RuntimeError(f"Auto-test de validaciones falló en: {fallidas}")

    # 7) Salidas exigidas por el enunciado
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out = df[["store_id", "error_a", "error_b", "difference", "favors"]]
    df_out.to_csv(OUT_DIR / "store_differences.csv", index=False)

    colores = {"B": "#2b7bba", "A": "#d62728", "empate": "#7f7f7f"}
    bar_colors = [colores[f] for f in df["favors"]]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(df["store_id"].astype(str), df["difference"],
           color=bar_colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0.0, color="black", linewidth=0.9)
    for x, v in zip(df["store_id"].astype(str), df["difference"]):
        ax.annotate(f"{v:+.2f}", (x, v), ha="center", fontsize=7,
                    va="bottom" if v >= 0 else "top",
                    xytext=(0, 3 if v >= 0 else -3), textcoords="offset points")
    ax.set_xlabel("store_id")
    ax.set_ylabel("difference = error_a - error_b")
    ax.set_title(f"Diferencia de error por tienda ({n_pares} pares validados por store_id)")
    ax.tick_params(axis="x", rotation=45)
    ax.margins(y=0.18)
    ax.grid(axis="y", alpha=0.3)
    handles = [Patch(color=colores["B"], label=f"favors B (n={n_pos})"),
               Patch(color=colores["A"], label=f"favors A (n={n_neg})")]
    if n_zero:
        handles.append(Patch(color=colores["empate"], label=f"empate (n={n_zero})"))
    ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "store_differences.png", dpi=120)
    plt.close(fig)

    # 8) resultados.json (contrato): vector de diferencias, conteos y extremos
    tabla_records = [
        {
            "store_id": str(r.store_id),
            "error_a": float(r.error_a),
            "error_b": float(r.error_b),
            "difference": float(r.difference),
            "favors": str(r.favors),
        }
        for r in df_out.itertuples(index=False)
    ]
    resultados = {
        "subtarea": "T2_auditar_datos_y_construir_pares",
        "csv_fuente": str(CSV_PATH),
        "csv_original_no_modificado": True,
        "columnas_esperadas": EXPECTED_COLS,
        "columnas_encontradas": [str(c) for c in df_raw.columns],
        "validaciones": {
            "columnas_exactas": True,
            "store_id_unico_y_no_nulo": True,
            "errores_numericos_finitos_no_negativos": True,
            "pareo_verificado_por_store_id": True,
            "self_test_valueerror": self_test,
        },
        "n_pares_validados": n_pares,
        "n_tiendas_unicas": int(df["store_id"].nunique()),
        "convencion_favors": "'B' si difference>0; 'A' si difference<0; 'empate' si difference==0",
        "nota_pareo": "Emparejamiento verificado por store_id (unicidad y ambos errores en la misma fila), no por longitud de vectores.",
        "diferencias": [float(v) for v in diff],
        "tabla_pareada": tabla_records,
        "conteos": {
            "positivas_favors_B": n_pos,
            "negativas_favors_A": n_neg,
            "empates_cero": n_zero,
            "total": n_pares,
        },
        "tiendas_favors_B": df.loc[df["favors"] == "B", "store_id"].astype(str).tolist(),
        "tiendas_favors_A": df.loc[df["favors"] == "A", "store_id"].astype(str).tolist(),
        "tiendas_empate": df.loc[df["favors"] == "empate", "store_id"].astype(str).tolist(),
        "extremos": extremos,
        "archivos_generados": {
            "csv": str(OUT_DIR / "store_differences.csv"),
            "figura": str(OUT_DIR / "store_differences.png"),
        },
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # 9) Resumen
    print("T2 completada: auditoría OK y tabla pareada construida.")
    print(f"  Pares validados por store_id : {n_pares}")
    print(f"  favors B (difference > 0)    : {n_pos}")
    print(f"  favors A (difference < 0)    : {n_neg}")
    print(f"  Empates (difference == 0)    : {n_zero}")
    if extremos["mayor_positiva_favors_B"] is not None:
        e = extremos["mayor_positiva_favors_B"]
        print(f"  Mayor diferencia a favor de B: {e['store_id']} ({e['difference']:+.4f})")
    if extremos["mayor_negativa_favors_A"] is not None:
        e = extremos["mayor_negativa_favors_A"]
        print(f"  Mayor diferencia a favor de A: {e['store_id']} ({e['difference']:+.4f})")
    print(f"  Salidas: {OUT_DIR / 'store_differences.csv'}, "
          f"{OUT_DIR / 'store_differences.png'}, resultados.json")


if __name__ == "__main__":
    main()
