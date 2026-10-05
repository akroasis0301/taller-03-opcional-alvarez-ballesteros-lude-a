#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2 — Auditar los datos y construir los pares.

Carga data/model_errors.csv con validaciones estrictas:
  * exactamente las columnas store_id, error_a, error_b;
  * store_id único y no nulo (ValueError que menciona el identificador);
  * errores numéricos, finitos y no negativos (ValueError en caso contrario);
  * el CSV original NO se modifica (se verifica comparando bytes).

Construye la tabla pareada (store_id, error_a, error_b, difference, favors) y
verifica el emparejamiento POR store_id (merge uno a uno contra una relectura
del CSV desde disco), no solo por la longitud de los vectores.

Salidas:
  * output/store_differences.csv  (tabla pareada, 5 columnas)
  * output/store_differences.png  (figura de diferencias por tienda)
  * resultados.json               (CONTRATO de la subtarea: todas las cifras)
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
PNG_PATH = OUT_DIR / "store_differences.png"
CSV_OUT_PATH = OUT_DIR / "store_differences.csv"
RESULTS_PATH = Path("resultados.json")  # contrato: SIEMPRE escribir resultados.json

EXPECTED_COLS = ["store_id", "error_a", "error_b"]


# ----------------------------------------------------------------------------
# Validaciones (cada una lanza ValueError con un mensaje descriptivo)
# ----------------------------------------------------------------------------
def _validar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """El CSV debe tener EXACTAMENTE las columnas store_id, error_a, error_b."""
    if list(df.columns) != EXPECTED_COLS:
        if set(df.columns) == set(EXPECTED_COLS) and len(df.columns) == 3:
            df = df[EXPECTED_COLS].copy()  # mismo conjunto, otro orden: reordenar
        else:
            raise ValueError(
                f"El CSV debe tener exactamente las columnas {EXPECTED_COLS}; "
                f"columnas encontradas: {list(df.columns)}"
            )
    return df


def _validar_store_id(df: pd.DataFrame) -> pd.DataFrame:
    """store_id: no nulo y único; el ValueError menciona el identificador."""
    nulos = df["store_id"].isna()
    if nulos.any():
        filas = df.index[nulos].tolist()
        raise ValueError(
            f"store_id contiene valores nulos (identificador ausente) "
            f"en las filas {filas}"
        )
    df["store_id"] = df["store_id"].astype(str)

    dup_mask = df["store_id"].duplicated(keep=False)
    if dup_mask.any():
        duplicados = sorted(df.loc[dup_mask, "store_id"].unique().tolist())
        raise ValueError(
            f"store_id debe ser único; identificadores duplicados: {duplicados}"
        )
    return df


def _validar_errores(df: pd.DataFrame) -> pd.DataFrame:
    """error_a y error_b: numéricos, finitos y no negativos."""
    for col in ("error_a", "error_b"):
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            vals = s.astype(float)
        else:
            coerced = pd.to_numeric(s, errors="coerce")
            no_num = coerced.isna() & s.notna()
            if no_num.any():
                malos = s[no_num].tolist()
                ids = df.loc[no_num, "store_id"].tolist()
                raise ValueError(
                    f"La columna {col} contiene valores no numéricos {malos} "
                    f"en tiendas {ids}"
                )
            vals = coerced.astype(float)

        if vals.isna().any():
            ids = df.loc[vals.isna(), "store_id"].tolist()
            raise ValueError(
                f"La columna {col} contiene valores nulos (no finitos) "
                f"en tiendas {ids}"
            )
        if not np.isfinite(vals.to_numpy()).all():
            finitos = np.isfinite(vals.to_numpy())
            ids = df.loc[~finitos, "store_id"].tolist()
            raise ValueError(
                f"La columna {col} contiene valores no finitos (inf/NaN) "
                f"en tiendas {ids}"
            )
        negativos = vals < 0
        if negativos.any():
            detalle = {
                str(sid): float(v)
                for sid, v in zip(df.loc[negativos, "store_id"], vals[negativos])
            }
            raise ValueError(
                f"La columna {col} contiene valores negativos, no permitidos: "
                f"{detalle}"
            )
        df[col] = vals
    return df


def cargar_y_validar(path: Path) -> pd.DataFrame:
    """Lee el CSV y ejecuta todas las validaciones (sin modificar el archivo)."""
    df = pd.read_csv(path, dtype={"store_id": str})
    df = _validar_columnas(df)
    df = _validar_store_id(df)
    df = _validar_errores(df)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Tabla pareada y verificación del emparejamiento
# ----------------------------------------------------------------------------
def construir_tabla_pareada(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva las 3 columnas originales y añade difference y favors."""
    paired = df[EXPECTED_COLS].copy()
    paired["difference"] = paired["error_a"] - paired["error_b"]
    # 'B' si la diferencia es positiva, 'A' si es negativa.
    # Diferencia exactamente 0: el enunciado no la define; se registra como 'tie'.
    paired["favors"] = np.select(
        [paired["difference"] > 0, paired["difference"] < 0],
        ["B", "A"],
        default="tie",
    )
    return paired.reset_index(drop=True)


def verificar_emparejamiento(paired: pd.DataFrame, src: pd.DataFrame) -> bool:
    """
    Confirma el pareo POR store_id: merge uno a uno contra el CSV releído,
    comprobando que cada tienda conserva sus valores de error_a y error_b.
    Comprobar solo longitudes iguales NO demuestra el emparejamiento.
    """
    ids_iguales = set(paired["store_id"]) == set(src["store_id"])
    merged = paired.merge(
        src[EXPECTED_COLS], on="store_id", how="left",
        suffixes=("", "_src"), validate="one_to_one",
    )
    valores_iguales = bool(
        (merged["error_a"] == merged["error_a_src"]).all()
        and (merged["error_b"] == merged["error_b_src"]).all()
    )
    return ids_iguales and len(merged) == len(paired) and valores_iguales


# ----------------------------------------------------------------------------
# Figura
# ----------------------------------------------------------------------------
def crear_figura(paired: pd.DataFrame, png_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colores = np.where(paired["favors"].to_numpy() == "B", "#2ca02c", "#d62728")
    ax.bar(paired["store_id"], paired["difference"], color=colores,
           edgecolor="black", linewidth=0.6)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("store_id")
    ax.set_ylabel("difference = error_a - error_b")
    ax.set_title("Diferencia de error por tienda (pareo verificado por store_id)")
    ax.legend(handles=[
        Patch(facecolor="#2ca02c", label="favors = 'B' (difference > 0)"),
        Patch(facecolor="#d62728", label="favors = 'A' (difference < 0)"),
    ])
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------
def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {CSV_PATH}")

    original_bytes = CSV_PATH.read_bytes()  # para demostrar que no se modifica

    # 1) Carga con validaciones (deben ejecutarse sin errores)
    df = cargar_y_validar(CSV_PATH)

    # 2) Tabla pareada: 3 columnas originales + difference + favors
    paired = construir_tabla_pareada(df)

    # 3) Verificación del emparejamiento por store_id (relectura independiente)
    src = cargar_y_validar(CSV_PATH)
    pairing_ok = verificar_emparejamiento(paired, src)
    if not pairing_ok:
        raise ValueError(
            "El emparejamiento por store_id no se pudo verificar: los pares "
            "(error_a, error_b) no corresponden a las mismas tiendas."
        )

    # 4) Salidas exigidas por el enunciado
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paired.to_csv(CSV_OUT_PATH, index=False)
    crear_figura(paired, PNG_PATH)

    csv_unchanged = CSV_PATH.read_bytes() == original_bytes

    # 5) CONTRATO: escribir resultados.json con TODAS las cifras de la subtarea
    tabla = [
        {
            "store_id": str(r.store_id),
            "error_a": float(r.error_a),
            "error_b": float(r.error_b),
            "difference": float(r.difference),
            "favors": str(r.favors),
        }
        for r in paired.itertuples(index=False)
    ]
    resultados = {
        "subtask": "T2",
        "description": "Tabla pareada error_a vs error_b por store_id, "
                       "con difference y favors",
        "csv_source": str(CSV_PATH),
        "validations_passed": True,
        "csv_original_unchanged": bool(csv_unchanged),
        "pairing_verified_by_store_id": bool(pairing_ok),
        "pairing_method": "merge uno a uno sobre store_id contra relectura del "
                          "CSV (no solo longitud de los vectores)",
        "n_rows": int(len(paired)),
        "n_unique_store_ids": int(paired["store_id"].nunique()),
        "columns": list(paired.columns),
        "n_favors_A": int((paired["favors"] == "A").sum()),
        "n_favors_B": int((paired["favors"] == "B").sum()),
        "n_ties": int((paired["favors"] == "tie").sum()),
        "mean_difference": float(paired["difference"].mean()),
        "table": tabla,
        "outputs": [str(CSV_OUT_PATH), str(PNG_PATH)],
    }
    RESULTS_PATH.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Verificación explícita de que el contrato quedó escrito en disco
    if not RESULTS_PATH.exists():
        raise RuntimeError(f"No se pudo escribir el contrato {RESULTS_PATH}")

    # 6) Resumen breve
    print("T2 completada.")
    print(f"  Validaciones ejecutadas sin errores: True")
    print(f"  Filas pareadas: {len(paired)} "
          f"(tiendas únicas: {paired['store_id'].nunique()})")
    print(f"  Columnas: {list(paired.columns)}")
    print(f"  favors -> A: {resultados['n_favors_A']}, "
          f"B: {resultados['n_favors_B']}, ties: {resultados['n_ties']}")
    print(f"  Media de difference: {resultados['mean_difference']:.6f}")
    print(f"  Emparejamiento verificado por store_id: {pairing_ok}")
    print(f"  CSV original sin modificar: {csv_unchanged}")
    print(f"  Archivos escritos: {CSV_OUT_PATH}, {PNG_PATH}, {RESULTS_PATH}")


if __name__ == "__main__":
    main()
