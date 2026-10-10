# -*- coding: utf-8 -*-
"""
T3 — Hacer visible la evidencia por tienda.

Crea una figura con UNA observación por tienda (barras horizontales) que muestra
el signo y la magnitud de `difference` = error_a - error_b, con una línea de
referencia visible en cero, y la guarda en output/store_differences.png.

Fuente de la tabla pareada (en orden de preferencia):
  1) 'tabla_pareada' de entrada/T2/resultados.json (resultado de la subtarea T2)
  2) CSV de T2 bajo entrada/*/output/store_differences.csv
  3) data/model_errors.csv (reconstruida con las mismas validaciones de T2)

Salidas:
  - output/store_differences.png  (figura exigida por el enunciado)
  - output/store_differences.csv  (tabla pareada, también exigida)
  - resultados.json               (contrato numérico de la subtarea)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RUTA_CSV_FUENTE = Path("data/model_errors.csv")
RUTA_FIGURA = Path("output/store_differences.png")
RUTA_CSV_SALIDA = Path("output/store_differences.csv")
RUTA_RESULTADOS = Path("resultados.json")

COLUMNAS_BASE = ["store_id", "error_a", "error_b"]
COLOR_B = "#1f77b4"        # difference > 0 -> favorece al modelo B
COLOR_A = "#d62728"        # difference < 0 -> favorece al modelo A
COLOR_EMPATE = "#7f7f7f"


# ---------------------------------------------------------------------------
# Validación y construcción de la tabla pareada (misma convención que T2)
# ---------------------------------------------------------------------------
def validar_y_construir(df: pd.DataFrame) -> pd.DataFrame:
    """Valida columnas, identificadores y errores; agrega difference y favors."""
    if list(df.columns) != COLUMNAS_BASE:
        raise ValueError(
            f"Columnas incorrectas: se esperaban exactamente {COLUMNAS_BASE} "
            f"y se encontraron {list(df.columns)}"
        )
    if df["store_id"].isna().any():
        filas = df.index[df["store_id"].isna()].tolist()
        raise ValueError(
            f"store_id contiene valores nulos en las filas {filas}; "
            "cada identificador de tienda debe ser no nulo"
        )
    duplicados = sorted(set(df.loc[df["store_id"].duplicated(), "store_id"].astype(str)))
    if duplicados:
        raise ValueError(
            f"store_id duplicado para {duplicados}; "
            "cada identificador de tienda debe ser único"
        )
    for col in ("error_a", "error_b"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"La columna '{col}' debe ser numérica")
        valores = df[col].to_numpy(dtype=float)
        if not np.isfinite(valores).all():
            raise ValueError(f"La columna '{col}' contiene valores no finitos")
        if (valores < 0).any():
            raise ValueError(f"La columna '{col}' contiene valores negativos")
    out = df.loc[:, COLUMNAS_BASE].copy()
    out["store_id"] = out["store_id"].astype(str)
    out["error_a"] = out["error_a"].astype(float)
    out["error_b"] = out["error_b"].astype(float)
    out["difference"] = out["error_a"] - out["error_b"]
    out["favors"] = np.where(
        out["difference"] > 0, "B", np.where(out["difference"] < 0, "A", "empate")
    )
    return out.sort_values("store_id").reset_index(drop=True)


def _rutas_resultados_previos():
    rutas = [Path("entrada/T2/resultados.json")]
    if Path("entrada").is_dir():
        for p in sorted(Path("entrada").glob("*/resultados.json")):
            if p not in rutas:
                rutas.append(p)
    return rutas


def cargar_tabla_pareada():
    """Devuelve (df_pareado, descripcion_fuente, resultados_previos_T2 o None)."""
    for ruta in _rutas_resultados_previos():
        try:
            with ruta.open("r", encoding="utf-8") as f:
                previo = json.load(f)
            tabla = previo.get("tabla_pareada") if isinstance(previo, dict) else None
            if tabla:
                df_prev = pd.DataFrame(tabla)
                if all(c in df_prev.columns for c in COLUMNAS_BASE):
                    return (
                        validar_y_construir(df_prev.loc[:, COLUMNAS_BASE]),
                        str(ruta),
                        previo,
                    )
        except Exception:
            continue
    if Path("entrada").is_dir():
        for ruta in sorted(Path("entrada").glob("*/output/store_differences.csv")):
            try:
                df_prev = pd.read_csv(ruta)
                if all(c in df_prev.columns for c in COLUMNAS_BASE):
                    return (
                        validar_y_construir(df_prev.loc[:, COLUMNAS_BASE]),
                        str(ruta),
                        None,
                    )
            except Exception:
                continue
    if not RUTA_CSV_FUENTE.exists():
        raise FileNotFoundError(
            "No se halló la tabla pareada de T2 (entrada/T2/) ni el CSV fuente "
            f"{RUTA_CSV_FUENTE}"
        )
    df_fuente = pd.read_csv(RUTA_CSV_FUENTE)
    if not all(c in df_fuente.columns for c in COLUMNAS_BASE):
        raise ValueError(f"{RUTA_CSV_FUENTE} no contiene las columnas {COLUMNAS_BASE}")
    return validar_y_construir(df_fuente.loc[:, COLUMNAS_BASE]), str(RUTA_CSV_FUENTE), None


def verificaciones_cruzadas(df: pd.DataFrame, previo) -> dict:
    ver = {}
    if RUTA_CSV_FUENTE.exists():
        try:
            ref = validar_y_construir(pd.read_csv(RUTA_CSV_FUENTE).loc[:, COLUMNAS_BASE])
            ver["coincide_con_data_model_errors_csv"] = bool(
                list(ref["store_id"]) == list(df["store_id"])
                and np.allclose(
                    ref[["error_a", "error_b"]].to_numpy(dtype=float),
                    df[["error_a", "error_b"]].to_numpy(dtype=float),
                )
            )
        except Exception:
            ver["coincide_con_data_model_errors_csv"] = None
    if isinstance(previo, dict) and previo.get("tabla_pareada"):
        try:
            ref = pd.DataFrame(previo["tabla_pareada"])
            if "difference" in ref.columns:
                m = df.merge(
                    ref[["store_id", "difference"]], on="store_id", suffixes=("", "_t2")
                )
                ver["coincide_con_diferencias_de_T2"] = bool(
                    np.allclose(
                        m["difference"].to_numpy(dtype=float),
                        m["difference_t2"].to_numpy(dtype=float),
                    )
                )
        except Exception:
            ver["coincide_con_diferencias_de_T2"] = None
    return ver


# ---------------------------------------------------------------------------
# Figura: una observación por tienda, signo y magnitud, referencia en cero
# ---------------------------------------------------------------------------
def crear_figura(df: pd.DataFrame, ruta_figura: Path) -> None:
    df = df.sort_values("store_id").reset_index(drop=True)
    n = len(df)
    y = np.arange(n)
    diffs = df["difference"].to_numpy(dtype=float)
    colores = [
        COLOR_B if f == "B" else COLOR_A if f == "A" else COLOR_EMPATE
        for f in df["favors"]
    ]

    fig = plt.figure(figsize=(9.5, 7.8))
    ax = fig.add_subplot(111)

    # Una barra por tienda: la longitud da la magnitud, el lado da el signo
    ax.barh(y, diffs, color=colores, edgecolor="black", linewidth=0.6,
            height=0.62, zorder=2)

    # Referencia VISIBLE en cero
    ax.axvline(0, color="black", linewidth=2.2, zorder=3)

    # Signo y magnitud de cada diferencia, escritos al extremo de cada barra
    margen = 0.05 * max(1.0, float(np.abs(diffs).max())) if n else 0.05
    for yi, d in zip(y, diffs):
        if d > 0:
            ax.text(d + margen, yi, f"{d:+.2f}", va="center", ha="left",
                    fontsize=9, color=COLOR_B, zorder=4)
        elif d < 0:
            ax.text(d - margen, yi, f"{d:+.2f}", va="center", ha="right",
                    fontsize=9, color=COLOR_A, zorder=4)
        else:
            ax.text(margen, yi, "0.00 (empate)", va="center", ha="left",
                    fontsize=9, color=COLOR_EMPATE, zorder=4)

    # Eje Y: una etiqueta por tienda (las 16 distinguibles)
    ax.set_yticks(y)
    ax.set_yticklabels(df["store_id"].tolist(), fontsize=10)
    ax.invert_yaxis()  # S01 arriba ... S16 abajo
    ax.set_ylim(n - 0.6, -0.9)

    limite = float(np.abs(diffs).max()) * 1.4 + 0.4 if n else 1.0
    ax.set_xlim(-limite, limite)
    ax.set_xticks(np.arange(-4, 5, 1))
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)

    n_b = int((df["favors"] == "B").sum())
    n_a = int((df["favors"] == "A").sum())
    n_e = int((df["favors"] == "empate").sum())

    ax.set_xlabel("difference = error_a - error_b", fontsize=11)
    ax.set_ylabel("store_id (tienda)", fontsize=11)
    ax.set_title(
        "Diferencia de error por tienda: una observación por tienda\n"
        "Signo y magnitud de difference = error_a - error_b "
        "(línea negra = referencia en cero)\n"
        f"{n_b} tiendas favorecen al modelo B · {n_a} favorecen al modelo A · "
        f"{n_e} empates",
        fontsize=11,
    )

    handles = [
        Patch(facecolor=COLOR_B, edgecolor="black",
              label="difference > 0 (favorece a B)"),
        Patch(facecolor=COLOR_A, edgecolor="black",
              label="difference < 0 (favorece a A)"),
        Line2D([0], [0], color="black", linewidth=2.2,
               label="referencia en cero (difference = 0)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.115),
              ncol=3, fontsize=8.5, frameon=False, handlelength=1.6,
              columnspacing=1.4)

    fig.subplots_adjust(left=0.115, right=0.965, top=0.885, bottom=0.16)
    ruta_figura.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(ruta_figura), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main() -> None:
    df, fuente, previo = cargar_tabla_pareada()
    verificaciones = verificaciones_cruzadas(df, previo)

    # CSV exigido por el enunciado (tabla pareada completa)
    RUTA_CSV_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUTA_CSV_SALIDA, index=False, encoding="utf-8")

    # Figura exigida por el enunciado
    crear_figura(df, RUTA_FIGURA)

    # Verificación de la figura generada
    figura_existe = RUTA_FIGURA.exists() and RUTA_FIGURA.stat().st_size > 0
    es_png = False
    dims = None
    if figura_existe:
        with RUTA_FIGURA.open("rb") as f:
            es_png = f.read(8) == b"\x89PNG\r\n\x1a\n"
        try:
            img = mpimg.imread(str(RUTA_FIGURA))
            dims = {"alto_px": int(img.shape[0]), "ancho_px": int(img.shape[1])}
        except Exception:
            dims = None

    # Verificación del CSV generado
    csv_ok = False
    if RUTA_CSV_SALIDA.exists():
        try:
            chk = pd.read_csv(RUTA_CSV_SALIDA)
            csv_ok = (
                list(chk.columns)
                == ["store_id", "error_a", "error_b", "difference", "favors"]
                and len(chk) == len(df)
                and bool(
                    np.allclose(
                        chk["difference"].to_numpy(dtype=float),
                        df["difference"].to_numpy(dtype=float),
                    )
                )
            )
        except Exception:
            csv_ok = False

    # Cifras para el informe (contrato de la subtarea)
    pos = df[df["difference"] > 0]
    neg = df[df["difference"] < 0]
    emp = df[df["difference"] == 0]
    i_abs = df["difference"].abs().idxmax()

    mayor_b = None
    if not pos.empty:
        j = pos["difference"].idxmax()
        mayor_b = {"store_id": str(pos.loc[j, "store_id"]),
                   "difference": float(pos.loc[j, "difference"])}
    mayor_a = None
    if not neg.empty:
        j = neg["difference"].idxmin()
        mayor_a = {"store_id": str(neg.loc[j, "store_id"]),
                   "difference": float(neg.loc[j, "difference"])}

    por_tienda = [
        {
            "store_id": str(r.store_id),
            "error_a": float(r.error_a),
            "error_b": float(r.error_b),
            "difference": float(r.difference),
            "favors": str(r.favors),
        }
        for r in df.itertuples(index=False)
    ]

    resultados = {
        "subtarea": "T3_figura_diferencias_por_tienda",
        "descripcion": (
            "Figura con una observación por tienda (barras horizontales) que muestra "
            "el signo y la magnitud de difference = error_a - error_b, con línea de "
            "referencia visible en cero."
        ),
        "fuente_tabla": fuente,
        "verificaciones_cruzadas": verificaciones,
        "n_tiendas": int(len(df)),
        "convencion_favors": "'B' si difference>0; 'A' si difference<0; 'empate' si difference==0",
        "por_tienda": por_tienda,
        "tiendas_favors_B": {"conteo": int(len(pos)), "tiendas": pos["store_id"].tolist()},
        "tiendas_favors_A": {"conteo": int(len(neg)), "tiendas": neg["store_id"].tolist()},
        "tiendas_empate": {"conteo": int(len(emp)), "tiendas": emp["store_id"].tolist()},
        "extremos": {
            "mayor_diferencia_favors_B": mayor_b,
            "mayor_diferencia_favors_A": mayor_a,
            "mayor_magnitud_absoluta": {
                "store_id": str(df.loc[i_abs, "store_id"]),
                "difference": float(df.loc[i_abs, "difference"]),
                "valor_absoluto": float(abs(df.loc[i_abs, "difference"])),
            },
        },
        "figura": {
            "ruta": str(RUTA_FIGURA),
            "existe": bool(figura_existe),
            "es_png_valido": bool(es_png),
            "bytes": int(RUTA_FIGURA.stat().st_size) if figura_existe else 0,
            "dimensiones_px": dims,
            "dpi": 120,
            "tipo": "barras horizontales, una por tienda, coloreadas según el signo",
            "n_barras": int(len(df)),
            "tiendas_en_eje_y": df["store_id"].tolist(),
            "referencia_cero": "línea vertical negra en x=0 (axvline), también en la leyenda",
            "etiquetas_signo_magnitud": "valor con signo a 2 decimales en el extremo de cada barra",
        },
        "csv": {
            "ruta": str(RUTA_CSV_SALIDA),
            "existe": bool(csv_ok),
            "filas": int(len(df)),
            "columnas": ["store_id", "error_a", "error_b", "difference", "favors"],
        },
    }
    RUTA_RESULTADOS.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 64)
    print("T3 — Figura de diferencias por tienda")
    print("=" * 64)
    print(f"Fuente de la tabla pareada : {fuente}")
    print(f"Tiendas representadas      : {len(df)}")
    print(f"Favorecen a B (diff > 0)   : {len(pos)} -> {', '.join(pos['store_id'])}")
    print(f"Favorecen a A (diff < 0)   : {len(neg)} -> {', '.join(neg['store_id'])}")
    print(f"Empates (diff = 0)         : {len(emp)}")
    if mayor_b:
        print(f"Mayor diferencia hacia B   : {mayor_b['store_id']} ({mayor_b['difference']:+.2f})")
    if mayor_a:
        print(f"Mayor diferencia hacia A   : {mayor_a['store_id']} ({mayor_a['difference']:+.2f})")
    print(f"Figura                     : {RUTA_FIGURA} "
          f"(existe={figura_existe}, png_valido={es_png}, dims={dims})")
    print(f"CSV                        : {RUTA_CSV_SALIDA} (verificado={csv_ok})")
    print(f"Resultados                 : {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
