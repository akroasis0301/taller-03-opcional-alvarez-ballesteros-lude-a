# -*- coding: utf-8 -*-
"""
T3 — Hacer visible la evidencia por tienda.

Genera una figura con una observación por tienda mostrando el signo y la
magnitud de `difference` (= error_a − error_b), con una línea de referencia
visible en cero. Cuenta cuántas tiendas favorecen a A y a B, comprueba si hay
empates e identifica las diferencias de mayor magnitud en cada dirección.

Entradas:
  - data/model_errors.csv        (datos de la tarea)
  - entrada/T2/resultados.json   (subtarea previa; verificación cruzada opcional)
Salidas:
  - output/store_differences.png (figura exigida por el enunciado)
  - output/store_differences.csv (csv exigido por el enunciado)
  - resultados.json              (contrato de la subtarea)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

RUTA_CSV = Path("data/model_errors.csv")
RUTA_T2 = Path("entrada/T2/resultados.json")
RUTA_PNG = Path("output/store_differences.png")
RUTA_CSV_OUT = Path("output/store_differences.csv")
RUTA_JSON = Path("resultados.json")

COLOR_B, COLOR_A, COLOR_TIE = "#1f77b4", "#d62728", "#7f7f7f"


def construir_tabla():
    """Tabla pareada por tienda con difference y favors (coherente con T2)."""
    if RUTA_CSV.exists():
        df = pd.read_csv(RUTA_CSV)
        req = ["store_id", "error_a", "error_b"]
        faltan = [c for c in req if c not in df.columns]
        if faltan:
            raise ValueError(f"Faltan columnas requeridas {faltan} en {RUTA_CSV}")
        if df["store_id"].isna().any() or df["store_id"].duplicated().any():
            raise ValueError("store_id debe ser único y no nulo (emparejamiento por tienda)")
        for c in ("error_a", "error_b"):
            v = pd.to_numeric(df[c], errors="coerce")
            if v.isna().any() or not np.isfinite(v.to_numpy(dtype=float)).all() or (v < 0).any():
                raise ValueError(f"La columna '{c}' debe ser numérica, finita y no negativa")
        tabla = df[req].copy()
        tabla["difference"] = tabla["error_a"] - tabla["error_b"]
        tabla["favors"] = np.where(tabla["difference"] > 0, "B",
                          np.where(tabla["difference"] < 0, "A", "tie"))
        fuente = str(RUTA_CSV)
    elif RUTA_T2.exists():
        with open(RUTA_T2, encoding="utf-8") as f:
            t2 = json.load(f)
        tabla = pd.DataFrame(t2["table"])[["store_id", "error_a", "error_b",
                                           "difference", "favors"]].copy()
        fuente = f"{RUTA_T2} (tabla de T2)"
    else:
        raise FileNotFoundError("No se encontró data/model_errors.csv ni entrada/T2/resultados.json")
    return tabla, fuente


def verificar_contra_T2(tabla):
    """Compara la tabla reconstruida con el resultado de T2, si está disponible."""
    if not RUTA_T2.exists():
        return None
    try:
        with open(RUTA_T2, encoding="utf-8") as f:
            t2 = json.load(f)
        ref = pd.DataFrame(t2["table"])
        m = tabla.merge(ref, on="store_id", suffixes=("_t3", "_t2"),
                        how="outer", indicator=True)
        ok = (
            (m["_merge"] == "both").all()
            and np.allclose(m["difference_t3"].to_numpy(dtype=float),
                            m["difference_t2"].to_numpy(dtype=float), atol=1e-9)
            and (m["favors_t3"] == m["favors_t2"]).all()
        )
        return bool(ok)
    except Exception:
        return False


def extremo(sub, lado):
    """Registro de mayor magnitud dentro de `sub` (lado: 'positiva' o 'negativa')."""
    if sub.empty:
        return None
    idx = sub["difference"].idxmax() if lado == "positiva" else sub["difference"].idxmin()
    r = sub.loc[idx]
    return {
        "store_id": str(r["store_id"]),
        "error_a": float(r["error_a"]),
        "error_b": float(r["error_b"]),
        "difference": float(r["difference"]),
        "magnitude": float(abs(r["difference"])),
    }


def main():
    tabla, fuente = construir_tabla()
    t2_ok = verificar_contra_T2(tabla)

    n_stores = int(len(tabla))
    n_A = int((tabla["favors"] == "A").sum())
    n_B = int((tabla["favors"] == "B").sum())
    n_ties = int((tabla["favors"] == "tie").sum())
    stores_A = tabla.loc[tabla["favors"] == "A", "store_id"].astype(str).tolist()
    stores_B = tabla.loc[tabla["favors"] == "B", "store_id"].astype(str).tolist()
    stores_tie = tabla.loc[tabla["favors"] == "tie", "store_id"].astype(str).tolist()

    max_hacia_B = extremo(tabla[tabla["difference"] > 0], "positiva")
    max_hacia_A = extremo(tabla[tabla["difference"] < 0], "negativa")

    # ------------------------------------------------------------- figura
    RUTA_PNG.parent.mkdir(parents=True, exist_ok=True)
    orden = tabla.sort_values("difference", ascending=True).reset_index(drop=True)
    colores = [COLOR_B if f == "B" else (COLOR_A if f == "A" else COLOR_TIE)
               for f in orden["favors"]]

    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(orden))
    vals = orden["difference"].to_numpy(dtype=float)
    ax.barh(y, vals, color=colores, edgecolor="black", linewidth=0.5, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(orden["store_id"].astype(str).tolist())
    ax.axvline(0, color="black", linewidth=1.4, zorder=3)  # referencia visible en cero
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    lim = max(2.6, float(np.abs(vals).max()) * 1.18)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("difference = error_a − error_b")
    ax.set_ylabel("store_id")
    ax.set_title(f"Diferencia de error por tienda (n = {n_stores}) — "
                 f"favorecen a A: {n_A} · favorecen a B: {n_B} · empates: {n_ties}")
    for yi, v in zip(y, vals):
        if v >= 0:
            ax.text(v + lim * 0.015, yi, f"{v:+.2f}", va="center", ha="left", fontsize=8)
        else:
            ax.text(v - lim * 0.015, yi, f"{v:+.2f}", va="center", ha="right", fontsize=8)
    handles = [Patch(facecolor=COLOR_B, edgecolor="black",
                     label="difference > 0 → favorece a B"),
               Patch(facecolor=COLOR_A, edgecolor="black",
                     label="difference < 0 → favorece a A")]
    if n_ties:
        handles.append(Patch(facecolor=COLOR_TIE, edgecolor="black",
                             label="difference = 0 → empate"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    plt.savefig(RUTA_PNG.as_posix(), dpi=120)
    plt.close(fig)

    # --------------------------------------------------------------- csv
    tabla_out = tabla[["store_id", "error_a", "error_b", "difference", "favors"]].copy()
    RUTA_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    tabla_out.to_csv(RUTA_CSV_OUT, index=False)

    # -------------------------------------------------------- resultados
    resultados = {
        "subtask": "T3",
        "description": ("Figura con una observación por tienda del signo y la magnitud de "
                        "difference, con referencia en cero; conteos por modelo, empates y "
                        "diferencias de mayor magnitud en cada dirección"),
        "source": fuente,
        "figure": RUTA_PNG.as_posix(),
        "figure_exists": RUTA_PNG.exists(),
        "csv": RUTA_CSV_OUT.as_posix(),
        "csv_exists": RUTA_CSV_OUT.exists(),
        "n_stores": n_stores,
        "n_favors_A": n_A,
        "n_favors_B": n_B,
        "n_ties": n_ties,
        "ties_exist": bool(n_ties > 0),
        "stores_favoring_A": stores_A,
        "stores_favoring_B": stores_B,
        "stores_tied": stores_tie,
        "max_difference_favoring_B": max_hacia_B,
        "max_difference_favoring_A": max_hacia_A,
        "max_abs_difference": float(tabla["difference"].abs().max()),
        "mean_difference": float(tabla["difference"].mean()),
        "t2_crosscheck_consistent": t2_ok,
        "table": tabla_out.to_dict(orient="records"),
    }
    with open(RUTA_JSON, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ----------------------------------------------------------- resumen
    print("T3 — Evidencia por tienda")
    print(f"  Figura : {RUTA_PNG.as_posix()} (existe: {RUTA_PNG.exists()})")
    print(f"  CSV    : {RUTA_CSV_OUT.as_posix()} (existe: {RUTA_CSV_OUT.exists()})")
    print(f"  Tiendas: {n_stores} | favorecen a A: {n_A} | favorecen a B: {n_B} | empates: {n_ties}")
    if max_hacia_B:
        print(f"  Mayor diferencia a favor de B: {max_hacia_B['store_id']} "
              f"({max_hacia_B['difference']:+.4f}, magnitud {max_hacia_B['magnitude']:.4f})")
    if max_hacia_A:
        print(f"  Mayor diferencia a favor de A: {max_hacia_A['store_id']} "
              f"({max_hacia_A['difference']:+.4f}, magnitud {max_hacia_A['magnitude']:.4f})")
    if t2_ok is not None:
        print(f"  Verificación contra T2: {'consistente' if t2_ok else 'INCONSISTENTE'}")


if __name__ == "__main__":
    main()
