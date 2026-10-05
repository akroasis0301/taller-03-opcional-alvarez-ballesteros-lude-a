#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 — Hacer visible la evidencia por tienda.

Figura con UNA observación por tienda que muestra el signo y la magnitud de
difference = error_a − error_b, con una línea de referencia visible en cero y
color/dirección que distinguen las diferencias que favorecen a A (difference<0)
y a B (difference>0).

Salidas (rutas relativas):
  - output/store_differences.png   (figura exigida por el enunciado, dpi=120)
  - output/store_differences.csv   (tabla exigida por el enunciado)
  - resultados.json                (contrato de cifras de la subtarea)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ rutas
RUTA_CSV = Path("data/model_errors.csv")
RUTA_T2 = Path("entrada/T2/resultados.json")
DIR_SALIDA = Path("output")
RUTA_PNG = DIR_SALIDA / "store_differences.png"
RUTA_CSV_SALIDA = DIR_SALIDA / "store_differences.csv"
RUTA_RESULTADOS = Path("resultados.json")

# ------------------------------------------------------------------ carga y tabla pareada
df = pd.read_csv(RUTA_CSV)

columnas_esperadas = ["store_id", "error_a", "error_b"]
if list(df.columns) != columnas_esperadas:
    raise ValueError(
        f"Se esperaban exactamente {columnas_esperadas}; encontrado: {list(df.columns)}"
    )
if df["store_id"].isna().any() or df["store_id"].duplicated().any():
    raise ValueError("store_id con nulos o duplicados; no se puede emparejar por tienda")

# difference y favors con la misma definición de T2
df["difference"] = df["error_a"] - df["error_b"]
df["favors"] = np.where(
    df["difference"] > 0, "B", np.where(df["difference"] < 0, "A", "tie")
)

n_tiendas = len(df)
store_ids = df["store_id"].tolist()
if n_tiendas != 16:
    print(f"ADVERTENCIA: se esperaban 16 tiendas y hay {n_tiendas}.")

# ------------------------------------------------------------------ consistencia con T2
consistencia_t2 = {"archivo_consultado": str(RUTA_T2), "disponible": False,
                   "coincide": None}
if RUTA_T2.exists():
    try:
        with open(RUTA_T2, encoding="utf-8") as f:
            t2 = json.load(f)
        tabla_t2 = pd.DataFrame(t2["tabla_pareada"])
        cmp_df = df[["store_id", "difference", "favors"]].merge(
            tabla_t2[["store_id", "difference", "favors"]],
            on="store_id", how="inner", suffixes=("_t3", "_t2"),
        )
        mismas = len(cmp_df) == len(df) == len(tabla_t2)
        dif_ok = bool(mismas and np.allclose(
            cmp_df["difference_t3"], cmp_df["difference_t2"], atol=1e-9))
        fav_ok = bool(mismas and (cmp_df["favors_t3"] == cmp_df["favors_t2"]).all())
        consistencia_t2 = {
            "archivo_consultado": str(RUTA_T2),
            "disponible": True,
            "mismas_tiendas": bool(mismas),
            "difference_coincide": dif_ok,
            "favors_coincide": fav_ok,
            "coincide": bool(dif_ok and fav_ok),
        }
    except Exception as exc:  # lectura de T2 es opcional; no bloquea la subtarea
        consistencia_t2["error_lectura"] = repr(exc)

# ------------------------------------------------------------------ cifras para el informe
n_A = int((df["favors"] == "A").sum())
n_B = int((df["favors"] == "B").sum())
n_ties = int((df["favors"] == "tie").sum())
ids_A = df.loc[df["favors"] == "A", "store_id"].tolist()
ids_B = df.loc[df["favors"] == "B", "store_id"].tolist()
ids_tie = df.loc[df["favors"] == "tie", "store_id"].tolist()

pos = df[df["difference"] > 0]
neg = df[df["difference"] < 0]

max_B = None
if not pos.empty:
    r = pos.loc[pos["difference"].idxmax()]
    max_B = {"store_id": str(r["store_id"]), "difference": float(r["difference"]),
             "error_a": float(r["error_a"]), "error_b": float(r["error_b"]),
             "favors": "B", "direccion": "positiva"}

max_A = None
if not neg.empty:
    r = neg.loc[neg["difference"].idxmin()]
    max_A = {"store_id": str(r["store_id"]), "difference": float(r["difference"]),
             "error_a": float(r["error_a"]), "error_b": float(r["error_b"]),
             "favors": "A", "direccion": "negativa"}

# ------------------------------------------------------------------ figura
COLOR_A = "#d62728"    # rojo: favorece a A (difference < 0)
COLOR_B = "#1f77b4"    # azul: favorece a B (difference > 0)
COLOR_TIE = "#7f7f7f"  # gris: empate (difference == 0), si lo hubiera

fig, ax = plt.subplots(figsize=(9, 7))
y_pos = np.arange(n_tiendas)
colores = df["favors"].map({"A": COLOR_A, "B": COLOR_B, "tie": COLOR_TIE}).to_numpy()

# Una barra por tienda: la dirección (derecha/izquierda) y el color codifican el
# signo; la longitud codifica la magnitud de difference.
ax.barh(y_pos, df["difference"].to_numpy(), color=colores,
        edgecolor="black", linewidth=0.6, height=0.62, zorder=3)

# Referencia claramente visible en cero
ax.axvline(0, color="black", linewidth=2.0, zorder=5)

# Etiqueta con signo y magnitud al extremo de cada barra
for yi, d in zip(y_pos, df["difference"].to_numpy()):
    if d >= 0:
        ax.text(d + 0.05, yi, f"{d:+.2f}", va="center", ha="left",
                fontsize=9, zorder=6)
    else:
        ax.text(d - 0.05, yi, f"{d:+.2f}", va="center", ha="right",
                fontsize=9, zorder=6)

ax.set_yticks(y_pos)
ax.set_yticklabels(store_ids)
ax.invert_yaxis()  # S01 arriba ... S16 abajo (orden del CSV)
ax.set_xlim(float(df["difference"].min()) - 0.9, float(df["difference"].max()) + 0.9)
ax.set_xlabel("difference = error_a − error_b")
ax.set_ylabel("Tienda (store_id)")
ax.set_title("Diferencia de error por tienda: error_a − error_b\n"
             "Derecha/azul favorece a B · Izquierda/rojo favorece a A · "
             "línea negra = referencia en cero")
ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)
ax.tick_params(axis="x", labelsize=9)

manejadores = [
    Patch(facecolor=COLOR_B, edgecolor="black", label="Favorece a B (difference > 0)"),
    Patch(facecolor=COLOR_A, edgecolor="black", label="Favorece a A (difference < 0)"),
    Line2D([0], [0], color="black", linewidth=2, label="Referencia en cero"),
]
ax.legend(handles=manejadores, loc="upper center", bbox_to_anchor=(0.5, -0.10),
          ncol=3, frameon=False, fontsize=9)

fig.tight_layout()
DIR_SALIDA.mkdir(parents=True, exist_ok=True)
fig.savefig(RUTA_PNG, dpi=120, bbox_inches="tight")
plt.close(fig)

if not RUTA_PNG.exists():
    raise RuntimeError(f"No se pudo guardar la figura en {RUTA_PNG}")

# ------------------------------------------------------------------ CSV exigido
df_out = df[["store_id", "error_a", "error_b", "difference", "favors"]]
df_out.to_csv(RUTA_CSV_SALIDA, index=False)

# ------------------------------------------------------------------ resultados.json
resultados = {
    "subtarea": "T3_figura_diferencias_por_tienda",
    "figura_png": str(RUTA_PNG),
    "csv_salida": str(RUTA_CSV_SALIDA),
    "n_tiendas": n_tiendas,
    "store_ids": store_ids,
    "n_favors_A": n_A,
    "n_favors_B": n_B,
    "n_empates": n_ties,
    "store_ids_favors_A": ids_A,
    "store_ids_favors_B": ids_B,
    "store_ids_empate": ids_tie,
    "diferencia_de_mayor_magnitud_favorece_B": max_B,
    "diferencia_de_mayor_magnitud_favorece_A": max_A,
    "difference_min": float(df["difference"].min()),
    "difference_max": float(df["difference"].max()),
    "difference_media": float(df["difference"].mean()),
    "tabla": df_out.to_dict(orient="records"),
    "consistencia_con_T2": consistencia_t2,
    "notas": {
        "signo": "difference = error_a − error_b; positivo favorece a B, "
                 "negativo favorece a A",
        "figura": "barras horizontales, una por tienda; dirección y color codifican "
                  "el signo, la longitud la magnitud; línea negra vertical en 0 "
                  "como referencia",
    },
}
with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------ resumen
print("T3 completada.")
print(f"Figura guardada en: {RUTA_PNG} ({n_tiendas} tiendas, una observación por tienda)")
print(f"CSV guardado en:    {RUTA_CSV_SALIDA}")
print(f"Tiendas que favorecen a A ({n_A}): {ids_A}")
print(f"Tiendas que favorecen a B ({n_B}): {ids_B}")
print(f"Empates: {n_ties}")
if max_B is not None:
    print(f"Mayor diferencia a favor de B: {max_B['store_id']} "
          f"(difference = {max_B['difference']:+.2f})")
if max_A is not None:
    print(f"Mayor diferencia a favor de A: {max_A['store_id']} "
          f"(difference = {max_A['difference']:+.2f})")
print(f"Consistencia con T2: {consistencia_t2.get('coincide')}")
