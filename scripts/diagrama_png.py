"""El diagrama del orquestador como PNG, con las aristas condicionales y su condición.

    uv run python scripts/correr.py --diagrama > resultados/orquestador.mmd   # lo exporta LangGraph
    uv run python scripts/diagrama_png.py                                     # → resultados/orquestador.png

Las aristas salen del Mermaid que exporta LangGraph (no se escriben a mano): las punteadas son
condicionales (-.->). Las etiquetas con la condición vienen de los enrutadores de
solver/orquestador.py (_tras_planificar, _tras_criticar, _tras_verificar…).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

POS = {
    "__start__": (0, 10), "leer": (0, 9), "indexar": (0, 8), "planificar": (0, 7), "siguiente": (0, 5.5),
    "investigar": (2.3, 5.5), "programar": (4.6, 5.5), "ejecutar": (4.6, 4), "criticar": (2.3, 4),
    "redactar": (0, 2.5), "verificar": (2.3, 2.5), "cerrar": (0, 1), "__end__": (0, 0),
}
# Las aristas del flujo principal, con su condición (de los enrutadores) y dónde va la etiqueta.
PRINCIPALES = {
    ("planificar", "planificar"): ("plan inválido\n(< 3 intentos)", (0.95, 7.0)),
    ("planificar", "siguiente"): ("plan válido (DAG, C1)", (0.08, 6.25)),
    ("siguiente", "investigar"): ("hay subtarea", (1.15, 5.72)),
    ("investigar", "programar"): ("", None), ("programar", "ejecutar"): ("", None), ("ejecutar", "criticar"): ("", None),
    ("criticar", "programar"): ("rechazado\ny < tope", (3.6, 4.75)),
    ("criticar", "siguiente"): ("aprobado o tope", (1.05, 4.55)),
    ("siguiente", "redactar"): ("cola vacía", (0.08, 4.0)),
    ("redactar", "verificar"): ("", None),
    ("verificar", "redactar"): ("forma o procedencia\nfalla (< 3, C5)", (0.75, 3.15)),
    ("verificar", "cerrar"): ("publicable", (1.35, 1.45)),
    ("__start__", "leer"): ("", None), ("leer", "indexar"): ("", None), ("indexar", "planificar"): ("", None),
    ("cerrar", "__end__"): ("", None),
}
CURVA = {("criticar", "programar"): 0.3, ("criticar", "siguiente"): -0.25, ("verificar", "redactar"): 0.35,
         ("redactar", "verificar"): 0.0, ("verificar", "cerrar"): 0.2}
AGENTE = {"leer": "[1] Lector", "indexar": "[2] Indexador (GraphRAG)", "planificar": "[3] Planificador",
          "investigar": "[4] Investigador", "programar": "[5] Programador", "ejecutar": "[6] Ejecutor (sandbox)",
          "criticar": "[7] Crítico", "redactar": "[8] Redactor", "verificar": "verificar (procedencia)",
          "siguiente": "siguiente (cola)", "cerrar": "cerrar", "__start__": "inicio", "__end__": "fin"}


def aristas(mmd: str) -> list[tuple[str, str, bool]]:
    return [(a, b, "-.->" in flecha) for a, flecha, b in re.findall(r"^\s*(\w+)\s+(-\.->|-->)\s+(\w+);", mmd, re.M)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mmd", default=str(RAIZ / "resultados" / "orquestador.mmd"))
    ap.add_argument("--salida", default=str(RAIZ / "resultados" / "orquestador.png"))
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    es = aristas(Path(a.mmd).read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(11, 10))
    for n, (x, y) in POS.items():
        agente = n in {"leer", "indexar", "planificar", "investigar", "programar", "ejecutar", "criticar", "redactar"}
        ax.text(x, y, AGENTE[n], ha="center", va="center", fontsize=9, zorder=3,
                bbox=dict(boxstyle="round,pad=0.45", fc="#F58518" if agente else "#E8E8E8", ec="#444444"))
    secundarias = 0
    for u, v, cond in es:
        (x1, y1), (x2, y2) = POS[u], POS[v]
        principal = (u, v) in PRINCIPALES
        if u == v:
            ax.annotate("", xy=(x1 + 0.6, y1 + 0.12), xytext=(x1 + 0.6, y1 - 0.12),
                        arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-3", ls="--", color="#C0392B"))
        elif principal:
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15, shrinkA=20,
                                         shrinkB=20, connectionstyle=f"arc3,rad={CURVA.get((u, v), 0)}",
                                         ls="--" if cond else "-", color="#C0392B" if cond else "#222222", lw=1.4))
        else:                                   # presupuesto agotado o error: hacia redactar o cerrar
            secundarias += 1
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, shrinkA=20,
                                         shrinkB=20, connectionstyle="arc3,rad=0.12", ls=":", color="#9E9E9E", lw=0.8))
        etiqueta, pos = PRINCIPALES.get((u, v), ("", None))
        if etiqueta and pos:
            ax.text(*pos, etiqueta, fontsize=7.5, color="#C0392B", ha="left", va="center", zorder=4,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.4))
    ax.text(3.0, 0.4, f"Punteadas grises ({secundarias}): salidas por presupuesto agotado → [8] Redactor\n"
                      "(entrega con la reserva) o por error → cerrar. Rojas: condicionales del flujo.\n"
                      "Negras: aristas fijas. Aristas leídas del Mermaid que exporta LangGraph.",
            fontsize=7.5, color="#555555", ha="center", va="center")
    ax.set_xlim(-1.2, 5.8)
    ax.set_ylim(-0.4, 10.5)
    ax.axis("off")
    ax.set_title("Orquestador (LangGraph): ocho agentes; aristas condicionales punteadas en rojo", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.salida, dpi=150)
    print(f"→ {a.salida}  ({len(es)} aristas, {sum(c for *_, c in es)} condicionales)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
