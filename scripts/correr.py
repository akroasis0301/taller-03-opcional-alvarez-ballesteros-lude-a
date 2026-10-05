"""Corre el solver sobre UNA tarea y resume lo que pasó.

    uv run python scripts/correr.py solver-v2/enunciados/tarea-a-generativo-discriminativo.pdf
    uv run python scripts/correr.py tareas/reales/semana2/enunciado.pdf --salida corridas/prueba/semana2
    uv run python scripts/correr.py <pdf> --variante sin_grafo        # o sin_critico
    uv run python scripts/correr.py --diagrama                        # el orquestador en Mermaid

Con --preguntar, una descarga se confirma por consola (freno 4); sin él, se niega.
Para evaluar con el golden set, usa solver-v2/evaluar_solver.py (ver docs/interfaces.md).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from solver.config import Config  # noqa: E402
from solver.orquestador import Solver, SolverSinCritico, SolverSinGrafo  # noqa: E402
from solver.traza import leer  # noqa: E402

VARIANTES = {"completo": Solver, "sin_grafo": SolverSinGrafo, "sin_critico": SolverSinCritico}


def preguntar(motivo: str) -> bool:
    return input(f"\n⚠ {motivo}\n¿Autorizas la descarga? [s/N] ").strip().lower() == "s"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada", nargs="?", help="PDF del enunciado (o carpeta de un paquete)")
    ap.add_argument("--salida", help="carpeta de la corrida (por defecto corridas/prueba/<nombre>)")
    ap.add_argument("--variante", choices=VARIANTES, default="completo")
    ap.add_argument("--preguntar", action="store_true", help="confirmar descargas por consola")
    ap.add_argument("--diagrama", action="store_true", help="imprime el orquestador en Mermaid")
    a = ap.parse_args()

    cfg = Config(confirmar=preguntar) if a.preguntar else Config()
    if a.diagrama:
        print(Solver(cfg).diagrama())
        return 0
    if not a.entrada:
        ap.error("falta la entrada")
    # Una carpeta por corrida: la anterior queda intacta como evidencia (2.c).
    salida = Path(a.salida or RAIZ / "corridas" / "prueba" / Path(a.entrada).stem / time.strftime("%Y%m%d-%H%M%S"))
    print(f"Solver {a.variante} → {salida}  (puede tardar minutos: el modelo razona)")
    r = VARIANTES[a.variante](cfg).solve(a.entrada, str(salida))

    print(f"\nstatus: {r['status']}   modelo: {r['model']}   duración: {r.get('duracion_s')} s")
    if r.get("error") or r.get("motivo_parada"):
        print(f"motivo: {r.get('error') or r.get('motivo_parada')}")
    for s in r["subtareas"]:
        print(f"  {s['id']:<4} {s['tipo']:<10} {s['status']:<9} intentos={s['intentos']}")
    print(f"tokens: {r['usage']}")
    filas = leer(r["trace"])
    print("decisiones del código:", dict(Counter(f["decision"] for f in filas if f["tipo"] == "decision")))
    errores = [f for f in filas if f["tipo"] == "error"]
    for e in errores[:5]:
        print(f"  ERROR [{e['agente']}] {e['error'][:200]}")
    print(f"entregables: {r['entregables']}\ntraza: {r['trace']}")
    if r.get("procedencia"):
        print(f"procedencia: {json.dumps(r['procedencia'], ensure_ascii=False)}")
    return 0 if r["status"] != "fallido" else 1


if __name__ == "__main__":
    raise SystemExit(main())
