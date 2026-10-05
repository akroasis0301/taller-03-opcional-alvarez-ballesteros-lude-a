"""Construye (una vez) el índice GraphRAG de las notas del curso y muestra qué quedó.

    uv run python scripts/indexar_notas.py            # usa la caché si ya existe
    uv run python scripts/indexar_notas.py --consulta "regresión logística estandarizada"

El solver lo construye solo la primera vez que lo necesita; este script sirve para pagarlo
antes de una corrida del evaluador (≈6 min con la H200) y para revisar las comunidades.
Su traza queda en cache/graphrag/notas-<clave>/traza_notas.jsonl. Necesita la VPN.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from solver import graphrag as G  # noqa: E402
from solver.cliente_llm import ClienteLLM  # noqa: E402
from solver.config import Config  # noqa: E402
from solver.traza import Traza  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--consulta", help="prueba la búsqueda global con esta consulta")
    a = ap.parse_args()

    cfg = Config()
    traza = Traza(Path(cfg.cache_graphrag) / f"indexar-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    cliente = ClienteLLM(cfg, traza)
    print(f"Modelo: {cliente.modelo} · notas: {cfg.notas} · hilos: {cfg.hilos_indexador}")
    t0 = time.perf_counter()
    indice = G.indice_notas(cliente, cfg, traza)
    if indice is None:
        print("No hay notas del curso.")
        return 1
    ents = indice["fusion"]["entidades"]
    print(f"\nÍndice {indice['clave']} ({time.perf_counter() - t0:.0f} s; construido el {indice['creado']} "
          f"en {indice['duracion_s']} s con {sum(indice['tokens'].values())} tokens)")
    print(f"  {len(indice['fragmentos'])} fragmentos · {len(ents)} entidades · "
          f"{len(indice['fusion']['relaciones'])} relaciones · {len(indice['comunidades'])} comunidades\n")
    for c in indice["comunidades"]:
        nombres = ", ".join(ents[m]["nombre"] for m in c["miembros"][:8])
        print(f"[comunidad {c['id']}] {len(c['miembros'])} entidades: {nombres}…\n  {c['resumen']}\n")

    embedder = G.elegir_embedder(cliente, cfg.embedder, traza)
    m = G._embeddings_notas(indice, embedder, cfg, traza)
    print(f"Embeddings: {embedder.nombre} · dimensión {m['entidades'].shape[1]}")
    if a.consulta:
        q = embedder.embed([a.consulta])[0]
        sim = m["comunidades"] @ q
        for i in sim.argsort()[::-1][:3]:
            print(f"  {sim[i]:.3f}  comunidad {indice['comunidades'][i]['id']}: {indice['comunidades'][i]['resumen'][:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
