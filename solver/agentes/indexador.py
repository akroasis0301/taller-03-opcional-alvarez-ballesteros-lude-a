"""[2] Indexador (GraphRAG) — el grafo del enunciado. C2.

Capa 1, el esqueleto determinístico (M1): cada sección es un nodo y cada referencia cruzada
(«la misma división de la Parte 1», «la pregunta anterior») es una arista `depende_de` que
extrae una REGLA, no el modelo: referencias explícitas («Parte 1»), «la pregunta anterior» y
anáforas («los dos modelos», «las mismas consultas») que remiten a la sección de trabajo previa. Es la arista que la Parte 0.b mostró que el RAG plano pierde.

Capa 2 (solver/graphrag.py): entidades y relaciones del LLM, fusionadas con las notas del
curso por nombre normalizado, embeddings para las semillas y comunidades con resumen.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import networkx as nx

REFERENCIA = re.compile(r"\b(Parte|Tarea|Pregunta|Ejercicio|Problema)\s+(\d+)\b", re.I)
ANTERIOR = re.compile(r"\b(?:la|el)\s+(?:parte|pregunta|tarea|ejercicio|punto|literal)\s+anterior\b", re.I)
# Anáfora: «entrena los dos modelos», «con las mismas consultas», «ambos clasificadores» remiten a
# lo definido antes sin nombrar la parte (corrida del 2026-10-04: la T3 no recibió la Parte 2).
ANAFORA = re.compile(r"\b(?:(?:los|las)\s+(?:dos|tres|mismos|mismas)|ambos|ambas)\s+"
                     r"(?:modelos?|clasificadores?|m[eé]todos?|algoritmos?|consultas?|datos|juicios|"
                     r"distribuci[oó]n(?:es)?|m[eé]tricas?|par[aá]metros|redes|estimadores)\b", re.I)


def esqueleto(secciones: list[dict]) -> nx.DiGraph:
    g = nx.DiGraph()
    claves = {s["clave"] for s in secciones}
    for s in secciones:
        g.add_node(s["clave"], titulo=s["titulo"], trabajo=s["trabajo"], id=s["id"], capa="enunciado")
    trabajo_previo = None
    for s in secciones:
        if s["clave"] == "Preámbulo":
            continue
        for m in REFERENCIA.finditer(s["texto"]):
            destino = f"{m.group(1).capitalize()} {m.group(2)}"
            if destino in claves and destino != s["clave"]:
                g.add_edge(s["clave"], destino, tipo="depende_de", regla="referencia",
                           evidencia=_frase(s["texto"], m.start()))
        if trabajo_previo and (m := ANTERIOR.search(s["texto"])):
            g.add_edge(s["clave"], trabajo_previo, tipo="depende_de", regla="anterior",
                       evidencia=_frase(s["texto"], m.start()))
        if trabajo_previo and s["trabajo"] and (m := ANAFORA.search(s["texto"])) \
                and not g.has_edge(s["clave"], trabajo_previo):
            g.add_edge(s["clave"], trabajo_previo, tipo="depende_de", regla="anafora",
                       evidencia=_frase(s["texto"], m.start()))
        if s["trabajo"]:
            trabajo_previo = s["clave"]
    return g


def _frase(texto: str, pos: int, radio: int = 60) -> str:
    return " ".join(texto[max(0, pos - radio): pos + radio].split())


def dependencias(g: nx.DiGraph, claves: list[str]) -> list[str]:
    """Cierre transitivo de `depende_de` desde unas secciones, en orden de documento."""
    solo_dep = nx.subgraph_view(g, filter_edge=lambda u, v: g.edges[u, v].get("tipo") == "depende_de")
    alcanzadas: set[str] = set()
    for c in claves:
        if c in g:
            alcanzadas |= nx.descendants(solo_dep, c)
    orden = list(g.nodes)
    return sorted(alcanzadas - set(claves), key=orden.index)


def guardar(g: nx.DiGraph, carpeta: Path) -> tuple[Path, Path | None]:
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "grafo.json"
    ruta.write_text(json.dumps(nx.node_link_data(g, edges="aristas"), ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return ruta, dibujar(g, carpeta / "grafo.png")


def cargar(ruta: Path) -> nx.DiGraph:
    return nx.node_link_graph(json.loads(Path(ruta).read_text(encoding="utf-8")), edges="aristas")


def dibujar(g: nx.DiGraph, destino: Path) -> Path | None:
    """El grafo del enunciado con las aristas depende_de visibles (entregable de la Parte 1)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    nodos = [n for n, d in g.nodes(data=True) if d.get("capa") == "enunciado"]
    sub = g.subgraph(nodos)
    pos = {n: (i, 0 if sub.nodes[n].get("trabajo") else -0.6) for i, n in enumerate(nodos)}
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(nodos)), 3.2))
    colores = ["#4C78A8" if sub.nodes[n].get("trabajo") else "#BBBBBB" for n in sub]
    nx.draw_networkx_nodes(sub, pos, node_color=colores, node_size=1400, ax=ax)
    nx.draw_networkx_labels(sub, pos, labels={n: n.replace(" ", "\n", 1) for n in sub},
                            font_size=8, font_color="white", ax=ax)
    aristas = [(u, v) for u, v, d in sub.edges(data=True) if d.get("tipo") == "depende_de"]
    nx.draw_networkx_edges(sub, pos, edgelist=aristas, edge_color="#E45756", width=2,
                           arrows=True, arrowsize=18, connectionstyle="arc3,rad=0.35",
                           node_size=1400, ax=ax)
    ax.set_title("Grafo del enunciado — aristas depende_de (rojo), extraídas por regla", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return destino
