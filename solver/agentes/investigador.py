"""[4] Investigador — el contexto de cada subtarea, con citas. C2.

Con grafo (baseline): siempre van el preámbulo (datos, tablas, reglas generales), la sección
literal de la subtarea y el cierre de sus dependencias `depende_de`. Es lo que la 0.b mostró:
la Parte 1 llega a la Parte 3 por la arista, no por parecerse a la pregunta.
Sin grafo (ablación de la 2.b): los k fragmentos más similares por TF-IDF, y nada más.

Lo que viene de las subtareas previas (sus resultados.json) no es recuperación: es el flujo de
datos del plan, y llega igual en las dos variantes.
Con la capa 2 (solver/graphrag.py), el orquestador SUMA a esto la búsqueda local en el grafo
de entidades y en las notas del curso, y los resúmenes de comunidad (búsqueda global), todo
con su cita. La sección literal y sus dependencias van siempre, haya capa 2 o no.
"""
from __future__ import annotations

import json

import networkx as nx

from solver.agentes.indexador import dependencias

K_PLANO = 2


def _bloque(s: dict, motivo: str) -> str:
    return f"[{s['clave']} — {motivo}]\n{s['titulo']}\n{s['texto']}"


def _tablas(documento: dict) -> str:
    if not documento.get("tablas"):
        return ""
    salida = []
    for t in documento["tablas"]:
        filas = t["filas"]
        salida.append(f"[tabla, página {t['pagina']}]\n" + "\n".join(" | ".join(f) for f in filas))
    return "\n\n".join(salida)


def con_grafo(subtarea: dict, documento: dict, grafo: nx.DiGraph) -> tuple[str, list[str]]:
    por_clave = {s["clave"]: s for s in documento["secciones"]}
    propias = [c for c in subtarea.get("secciones") or [] if c in por_clave]
    deps = dependencias(grafo, propias)
    citas = ["Preámbulo"] + propias + deps
    bloques = [_bloque(por_clave["Preámbulo"], "preámbulo")] if "Preámbulo" in por_clave else []
    bloques += [_bloque(por_clave[c], "sección de la subtarea") for c in propias]
    bloques += [_bloque(por_clave[c], f"dependencia depende_de") for c in deps]
    if t := _tablas(documento):
        bloques.append(t)
    return "\n\n".join(bloques), citas


def plano(subtarea: dict, documento: dict, k: int = K_PLANO) -> tuple[str, list[str]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    secciones = documento["secciones"]
    consulta = f"{subtarea.get('objetivo', '')} {subtarea.get('criterio', '')}"
    textos = [f"{s['titulo']} {s['texto']}" for s in secciones]
    vec = TfidfVectorizer().fit(textos + [consulta])
    sim = cosine_similarity(vec.transform([consulta]), vec.transform(textos))[0]
    top = sorted(range(len(secciones)), key=lambda i: -sim[i])[:k]
    return ("\n\n".join(_bloque(secciones[i], f"similitud {sim[i]:.3f}") for i in top),
            [secciones[i]["clave"] for i in top])


def previos(subtarea: dict, plan: list[dict], max_car: int = 4000) -> str:
    """Los resultados aprobados de las subtareas de las que depende (flujo de datos del plan)."""
    por_id = {s["id"]: s for s in plan}
    salida = []
    for d in subtarea.get("depende_de") or []:
        s = por_id.get(d, {})
        if s.get("status") == "aprobada" and s.get("resultados") is not None:
            texto = json.dumps(s["resultados"], ensure_ascii=False)[:max_car]
            salida.append(f"[{d} — {s.get('objetivo', '')}] archivos en entrada/{d}/; "
                          f"resultados.json:\n{texto}")
        elif s:
            salida.append(f"[{d}] no tiene resultados aprobados (status: {s.get('status')})")
    return "\n\n".join(salida)


def investigar(subtarea: dict, documento: dict, grafo: nx.DiGraph | None, plan: list[dict],
               usar_grafo: bool = True) -> dict:
    texto, citas = con_grafo(subtarea, documento, grafo) if usar_grafo else plano(subtarea, documento)
    return {"texto": texto, "citas": citas, "previos": previos(subtarea, plan),
            "modo": "grafo" if usar_grafo else f"plano_top{K_PLANO}"}
