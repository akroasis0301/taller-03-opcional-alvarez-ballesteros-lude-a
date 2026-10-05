"""[3] Planificador — subtareas con tipo, dependencias y criterio. C1.

El LLM propone el plan; el CÓDIGO lo valida. Un plan inválido vuelve al planificador con la
lista de problemas, por su nombre. Lo que se comprueba:
  - ids únicos y no vacíos; tipo ∈ {calculo, conceptual};
  - dependencias que existen, sin autodependencias y sin ciclos (es un DAG);
  - cada sección de trabajo del enunciado cubierta por al menos una subtarea;
  - coherencia con el grafo: si la Parte 3 depende de la Parte 1, la subtarea que cubre la
    Parte 3 depende (directa o transitivamente) de la que calcula la Parte 1.
"""
from __future__ import annotations

import json

import networkx as nx

from solver.cliente_llm import ClienteLLM

TIPOS = {"calculo", "conceptual"}
MAX_SUBTAREAS = 12

SISTEMA = """Eres el planificador de un solver que resuelve tareas de una maestría en IA.
Recibes el enunciado dividido en secciones y devuelves un plan en JSON. Reglas:
- Cada subtarea cubre una o más secciones de trabajo, por su clave exacta (p. ej. "Parte 1").
- tipo "calculo": requiere ejecutar código (cargar datos, entrenar, medir, graficar). Toda cifra
  que pida el enunciado sale de una subtarea de cálculo: nunca se calcula de memoria.
- tipo "conceptual": solo explicación o discusión, sin código; la escribe el redactor después
  usando los resultados de las subtareas de las que depende.
- depende_de: ids de subtareas cuyos resultados necesita. Si una sección dice «la misma división
  de la Parte 1», la subtarea que la cubre depende de la que cubre la Parte 1.
- criterio: cómo se comprueba que quedó bien (qué cifras, tablas o figuras deben existir).
- No crees subtareas para redactar el reporte: eso lo hace el redactor.
Devuelve SOLO este JSON:
{"subtareas": [{"id": "T1", "tipo": "calculo", "secciones": ["Parte 1"], "depende_de": [],
  "objetivo": "...", "criterio": "..."}]}"""


def mensajes(documento: dict, aristas: list[tuple[str, str]], problemas: list[str] | None,
             plan_previo: dict | None) -> list[dict]:
    partes = [f"## {s['clave']}{' (de trabajo)' if s['trabajo'] else ''}\n{s['titulo']}\n{s['texto']}"
              for s in documento["secciones"]]
    usuario = ("ENUNCIADO POR SECCIONES:\n\n" + "\n\n".join(partes) +
               "\n\nDEPENDENCIAS ENTRE SECCIONES (extraídas del texto):\n" +
               ("\n".join(f"- {a} depende de {b}" for a, b in aristas) or "- ninguna") +
               "\n\nSECCIONES DE TRABAJO QUE DEBEN QUEDAR CUBIERTAS: " +
               ", ".join(s["clave"] for s in documento["secciones"] if s["trabajo"]))
    msgs = [{"role": "system", "content": SISTEMA}, {"role": "user", "content": usuario}]
    if problemas:
        msgs += [{"role": "assistant", "content": json.dumps(plan_previo, ensure_ascii=False)},
                 {"role": "user", "content": "El plan es inválido. Corrige estos problemas y devuelve "
                  "el plan completo:\n" + "\n".join(f"- {p}" for p in problemas)}]
    return msgs


def validar(plan: dict, documento: dict, grafo: nx.DiGraph) -> list[str]:
    problemas: list[str] = []
    subtareas = plan.get("subtareas") if isinstance(plan, dict) else None
    if not isinstance(subtareas, list) or not subtareas:
        return ["el plan no tiene una lista 'subtareas' con al menos una subtarea"]
    if len(subtareas) > MAX_SUBTAREAS:
        problemas.append(f"demasiadas subtareas ({len(subtareas)} > {MAX_SUBTAREAS})")
    ids = [str(s.get("id", "")).strip() for s in subtareas]
    if any(not i for i in ids):
        problemas.append("hay subtareas sin id")
    repetidos = sorted({i for i in ids if ids.count(i) > 1})
    if repetidos:
        problemas.append(f"ids repetidos: {repetidos}")
    claves = {s["clave"] for s in documento["secciones"]}
    dag = nx.DiGraph()
    dag.add_nodes_from(ids)
    for s, i in zip(subtareas, ids):
        if s.get("tipo") not in TIPOS:
            problemas.append(f"{i}: tipo {s.get('tipo')!r} no es 'calculo' ni 'conceptual'")
        for sec in s.get("secciones") or []:
            if sec not in claves:
                problemas.append(f"{i}: la sección {sec!r} no existe en el enunciado")
        for d in s.get("depende_de") or []:
            if d == i:
                problemas.append(f"{i} depende de sí misma")
            elif d not in ids:
                problemas.append(f"{i} depende de {d}, que no existe")
            else:
                dag.add_edge(d, i)
        if not (s.get("criterio") or "").strip():
            problemas.append(f"{i}: falta el criterio")
    if not nx.is_directed_acyclic_graph(dag):
        problemas.append(f"las dependencias forman un ciclo: {nx.find_cycle(dag)}")
        return problemas
    cubiertas = {sec for s in subtareas for sec in (s.get("secciones") or [])}
    faltan = [s["clave"] for s in documento["secciones"] if s["trabajo"] and s["clave"] not in cubiertas]
    if faltan:
        problemas.append(f"secciones de trabajo sin subtarea: {faltan}")
    # Coherencia con las aristas depende_de del grafo del enunciado.
    calculo_de = {}
    for s, i in zip(subtareas, ids):
        if s.get("tipo") == "calculo":
            for sec in s.get("secciones") or []:
                calculo_de.setdefault(sec, set()).add(i)
    for s, i in zip(subtareas, ids):
        for sec in s.get("secciones") or []:
            for _, dep in grafo.out_edges(sec) if sec in grafo else []:
                proveedores = calculo_de.get(dep, set()) - {i}
                if proveedores and not proveedores & (nx.ancestors(dag, i) | {i}):
                    problemas.append(f"{i} cubre {sec}, que depende de {dep} "
                                     f"(subtarea {sorted(proveedores)}), pero {i} no depende de ella")
    return problemas


def ordenar(plan: dict) -> list[dict]:
    """Subtareas en orden topológico estable (por id ante empates)."""
    subtareas = {s["id"]: s for s in plan["subtareas"]}
    dag = nx.DiGraph()
    dag.add_nodes_from(subtareas)
    dag.add_edges_from((d, i) for i, s in subtareas.items() for d in s.get("depende_de") or [])
    return [{**subtareas[i], "status": "pendiente", "intentos": 0}
            for i in nx.lexicographical_topological_sort(dag)]


def planificar(llm: ClienteLLM, documento: dict, grafo: nx.DiGraph,
               problemas: list[str] | None = None, plan_previo: dict | None = None) -> tuple[dict, list[str]]:
    aristas = [(u, v) for u, v, d in grafo.edges(data=True) if d.get("tipo") == "depende_de"]
    try:
        plan = llm.pedir_json("planificador", mensajes(documento, aristas, problemas, plan_previo))
    except ValueError as err:          # JSON imposible incluso tras corregir: es un problema más
        return {}, [f"el planificador no devolvió JSON válido: {err}"]
    return plan, validar(plan, documento, grafo)
