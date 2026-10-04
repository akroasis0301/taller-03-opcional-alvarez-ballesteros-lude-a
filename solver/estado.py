"""El estado que viaja por el grafo del orquestador (LangGraph).

Regla: en el estado van datos pequeños y serializables (el checkpointer de la extensión D
los guarda). Lo pesado —el grafo de conocimiento, los embeddings, los artefactos de cada
ejecución— vive en disco dentro de la carpeta de la corrida, y el estado guarda su ruta.
"""
from __future__ import annotations

from typing import Literal, TypedDict

TipoSubtarea = Literal["calculo", "conceptual", "redaccion"]
StatusSubtarea = Literal["pendiente", "aprobada", "fallida", "omitida"]


class Subtarea(TypedDict, total=False):
    id: str                       # "T1", "T2"…
    tipo: TipoSubtarea
    secciones: list[str]          # nodos del enunciado que cubre ("Parte 1", …)
    depende_de: list[str]         # ids de otras subtareas
    objetivo: str
    criterio: str                 # cómo sabe el crítico que está bien
    status: StatusSubtarea
    intentos: int
    carpeta: str                  # corridas/tarea-X/subtareas/T1/
    resultados: dict              # el resultados.json aprobado
    correccion: str               # la última devolución del crítico


class Estado(TypedDict, total=False):
    # entrada
    ruta_entrada: str             # PDF o carpeta (tarea D)
    carpeta_salida: str
    # lector e indexador
    documento: str                # ruta a documento.json (páginas, secciones, tablas)
    grafo: str                    # ruta a grafo.json
    formato_entregable: str       # "md" | "pdf" | "ipynb"
    restricciones: dict           # palabras_max, paginas_max, secciones exigidas…
    # planificador
    plan: list[Subtarea]
    intentos_plan: int
    problemas_plan: list[str]
    # cola de trabajo
    cola: list[str]               # ids pendientes, en orden topológico
    actual: str | None
    # redactor y procedencia
    entregables: list[str]
    cifras_sin_origen: list[str]
    intentos_redaccion: int
    # cierre
    status: Literal["completado", "parcial", "fallido"]
    motivo_parada: str            # "cola_vacia" | "presupuesto" | "error: …"
