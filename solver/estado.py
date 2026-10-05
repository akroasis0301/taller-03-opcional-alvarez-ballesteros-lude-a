"""El estado que viaja por el grafo del orquestador (LangGraph).

Regla: en el estado van datos pequeños y serializables (el checkpointer de la extensión D
los guarda). Lo pesado —el documento leído, el grafo, los artefactos de cada ejecución—
vive en disco dentro de la carpeta de la corrida, y el estado guarda su ruta.
"""
from __future__ import annotations

from typing import Literal, TypedDict

TipoSubtarea = Literal["calculo", "conceptual"]
StatusSubtarea = Literal["pendiente", "aprobada", "fallida", "omitida"]


class Subtarea(TypedDict, total=False):
    id: str                       # "T1", "T2"…
    tipo: TipoSubtarea
    secciones: list[str]          # claves del enunciado que cubre ("Parte 1", …)
    depende_de: list[str]         # ids de otras subtareas
    objetivo: str
    criterio: str                 # cómo sabe el crítico que está bien
    status: StatusSubtarea
    intentos: int
    carpeta: str                  # subtareas/T1/intento-N aprobado
    resultados: dict              # su resultados.json aprobado
    correccion: str               # la última devolución del crítico


class Estado(TypedDict, total=False):
    # entrada
    ruta_entrada: str
    carpeta_salida: str
    # [1] lector y [2] indexador
    documento: str                # cache_solver/enunciado_leido.json
    grafo: str                    # grafo.json
    restricciones: dict           # entregable, formato, palabras_max, paginas_max, secciones
    # [3] planificador
    plan: list[Subtarea]
    plan_previo: dict
    intentos_plan: int
    problemas_plan: list[str]
    # cola y subtarea actual
    cola: list[str]
    actual: str | None
    contexto: dict                # [4] investigador: texto, previos, citas, modo
    codigo: str                   # [5] programador
    ejecucion: dict               # [6] ejecutor
    rechazadas: dict              # huellas de scripts rechazados por subtarea (freno de repetición)
    # [8] redactor y procedencia
    borrador: str
    problemas_redaccion: list[str]
    intentos_redaccion: int
    procedencia: dict
    entregables: list[str]
    notas: list[str]
    # cierre
    status: Literal["completado", "parcial", "fallido"]
    motivo_parada: str            # "" | "presupuesto" | "error en <nodo>: …" | "plan inválido"
