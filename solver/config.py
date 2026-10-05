"""Configuración del solver: frenos, ablaciones y ganchos para el grupo.

Todo lo que P2 (ablaciones) y P3 (frenos, extensión) necesitan cambiar vive aquí, no
dentro de los agentes. Los valores por defecto se leen del .env; un test o una variante
los sobreescribe al construir el Config.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

RAIZ = Path(__file__).resolve().parents[1]

try:  # el .env es opcional: sin él valen los valores por defecto
    from dotenv import load_dotenv

    load_dotenv(RAIZ / ".env")
except ImportError:  # pragma: no cover
    pass


def _entero(nombre: str, defecto: int) -> int:
    try:
        return int(os.environ.get(nombre, defecto))
    except ValueError:
        return defecto


def negar_siempre(motivo: str) -> bool:
    """Confirmación humana por defecto: sin una persona delante, lo irreversible no corre."""
    return False


@dataclass
class Config:
    # --- Frenos (Parte 3, P3 los fuerza) ---
    # 500 k: con 300 k la Tarea C (r2) y la Semana 2 (exploratoria) se quedaron sin presupuesto con
    # subtareas pendientes; el modelo de la H200 razona siempre y lo cobra en la salida. La reserva
    # del redactor es el 10 %: el redactor de la Semana 2 gastó 36 k en una sola llamada.
    presupuesto_tokens: int = field(default_factory=lambda: _entero("SOLVER_PRESUPUESTO_TOKENS", 500_000))
    reserva_redactor: int = field(default_factory=lambda: _entero("SOLVER_RESERVA_REDACTOR", 50_000))
    max_intentos: int = field(default_factory=lambda: _entero("SOLVER_MAX_INTENTOS", 3))
    timeout_s: int = field(default_factory=lambda: _entero("SANDBOX_TIMEOUT_S", 120))
    # Recibe el motivo (p. ej. "fetch_openml en T2") y devuelve True si una persona aprueba.
    confirmar: Callable[[str], bool] = negar_siempre

    # --- Ablaciones (Parte 2.b, P2 las corre) ---
    usar_grafo: bool = True      # False: el investigador solo recibe los k fragmentos más similares
    usar_critico: bool = True    # False: un intento por script y ninguna comprobación

    # --- LLM ---
    max_tokens: int = 32_768             # el modelo de la H200 razona siempre y lo cobra aquí
    max_tokens_tope: int = 65_536        # hasta dónde se duplica si vuelve vacío por longitud
    reintentos_vacio: int = 2
    # Extraer entidades no necesita 32 k de razonamiento: en la Tarea C (r2, 2026-10-04) la
    # tabla del corpus razonó hasta 32 768 sin responder y hubo que repetirla con 65 536.
    # 16 384 cubre 43 de las 44 llamadas del índice de las notas y todas las de A, B y C
    # salvo las dos que se fueron al tope; esas se omiten en vez de costar 32 k + 65 k.
    max_tokens_indexador: int = field(default_factory=lambda: _entero("SOLVER_MAX_TOKENS_INDEXADOR", 16_384))
    reintentos_red: int = 1              # solo errores de conexión; un timeout NO se reintenta
    # Tiempo máximo de UNA llamada al LLM. h200.py usa 600 s; con el Ollama compartido,
    # una llamada del programador pasó de 10 min (corrida del 2026-10-04).
    timeout_llm_s: int = field(default_factory=lambda: _entero("H200_TIMEOUT_S", 900))
    # --- GraphRAG capa 2 (M2c) ---
    # SOLVER_CAPA2=0 la apaga (las pruebas viejas y los frenos de P3 no la necesitan).
    capa2: bool = field(default_factory=lambda: os.environ.get("SOLVER_CAPA2", "1") != "0")
    notas: Path | None = field(default_factory=lambda: Path(os.environ.get(
        "SOLVER_NOTAS", RAIZ / "conocimiento" / "notas-teoricas")))
    cache_graphrag: Path = field(default_factory=lambda: Path(os.environ.get(
        "SOLVER_CACHE_GRAPHRAG", RAIZ / "cache" / "graphrag")))
    hilos_indexador: int = field(default_factory=lambda: _entero("SOLVER_HILOS_INDEXADOR", 8))
    # El índice de las notas se construye una vez y se amortiza: tiene su propio presupuesto.
    presupuesto_notas: int = field(default_factory=lambda: _entero("SOLVER_PRESUPUESTO_NOTAS", 2_000_000))
    # Un objeto con .nombre y .embed(textos) -> matriz normalizada. None = bge-m3 de la H200
    # (con respaldo léxico si no responde).
    embedder: object | None = None

    # Un objeto con .modelo y .chat(...) como el de solver-v2/h200.py. None = la H200 real.
    # P3 inyecta aquí un modelo de guion para forzar los frenos sin gastar.
    llm: object | None = None
