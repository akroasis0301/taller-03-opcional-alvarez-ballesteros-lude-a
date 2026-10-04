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
    presupuesto_tokens: int = field(default_factory=lambda: _entero("SOLVER_PRESUPUESTO_TOKENS", 300_000))
    reserva_redactor: int = field(default_factory=lambda: _entero("SOLVER_RESERVA_REDACTOR", 30_000))
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
    reintentos_red: int = 1
    # Un objeto con .modelo y .chat(...) como el de solver-v2/h200.py. None = la H200 real.
    # P3 inyecta aquí un modelo de guion para forzar los frenos sin gastar.
    llm: object | None = None
