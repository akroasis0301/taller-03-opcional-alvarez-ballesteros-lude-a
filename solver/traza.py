"""C6 — La traza se escribe en todo camino.

Cada evento es una línea JSON en traza.jsonl, escrita y cerrada en el momento: si la
corrida muere a la mitad, todo lo anterior ya está en disco. Nada se guarda en memoria
esperando un «al final».

Tipos de evento (los campos que P2 suma en la 2.b y lee en la 2.c):
  llamada    agente, subtarea, modelo, tokens_entrada, tokens_salida, latencia_s, fin,
             intento, error
  ejecucion  agente, subtarea, intento, returncode, duracion_s, archivos, error
  decision   agente, subtarea, decision, motivo      (lo que decidió el CÓDIGO)
  error      agente, subtarea, error                 (excepciones capturadas)
  inicio / fin de corrida

Antes de escribir, todo valor de texto pasa por enmascarar(): una traza guarda mensajes
enteros y se sube al repositorio.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import defaultdict
from pathlib import Path

_PATRONES = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)(\s*[=:]\s*)['\"]?[^\s'\",]{8,}"),
]
_NOMBRES_SECRETOS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def _secretos_del_entorno() -> list[str]:
    return [v for k, v in os.environ.items()
            if any(s in k.upper() for s in _NOMBRES_SECRETOS) and len(v) >= 8]


def enmascarar(texto: str) -> str:
    for valor in _secretos_del_entorno():
        texto = texto.replace(valor, "***")
    for patron in _PATRONES:
        if patron.groups >= 2:
            texto = patron.sub(lambda m: f"{m.group(1)}{m.group(2)}***", texto)
        else:
            texto = patron.sub("***", texto)
    return texto


def _limpiar(valor):
    if isinstance(valor, str):
        return enmascarar(valor)
    if isinstance(valor, dict):
        return {k: _limpiar(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_limpiar(v) for v in valor]
    if isinstance(valor, Path):
        return str(valor)
    return valor


class Traza:
    def __init__(self, ruta: str | Path):
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._tokens = defaultdict(lambda: [0, 0])   # agente -> [entrada, salida]
        # El indexador llama al LLM en paralelo (capa 2): numerar, sumar y escribir es atómico.
        self._candado = threading.RLock()

    # ------------------------------------------------------------------ escritura
    def evento(self, tipo: str, **campos) -> dict:
        campos = _limpiar(campos)
        with self._candado:
            self._seq += 1
            fila = {"seq": self._seq, "ts": round(time.time(), 3), "tipo": tipo, **campos}
            with self.ruta.open("a", encoding="utf-8") as f:   # abrir-escribir-cerrar: sobrevive a un crash
                f.write(json.dumps(fila, ensure_ascii=False, default=str) + "\n")
        return fila

    def llamada(self, agente: str, *, modelo: str, tokens_entrada: int, tokens_salida: int,
                latencia_s: float, fin: str | None, intento: int = 1, subtarea: str | None = None,
                error: str | None = None, **extra) -> dict:
        with self._candado:
            self._tokens[agente][0] += tokens_entrada
            self._tokens[agente][1] += tokens_salida
            return self.evento("llamada", agente=agente, subtarea=subtarea, modelo=modelo,
                               tokens_entrada=tokens_entrada, tokens_salida=tokens_salida,
                               latencia_s=latencia_s, fin=fin, intento=intento, error=error, **extra)

    def ejecucion(self, agente: str, *, subtarea: str, intento: int, returncode: int | None,
                  duracion_s: float, archivos: list[str], error: str | None = None, **extra) -> dict:
        return self.evento("ejecucion", agente=agente, subtarea=subtarea, intento=intento,
                           returncode=returncode, duracion_s=duracion_s, archivos=archivos,
                           error=error, **extra)

    def decision(self, agente: str, decision: str, motivo: str = "",
                 subtarea: str | None = None, **extra) -> dict:
        return self.evento("decision", agente=agente, subtarea=subtarea, decision=decision,
                           motivo=motivo, **extra)

    def error(self, agente: str, error: BaseException | str, subtarea: str | None = None) -> dict:
        texto = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else error
        return self.evento("error", agente=agente, subtarea=subtarea, error=texto)

    # ------------------------------------------------------------------ lectura
    @property
    def tokens_por_agente(self) -> dict[str, dict[str, int]]:
        return {a: {"tokens_entrada": e, "tokens_salida": s} for a, (e, s) in self._tokens.items()}

    @property
    def tokens_totales(self) -> dict[str, int]:
        return {"tokens_entrada": sum(e for e, _ in self._tokens.values()),
                "tokens_salida": sum(s for _, s in self._tokens.values())}


def leer(ruta: str | Path) -> list[dict]:
    """Para P2: las filas de una traza, listas para un DataFrame."""
    return [json.loads(l) for l in Path(ruta).read_text(encoding="utf-8").splitlines() if l.strip()]
