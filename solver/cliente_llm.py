"""El cliente del LLM: la H200 del curso, con lo que un agente necesita alrededor.

No reescribimos el cliente HTTP: usamos solver-v2/h200.py, el del profesor, que ya lee el
id del modelo de /v1/models y separa el razonamiento. Aquí se añade lo que el solver exige:

  - Presupuesto (freno de la Parte 3): se comprueba ANTES de cada llamada. Los agentes
    de trabajo se detienen al llegar a la reserva; el redactor puede gastarla.
  - Contenido vacío: el modelo razona siempre y lo cobra del mismo cupo de salida. Si
    vuelve vacío con fin=length, se reintenta con el doble de max_tokens (hasta un tope).
    Nunca se devuelve una cadena vacía como si fuera una respuesta.
  - Traza (C6): cada llamada —también la que falla— deja su evento.
  - JSON: pedir_json() parsea y, si el modelo devuelve algo inválido, le muestra el error.
  - LLMGuion: un modelo falso con la misma interfaz, para tests y para forzar frenos.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from solver.config import RAIZ, Config
from solver.traza import Traza

AGENTE_CON_RESERVA = "redactor"
MAX_CARACTERES_MENSAJE = 20_000   # por mensaje en la traza: suficiente para la 2.c


class PresupuestoAgotado(RuntimeError):
    """El freno saltó: el orquestador debe ir al redactor con lo que tenga."""


class LLMVacio(RuntimeError):
    """El modelo devolvió contenido vacío incluso después de los reintentos."""


def cargar_h200(timeout_s: float = 600, modelo: str | None = None):
    """La H200 real. Falla con un mensaje claro si no hay VPN (lo hace h200.py).

    Por defecto, el modelo es el primero que devuelve /v1/models (el vLLM sirve uno solo).
    Con H200_MODELO se elige entre los que sirve el endpoint —el Ollama del 11434 sirve
    varios—, pero solo si de verdad está en esa lista: el id se comprueba, no se inventa.
    """
    kit = str(RAIZ / "solver-v2")
    if kit not in sys.path:
        sys.path.insert(0, kit)
    from h200 import H200  # noqa: E402  (vive en el kit, no es un paquete)
    h = H200(timeout=timeout_s)
    pedido = (modelo or os.environ.get("H200_MODELO", "")).strip()
    if pedido:
        servidos = [m["id"] for m in h._pedir(h.PUERTO, "v1/models", timeout=8)["data"]]
        if pedido not in servidos:
            raise RuntimeError(f"H200_MODELO={pedido!r} no está entre los modelos que sirve "
                               f"{h.HOST}:{h.PUERTO}: {servidos}")
        h.modelo = pedido
    return h


def es_timeout(err: BaseException) -> bool:
    """Un timeout no es una caída de red: repetir la misma llamada lenta con el mismo límite
    solo duplica la espera (corrida del 2026-10-04: 600 s + 600 s perdidos en la T2)."""
    return isinstance(err, TimeoutError) or "timed out" in str(err).lower()


_PENSAMIENTO = re.compile(r"<think>.*?</think>\s*", re.S)


def sin_pensamiento(texto: str) -> str:
    """Ollama (qwen3, gpt-oss) puede dejar el razonamiento dentro del contenido entre
    <think>…</think>. No es la respuesta: se quita antes de usarla o parsearla."""
    return _PENSAMIENTO.sub("", texto)


def _recortar(mensajes: list[dict]) -> list[dict]:
    salida = []
    for m in mensajes:
        c = m.get("content") or ""
        if len(c) > MAX_CARACTERES_MENSAJE:
            c = c[:MAX_CARACTERES_MENSAJE] + f"… [{len(c) - MAX_CARACTERES_MENSAJE} car. omitidos]"
        salida.append({**m, "content": c})
    return salida


class ClienteLLM:
    def __init__(self, cfg: Config, traza: Traza):
        self.cfg = cfg
        self.traza = traza
        self.llm = cfg.llm if cfg.llm is not None else cargar_h200(cfg.timeout_llm_s)
        self.modelo = getattr(self.llm, "modelo", "desconocido")

    # ------------------------------------------------------------------ presupuesto
    @property
    def consumido(self) -> int:
        t = self.traza.tokens_totales
        return t["tokens_entrada"] + t["tokens_salida"]

    def disponible(self, agente: str) -> int:
        limite = self.cfg.presupuesto_tokens
        if agente != AGENTE_CON_RESERVA:
            limite -= self.cfg.reserva_redactor
        return limite - self.consumido

    # ------------------------------------------------------------------ llamadas
    def chat(self, agente: str, mensajes: list[dict], *, subtarea: str | None = None,
             json_mode: bool = False, tools: list[dict] | None = None,
             max_tokens: int | None = None) -> dict:
        """Una respuesta no vacía del modelo, o una excepción. Nunca una cadena vacía."""
        max_tokens = max_tokens or self.cfg.max_tokens
        intento = vacios = red = 0
        while True:
            if self.disponible(agente) <= 0:          # freno: ANTES de la llamada
                self.traza.decision(agente, "presupuesto_agotado", subtarea=subtarea,
                                    motivo=f"consumido={self.consumido} de "
                                           f"{self.cfg.presupuesto_tokens} (reserva "
                                           f"{self.cfg.reserva_redactor})")
                raise PresupuestoAgotado(f"{agente}: presupuesto agotado ({self.consumido} tokens)")
            intento += 1
            t0 = time.perf_counter()
            try:
                r = self.llm.chat(mensajes, tools=tools, json_mode=json_mode, max_tokens=max_tokens)
            except Exception as err:                  # red, HTTP, VPN caída…
                self.traza.llamada(agente, modelo=self.modelo, tokens_entrada=0, tokens_salida=0,
                                   latencia_s=round(time.perf_counter() - t0, 2), fin=None,
                                   intento=intento, subtarea=subtarea,
                                   error=f"{type(err).__name__}: {err}")
                if not es_timeout(err) and red < self.cfg.reintentos_red:
                    red += 1
                    continue
                raise
            uso = r.get("uso") or {}
            crudo = r.get("contenido") or ""
            contenido = sin_pensamiento(crudo)
            r = {**r, "contenido": contenido}
            vacio = not contenido.strip() and not r.get("tool_calls")
            self.traza.llamada(
                agente, modelo=self.modelo, subtarea=subtarea, intento=intento,
                tokens_entrada=int(uso.get("prompt_tokens", 0)),
                tokens_salida=int(uso.get("completion_tokens", 0)),
                latencia_s=r.get("latencia_s", round(time.perf_counter() - t0, 2)),
                fin=r.get("fin"), max_tokens=max_tokens,
                error=f"contenido vacío (fin={r.get('fin')})" if vacio else None,
                entrada=_recortar(mensajes), salida=contenido,
                razonamiento_caracteres=len(r.get("razonamiento") or "") + len(crudo) - len(contenido))
            if not vacio:
                return r
            if vacios >= self.cfg.reintentos_vacio:
                raise LLMVacio(f"{agente}: contenido vacío tras {vacios + 1} intentos "
                               f"(fin={r.get('fin')}, max_tokens={max_tokens})")
            vacios += 1
            if r.get("fin") == "length":              # razonó hasta el tope: más espacio
                max_tokens = min(max_tokens * 2, self.cfg.max_tokens_tope)

    def pedir(self, agente: str, mensajes: list[dict], **kw) -> str:
        return self.chat(agente, mensajes, **kw)["contenido"]

    def pedir_json(self, agente: str, mensajes: list[dict], *, reintentos: int = 2, **kw) -> dict:
        """Un objeto JSON. Si el modelo devuelve algo inválido, se le muestra el error."""
        mensajes = list(mensajes)
        for i in range(reintentos + 1):
            texto = self.pedir(agente, mensajes, json_mode=True, **kw)
            try:
                return extraer_json(texto)
            except ValueError as err:
                self.traza.decision(agente, "json_invalido", motivo=str(err),
                                    subtarea=kw.get("subtarea"), intento=i + 1)
                if i == reintentos:
                    raise
                mensajes += [{"role": "assistant", "content": texto},
                             {"role": "user", "content": f"Tu respuesta no es JSON válido ({err}). "
                              "Devuelve solo el objeto JSON, sin texto alrededor."}]
        raise AssertionError("inalcanzable")


def extraer_json(texto: str) -> dict:
    """Tolera ```json … ``` y texto alrededor; exige un objeto."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto.strip())
    try:
        valor = json.loads(t)
    except json.JSONDecodeError:
        ini, fin = t.find("{"), t.rfind("}")
        if ini < 0 or fin <= ini:
            raise ValueError("no hay ningún objeto JSON en la respuesta") from None
        try:
            valor = json.loads(t[ini:fin + 1])
        except json.JSONDecodeError as err:
            raise ValueError(f"JSON mal formado: {err.msg} (posición {err.pos})") from None
    if not isinstance(valor, dict):
        raise ValueError(f"se esperaba un objeto y llegó {type(valor).__name__}")
    return valor


class LLMGuion:
    """Modelo de guion: misma interfaz que H200.chat, sin red ni costo.

    Cada respuesta del guion puede ser un str (el contenido), un dict (campos de la
    respuesta, p. ej. {"contenido": "", "fin": "length"}), una función de los mensajes,
    o una excepción que se lanza. La última respuesta se repite cuando el guion se acaba.
    """

    def __init__(self, respuestas: list, modelo: str = "guion",
                 tokens_entrada: int = 1_000, tokens_salida: int = 200):
        self.respuestas = list(respuestas)
        self.modelo = modelo
        self.tokens = (tokens_entrada, tokens_salida)
        self.llamadas: list[dict] = []

    def chat(self, mensajes, tools=None, json_mode=False, max_tokens=16384, temperature=0.0):
        self.llamadas.append({"mensajes": mensajes, "json_mode": json_mode, "max_tokens": max_tokens})
        r = self.respuestas.pop(0) if len(self.respuestas) > 1 else self.respuestas[0]
        if isinstance(r, BaseException):
            raise r
        if callable(r):
            r = r(mensajes)
        if isinstance(r, str):
            r = {"contenido": r}
        base = {"contenido": "", "tool_calls": [], "razonamiento": "", "fin": "stop",
                "uso": {"prompt_tokens": self.tokens[0], "completion_tokens": self.tokens[1]},
                "latencia_s": 0.0}
        return {**base, **r}
