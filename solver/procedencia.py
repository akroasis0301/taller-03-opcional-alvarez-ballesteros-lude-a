"""C5 — Procedencia: cada cifra del entregable sale de una ejecución o del enunciado.

Es la respuesta a la Parte 0.a. Usa la MISMA regla que evaluar_solver.py, para que lo que el
solver acepta sea lo que el evaluador acepta: se miran las cifras con dos o más decimales
(fuera de los bloques de código), se descartan las que trae el enunciado, y cada una debe
coincidir —dentro del redondeo de sus propios decimales— con algún número que escribió una
ejecución APROBADA (JSON, CSV, stdout). Las de intentos rechazados no respaldan nada.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

NUM = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)(\s?%)?(?![\w])")   # la del evaluador


def sin_codigo(texto: str) -> str:
    return re.sub(r"```.*?```", "", texto, flags=re.S)


def numeros(texto: str) -> list[tuple[float, int, str]]:
    """(valor, decimales, texto original). Los porcentajes valen también como fracción."""
    salida = []
    for m in NUM.finditer(texto):
        crudo, pct = m.group(1).replace(",", "."), m.group(2)
        try:
            v = float(crudo)
        except ValueError:
            continue
        dec = len(crudo.split(".")[1]) if "." in crudo else 0
        if pct:
            salida.append((v / 100, dec + 2, m.group(0)))
        salida.append((v, dec, m.group(0)))
    return salida


# En los artefactos, las cifras pueden venir en notación científica (5.006898814879229e-05): la
# regla del evaluador no las lee, y sin esto el 5.0069 que el redactor escribió por 5.0069e-05
# (Tarea B, 2026-10-05: un error de cinco órdenes de magnitud) no tenía con qué compararse.
CIFRA_ARTEFACTO = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![\w.])")


def valores_artefacto(texto: str) -> list[float]:
    salida = []
    for m in CIFRA_ARTEFACTO.finditer(texto):
        try:
            salida.append(float(m.group(1)))
        except ValueError:
            continue
    return salida


def _salidas_notebook(nb: Path) -> str:
    datos = json.loads(Path(nb).read_text(encoding="utf-8"))
    return "\n".join("".join(o.get("text", "")) + "".join(o.get("data", {}).get("text/plain", ""))
                     for celda in datos.get("cells", []) for o in celda.get("outputs", []))


def artefactos(carpetas: list[Path], notebooks: list[Path] | tuple = ()) -> list[float]:
    """Lo que escribieron las ejecuciones APROBADAS. Los notebooks se pasan aparte: el entregable
    vive en la raíz de la corrida, y pasar la raíz como carpeta haría que los intentos RECHAZADOS
    (cache_solver/rechazados) respaldaran cifras."""
    valores: list[float] = []
    for c in carpetas:
        for ext in ("json", "csv", "txt", "log", "out"):
            for f in Path(c).rglob(f"*.{ext}"):
                if f.stat().st_size <= 5_000_000 and ".tmp" not in f.parts:
                    valores += valores_artefacto(f.read_text(encoding="utf-8", errors="replace"))
        for nb in Path(c).rglob("*.ipynb"):
            valores += valores_artefacto(_salidas_notebook(nb))
    for nb in notebooks:
        valores += valores_artefacto(_salidas_notebook(nb))
    return valores


def _cifras(texto: str) -> list[tuple[list[tuple[float, int]], str]]:
    """Cada cifra del texto con sus lecturas posibles: un porcentaje vale como fracción O como
    número. El evaluador del kit las cuenta como dos cifras distintas, y «39.53 %» siempre deja
    una sin respaldo aunque 0.3953 esté medido; aquí basta con que una lectura tenga respaldo."""
    salida = []
    for m in NUM.finditer(texto):
        crudo, pct = m.group(1).replace(",", "."), m.group(2)
        try:
            v = float(crudo)
        except ValueError:
            continue
        dec = len(crudo.split(".")[1]) if "." in crudo else 0
        lecturas = [(v / 100, dec + 2), (v, dec)] if pct else [(v, dec)]
        salida.append((lecturas, m.group(0).strip()))
    return salida


def _coincide(a: float, v: float, d: int) -> bool:
    return abs(a - v) <= 0.5 * 10 ** -d + 1e-9


def sugerir(v: float, d: int, medidos: list[float]) -> str | None:
    """Una pista concreta para el redactor: el valor medido más probable detrás de la cifra."""
    cerca = [a for a in medidos if abs(a - v) <= 1.5 * 10 ** -d]
    if cerca:
        a = min(cerca, key=lambda x: abs(x - v))
        return f"el medido es {a!r}: redondeado a {d} decimales es {round(a, d):.{d}f}"
    for a in medidos:
        if a > 0 and v > 0:
            k = round(math.log10(a / v))
            if k != 0 and _coincide(a / 10 ** k, v, d):
                return f"el medido es {a:.6g} (¿perdiste el factor 10^{k}?)"
    return None


def verificar(entregable_texto: str, enunciado: str, carpetas_aprobadas: list[Path],
              notebooks: list[Path] | tuple = ()) -> dict:
    dados = {v for lecturas, _ in _cifras(enunciado) for v, _ in lecturas}
    propias = [(lecturas, t) for lecturas, t in _cifras(sin_codigo(entregable_texto))
               if any(d >= 2 for _, d in lecturas) and not any(v in dados for v, _ in lecturas)]
    if not propias:
        return {"fraccion": 1.0, "total": 0, "sin_origen": [], "sugerencias": {}}
    medidos = artefactos(carpetas_aprobadas, notebooks)
    sin, sugerencias = [], {}
    for lecturas, t in propias:
        if not any(_coincide(a, v, d) for v, d in lecturas if d >= 2 for a in medidos):
            sin.append(t)
            v, d = next((v, d) for v, d in reversed(lecturas) if d >= 2)
            if (s := sugerir(v, d, medidos)) and t not in sugerencias:
                sugerencias[t] = s
    return {"fraccion": 1 - len(sin) / len(propias), "total": len(propias),
            "sin_origen": sorted(set(sin)), "sugerencias": sugerencias}


MARCA = "[cifra sin respaldo]"


def marcar(texto: str, cifras: list[str]) -> str:
    """Reemplaza cada cifra sin respaldo por una marca visible (F4): lo que no se midió no se publica."""
    for t in sorted(set(cifras), key=len, reverse=True):
        texto = re.sub(rf"(?<![\w.,]){re.escape(t)}(?![\w])", MARCA, texto)
    return texto
