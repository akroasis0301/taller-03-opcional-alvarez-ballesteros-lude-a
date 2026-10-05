"""C5 — Procedencia: cada cifra del entregable sale de una ejecución o del enunciado.

Es la respuesta a la Parte 0.a. Usa la MISMA regla que evaluar_solver.py, para que lo que el
solver acepta sea lo que el evaluador acepta: se miran las cifras con dos o más decimales
(fuera de los bloques de código), se descartan las que trae el enunciado, y cada una debe
coincidir —dentro del redondeo de sus propios decimales— con algún número que escribió una
ejecución APROBADA (JSON, CSV, stdout). Las de intentos rechazados no respaldan nada.
"""
from __future__ import annotations

import json
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


def artefactos(carpetas: list[Path]) -> list[float]:
    valores: list[float] = []
    for c in carpetas:
        for ext in ("json", "csv", "txt", "log", "out"):
            for f in Path(c).rglob(f"*.{ext}"):
                if f.stat().st_size <= 5_000_000 and ".tmp" not in f.parts:
                    valores += [v for v, _, _ in numeros(f.read_text(encoding="utf-8", errors="replace"))]
        for nb in Path(c).rglob("*.ipynb"):
            datos = json.loads(nb.read_text(encoding="utf-8"))
            for celda in datos.get("cells", []):
                for o in celda.get("outputs", []):
                    texto = "".join(o.get("text", "")) + "".join(o.get("data", {}).get("text/plain", ""))
                    valores += [v for v, _, _ in numeros(texto)]
    return valores


def verificar(entregable_texto: str, enunciado: str, carpetas_aprobadas: list[Path]) -> dict:
    dados = {v for v, _, _ in numeros(enunciado)}
    propias = [(v, d, t) for v, d, t in numeros(sin_codigo(entregable_texto)) if d >= 2 and v not in dados]
    if not propias:
        return {"fraccion": 1.0, "total": 0, "sin_origen": []}
    medidos = artefactos(carpetas_aprobadas)
    sin_origen = sorted({t.strip() for v, d, t in propias
                         if not any(abs(a - v) <= 0.5 * 10 ** -d + 1e-9 for a in medidos)})
    respaldadas = sum(any(abs(a - v) <= 0.5 * 10 ** -d + 1e-9 for a in medidos) for v, d, _ in propias)
    return {"fraccion": respaldadas / len(propias), "total": len(propias), "sin_origen": sin_origen}
